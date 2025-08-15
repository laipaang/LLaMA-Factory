from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2MLP, KwargsForCausalLM, Qwen2Model, Qwen2RMSNorm, Qwen2RotaryEmbedding ,Qwen2Config, Qwen2DecoderLayer
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import StaticCache, SlidingWindowCache
from transformers.utils.generic import ModelOutput
from typing import Callable, Optional, Tuple, Union, Dict, Any
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

@dataclass
class DenseRetrievalCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs with retrieval vectors.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head.
        past_key_values (`Cache`, *optional*):
            Cached hidden states for fast autoregressive decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        dr_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, dr_dim)`):
            Dense retrieval vectors for each token.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of attention weights.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    dr_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    '''
        Get the last valid token's hidden states
    '''
    # https://huggingface.co/BAAI/bge-multilingual-gemma2
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class NluHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            raise ValueError("hidden_size is not defined in config")

        if hasattr(config, "jm63_dim"):
            jm63_dim = config.jm63_dim
        else:
            jm63_dim = 4

        if hasattr(config, "tw_dim"):
            tw_dim = config.tw_dim
        else:
            tw_dim = 3

        self.next_sent_feat_linear = nn.Linear(hidden_size, hidden_size)
        self.jm63_linear = nn.Linear(hidden_size, jm63_dim)
        self.tw_linear = nn.Linear(hidden_size, tw_dim)

        #init
        nn.init.trunc_normal_(self.next_sent_feat_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.next_sent_feat_linear.bias, 0.0)
        nn.init.trunc_normal_(self.jm63_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.jm63_linear.bias, 0.0)
        nn.init.trunc_normal_(self.tw_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.tw_linear.bias, 0.0)

    def forward(self, hidden_states):
        next_sent_feat = torch.tanh(self.next_sent_feat_linear(hidden_states))
        reward_logits = self.jm63_linear(next_sent_feat)  # [batch_size, 2]
        tw_logits = self.tw_linear(next_sent_feat)
        return reward_logits, tw_logits

class DenseRetrievalHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            raise ValueError(f'hidden_size not found in config.')            
        if hasattr(config, "dr_dim"):
            dr_dim = config.dr_dim
        else:
            raise ValueError(f'dr_dim not found in config.')            

        self.dr_next_sent_feat_linear = nn.Linear(hidden_size, hidden_size)
        self.dr_linear = nn.Linear(hidden_size, dr_dim)
        # self.silu = nn.SiLU()
        self.tanh = nn.Tanh()

    def forward(self, hidden_states):
        output = self.dr_next_sent_feat_linear(hidden_states)
        output = self.tanh(output)
        output = self.dr_linear(output)
        output = self.tanh(output)
        output = output / output.norm(p=2, dim=-1, keepdim=True)
        return output

def compute_energy(emb_cls_a, emb_cls_b, temperature):
    """compute_energy
    """
    energy_logits = torch.matmul(emb_cls_a, emb_cls_b.T) 
    energy_logits = energy_logits / temperature
    energy = -temperature * torch.logsumexp(energy_logits, 1)
    return energy

def compute_hingle_loss(insample_energy, outsample_energy, m_in, m_out, mask):
    """compute_hingle_loss
    """
    relu = nn.ReLU()
    loss = torch.mean(torch.pow(relu(insample_energy - m_in), 2) * mask) + \
            torch.mean(torch.pow(relu(m_out - outsample_energy), 2) * mask)
    # print("insample_energy:", insample_energy)
    # print("outsample_energy:", outsample_energy)
    return loss


class Qwen2ForCausalLMPNLDenseRetrievalTAP(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.dr_margin = getattr(config, 'dr_margin', 1)
        self.dr_weight = getattr(config, 'dr_weight', 1)
        self.use_dense_retrieval = getattr(config, 'use_dense_retrieval', False)
        self.dr_temperature = getattr(config, 'dr_temperature', 1)
        self.is_sft = getattr(config, 'is_sft', 1)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.context_feature_model = DenseRetrievalHead(config=config)
        self._init_dr()
        self.nlu_head = NluHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_dr(self):
        nn.init.trunc_normal_(self.context_feature_model.dr_next_sent_feat_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.context_feature_model.dr_next_sent_feat_linear.bias, 0.0)
        nn.init.trunc_normal_(self.context_feature_model.dr_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.context_feature_model.dr_linear.bias, 0.0)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        dr_slice = None,
        is_dr = 0,
        is_use_sft_loss=None,
        cls_soft_label=None,
        is_use_cls_loss=None,
        tw_soft_label=None,
        is_use_tw_loss=None,
        sample_length=None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # nlu 
        sample_length_expanded = sample_length.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        next_sent_feat = torch.gather(hidden_states, dim=1, index=sample_length_expanded).squeeze(1)
        reward_logits, tw_logits = self.nlu_head(next_sent_feat)   
        #probs, tw_probs, logits
        eps = 1e-10
        device = reward_logits.device if hasattr(reward_logits, 'device') else 'cpu'
        loss = torch.tensor(0.0, device=device)
        lm_loss = torch.tensor(0.0, device=device)
        reward_loss = torch.tensor(0.0, device=device)
        tw_loss = torch.tensor(0.0, device=device)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        #print("logits:", logits.size())
        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        
        if dr_slice is not None:
            batch_size = logits.size(0) 
            batch_indices = torch.arange(batch_size).unsqueeze(-1).expand(-1, 5)  # shape [batch_size, 5]
            dr_logits = hidden_states[batch_indices, dr_slice - 1, :]
            dr_logits = self.context_feature_model(dr_logits) # batchsize, seq_len, hiddensize
            # len(source_ids),
            # len(input_ids) + len(dr_tgt_positive_ids), 
            # len(input_ids) + len(dr_tgt_positive_ids) + len(dr_tgt_hardneg_lp_ids), 
            # len(input_ids) + len(dr_tgt_positive_ids) + len(dr_tgt_hardneg_lp_ids) + len(dr_tgt_hardneg_bidword_ids), 
            # len(input_ids) + len(dr_tgt_positive_ids) + len(dr_tgt_hardneg_lp_ids) + len(dr_tgt_hardneg_bidword_ids) + len(dr_ood_query_ids)
            query_embedding = dr_logits[:, 0, :].squeeze(dim=1).contiguous()
            ad_positive_embedding = dr_logits[:, 1, :].squeeze(dim=1).contiguous()
            ad_hardneg_lp_embedding = dr_logits[:, 2, :].squeeze(dim=1).contiguous()
            ad_hardneg_bidword_embedding = dr_logits[:, 3, :].squeeze(dim=1).contiguous()
            query_ood_embedding = dr_logits[:, 4, :].squeeze(dim=1).contiguous()

            # # batch random negatives across gpus
            # assert torch.distributed.is_initialized(), "Distributed not initialized!"
            # global_rank = torch.distributed.get_rank()
            # world_size = torch.distributed.get_world_size()
            # # query_embedding_list = [torch.zeros_like(query_embedding) for _ in range(world_size)]
            # # query_ood_embedding_list = [torch.zeros_like(query_ood_embedding) for _ in range(world_size)]
            # ad_positive_embedding_list = [torch.zeros_like(ad_positive_embedding) for _ in range(world_size)]
            # # ad_hardneg_lp_embedding_list = [torch.zeros_like(ad_hardneg_lp_embedding) for _ in range(world_size)]
            # # ad_hardneg_bidword_list = [torch.zeros_like(ad_hardneg_bidword) for _ in range(world_size)]
            # # is_use_sft_loss_list = [torch.zeros_like(is_use_sft_loss) for _ in range(world_size)]
            # # is_dr_list = [torch.zeros_like(is_dr) for _ in range(world_size)]
            # # torch.distributed.all_gather(query_embedding_list, query_embedding)
            # # torch.distributed.all_gather(query_ood_embedding_list, query_ood_embedding)
            # torch.distributed.all_gather(ad_positive_embedding_list, ad_positive_embedding)
            # # torch.distributed.all_gather(ad_hardneg_lp_embedding_list, ad_hardneg_lp_embedding)
            # # torch.distributed.all_gather(ad_hardneg_bidword_list, ad_hardneg_bidword)
            # # torch.distributed.all_gather(is_use_sft_loss_list, is_use_sft_loss)
            # # torch.distributed.all_gather(is_dr_list, is_dr)
            
            
            # #print ("drtap global_rank",global_rank)
            # #print ("drtap ori:",next_sent_feat_query1)
            # #print ("drtap rank_data:", next_sent_feat_query1_list[global_rank])
            # #print ("drtap dp_rank:",dp_rank)
            # #critical, use current_rank vector, because all_gather op not support gradient
            # # query_embedding_list[global_rank] = query_embedding
            # # query_ood_embedding_list[global_rank] = query_ood_embedding
            # ad_positive_embedding_list[global_rank] = ad_positive_embedding
            # # ad_hardneg_lp_embedding_list[global_rank] = ad_hardneg_lp_embedding
            # # ad_hardneg_bidword_list[global_rank] = ad_hardneg_bidword
            # #cross-batch concat
            # # query_embedding = torch.cat(query_embedding_list, 0)
            # # query_ood_embedding = torch.cat(query_ood_embedding_list, 0)
            # ad_positive_embedding = torch.cat(ad_positive_embedding_list, 0)
            # # ad_hardneg_lp_embedding = torch.cat(ad_hardneg_lp_embedding_list, 0)
            # # ad_hardneg_bidword = torch.cat(ad_hardneg_bidword_list, 0)
            # # is_use_sft_loss = torch.cat(is_use_sft_loss_list, 0)
            # # is_dr = torch.cat(is_dr_list, 0)
            # #####################################################
            #(batchsize, dim) x (batchsize*gpu_num ,dim)-> batchsize, batchsize*gpu_num
            dr_matmul = torch.matmul(query_embedding, torch.transpose(ad_positive_embedding, 0, 1))
            #query_embedding @ ad_positive_embedding.T
            dr_hardlp_sim = torch.sum(torch.mul(query_embedding, ad_hardneg_lp_embedding), dim=1, keepdim=True)
            dr_hardbidword_sim = torch.sum(torch.mul(query_embedding, ad_hardneg_bidword_embedding), dim=1, keepdim=True)
           
            
            margin = torch.full(size=(dr_matmul.size(0),), fill_value=self.dr_margin, device=dr_matmul.device)
            margin = torch.diag(margin)
            dr_preds = dr_matmul - margin
            dr_preds = torch.cat((dr_preds, dr_hardlp_sim, dr_hardbidword_sim), 1) #(bs, bs+2)
            dr_preds = dr_preds / self.dr_temperature
            # build dr_labels
            dr_labels = torch.arange(0, batch_size, 1, dtype=torch.int64, device=dr_logits.device)
            dr_celoss = F.cross_entropy(dr_preds, dr_labels, reduction='none')
            dr_celoss = (dr_celoss * is_dr).mean() * self.dr_weight

            # free energy 
            #temperature = 0.05 
            insample_energy = compute_energy(query_embedding, ad_positive_embedding, 0.05)
            outsample_energy = compute_energy(query_ood_embedding, ad_positive_embedding, 0.05)
            # print("insample_energy:", insample_energy)
            # print("outsample_energy:", outsample_energy)
            # tmp = 0.05 / log(batch_size) = 0.05 * math.log(32, 10) = 0.075
            # margin+ = -0.075 * (1 + 0.5 / 0.075) = -0.6732867951399863
            # margin- = -0.075 * (1 + 0.2 / 0.075) = -0.37328679513998636
            # m_in = -0.575 #-100.0
            # m_out = -0.275 #-28.0
            m_in = -0.64 
            m_out = -0.14
            energy_loss = compute_hingle_loss(insample_energy, outsample_energy, m_in, m_out, is_dr)


        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                reduction='none',
                ignore_index=-100  # 假设用-100表示padding
            )
            loss_per_token = loss_per_token.view(shift_labels.shape)    # 还原成 (bsz, seq_len)
            lm_loss = (loss_per_token.sum(dim=1) / (shift_labels != -100).sum(dim=1)).unsqueeze(-1)  # 对 seq_len 取 Mean, 忽略 ignore_index
            is_use_sft_loss = is_use_sft_loss if is_use_sft_loss is not None else torch.tensor(1.0)
            lm_loss = torch.multiply(lm_loss, is_use_sft_loss)

        # 处理 reward_loss
        # if cls_soft_label is not None:
        #     log_probs = F.log_softmax(reward_logits, dim=-1)
        #     reward_loss = F.kl_div(log_probs, cls_soft_label, reduction='none').sum(dim=1, keepdim=True)  # (bsz, 1)
        #     is_use_cls_loss = is_use_cls_loss if is_use_cls_loss is not None else torch.tensor(0.0)
        #     reward_loss = torch.multiply(reward_loss, is_use_cls_loss)

        # 处理 tw_loss
        if tw_soft_label is not None:
            softmax = nn.Softmax(dim=1)
            tw_softmax = softmax(tw_logits)
            epsl = 1e-10
            tw_softmax = torch.clamp(tw_softmax, min=epsl, max=1-epsl) 
            tw_logsoftmax = torch.log(tw_softmax)
            #log_tw_probs = F.log_softmax(tw_logits, dim=-1)
            #tw_loss = F.kl_div(log_tw_probs, tw_soft_label, reduction='none').sum(dim=1, keepdim=True)  # (bsz, 1)
            tw_loss = -(tw_soft_label*tw_logsoftmax).sum(dim=1, keepdim=True) #/ (tw_soft_label).sum(dim=1, keepdim=True)

            is_use_tw_loss = is_use_tw_loss if is_use_tw_loss is not None else torch.tensor(0.0)
            tw_loss = torch.multiply(tw_loss, is_use_tw_loss)

        # 计算总损失
        loss = lm_loss + dr_celoss + tw_loss + 5.0 * energy_loss
        print("lm_loss", lm_loss.mean(), 'dr_celoss:', dr_celoss.mean(), 'tw_loss:', tw_loss.mean(), 'energy_loss:', energy_loss.mean())
        # loss = lm_loss + dr_celoss + reward_loss + tw_loss + 5.0 * energy_loss
        # print("lm_loss", lm_loss.mean(), 'dr_celoss:', dr_celoss.mean(), 'reward_loss:', reward_loss.mean(), 'tw_loss:', tw_loss.mean(), 'energy_loss:', energy_loss.mean())
        loss = loss.mean()

        return CausalLMOutputWithPast(
            loss=loss if loss != 0 else None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ =='__main__':
    # model_path = '/Users/chenlei45/projects/transformers/qwen_model'
    model_path = '/root/paddlejob/workspace/env_run/Qwen2.5'
    model = Qwen2ForCausalLMPNLDenseRetrievalTAP.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 测试输入
    prompt = "阿里巴巴是一家什么公司？"
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # 处理输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generate_kwargs = {
        "max_new_tokens": 10,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "return_dict_in_generate": True,
        "output_hidden_states": True
    }

    # 执行推理
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
    # dr_embedding = last_token_pool(full_output.last_hidden_state, model_inputs['attention_mask'])
    # print("dr_embedding", full_output.dr_hidden_states.shape, full_output.dr_hidden_states)
