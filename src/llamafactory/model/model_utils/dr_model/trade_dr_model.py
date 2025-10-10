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
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

def compute_energy(emb_a, emb_b, temperature):
    logits = emb_a @ emb_b.T
    logits = logits / temperature
    energy = -temperature * torch.logsumexp(logits, dim=1, keepdim=True)
    return energy.squeeze()

def compute_hingle_loss(insample_energy, outsample_energy, m_in, m_out, is_dr):
    # print(f'hingle_loss shape: {torch.pow(F.relu(insample_energy - m_in), 2).shape}, is_dr shape: {is_dr.shape}')
    loss_term1 = (torch.pow(F.relu(insample_energy - m_in), 2) * is_dr.squeeze()).mean()
    loss_term2 = (torch.pow(F.relu(m_out - outsample_energy), 2) * is_dr.squeeze()).mean()
    return loss_term1 + loss_term2

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
        # output = output / output.norm(p=2, dim=-1, keepdim=True)
        return output
        
class TradeQwenDR(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.dr_margin = getattr(config, 'dr_margin', 0)
        self.dr_weight = getattr(config, 'dr_weight', 1)
        self.use_dense_retrieval = getattr(config, 'use_dense_retrieval', False)
        self.dr_temperature = getattr(config, 'dr_temperature', 1)
        self.is_sft = getattr(config, 'is_sft', 1)
        self.use_cross_device_batch_negatives = getattr(config, 'use_cross_device_batch_negatives', False)

        # ood
        self.m_in = getattr(config, 'm_in', -0.64)
        self.m_out = getattr(config, 'm_out', -0.14)
        self.ood_temperature = getattr(config, 'ood_temperature', 0.05)
        self.energy_loss_weight = getattr(config, 'energy_loss_weight', 20)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 分布式相关属性
        if self.use_cross_device_batch_negatives and self.use_dense_retrieval:
            self.is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
            if self.is_distributed == False:
                self.use_cross_device_batch_negatives = False
                print("WARNING: use_cross_device_batch_negatives is set to True but distributed training is not initialized.")
            self.rank = dist.get_rank() if self.is_distributed else 0
            self.world_size = dist.get_world_size() if self.is_distributed else 1
            self.process_rank = dist.get_rank()

        if self.use_dense_retrieval:
            self.dr_dim = getattr(config, 'dr_dim', 128)
            self.context_feature_model = DenseRetrievalHead(config=config)
            self._init_dr()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_dr(self):
        nn.init.trunc_normal_(self.context_feature_model.dr_next_sent_feat_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.context_feature_model.dr_next_sent_feat_linear.bias, 0.0)
        nn.init.trunc_normal_(self.context_feature_model.dr_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.context_feature_model.dr_linear.bias, 0.0)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        """Gather a tensor from all processes in a distributed setting.

        Args:
            t (Optional[torch.Tensor]): The input tensor to be gathered. If `None`, no gathering is performed.

        Returns:
            Union[torch.Tensor, None]: A concatenated tensor from all processes if ``t`` is not ``None``, 
                otherwise returns ``None``.
        """
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def distributed_in_batch_negatives(self, query_embedding, ad_embedding, dr_margin, dr_temperature, dr_weight, is_dr, dim):
        """
        分布式环境下的In-Batch Negatives计算
        """

        from acclgr.ops import GlobalNegativeSampling

        B, device = query_embedding.size(0), query_embedding.device
        labels = torch.arange(B, device=device)
        idx_2d = [torch.arange(B, device=device), labels]
        is_dr = is_dr.squeeze()
        labels[is_dr == 0] = -100 # ignore

        global_ad_embedding, mask, labels = GlobalNegativeSampling.apply(ad_embedding, labels)
        logits = query_embedding @ global_ad_embedding.T

        logits[idx_2d] -= dr_margin
        logits /= dr_temperature
        logits = logits.masked_fill(mask, float('-inf'))

        loss = F.cross_entropy(logits, labels, ignore_index=-100)

        return loss * dr_weight

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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        sft_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        dr_celoss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        energy_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        if self.use_dense_retrieval:
            if dr_slice is not None:
                batch_size = logits.size(0) 
                batch_indices = torch.arange(batch_size).unsqueeze(-1).expand(-1, 3)  # shape [batch_size, 2]
                dr_logits = hidden_states[batch_indices, dr_slice - 1, :]
                dr_logits = self.context_feature_model(dr_logits)
                
                query_embedding = dr_logits[:, 0, :].squeeze(dim=1).contiguous()
                ad_embedding = dr_logits[:, 1, :].squeeze(dim=1).contiguous()
                ood_embeding = dr_logits[:, 2, :].squeeze(dim=1).contiguous()

                # normalize
                query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)
                ad_embedding = ad_embedding / ad_embedding.norm(p=2, dim=-1, keepdim=True)
                ood_embeding = ood_embeding / ood_embeding.norm(p=2, dim=-1, keepdim=True)

                if self.use_cross_device_batch_negatives:
                    dr_celoss = self.distributed_in_batch_negatives(
                        query_embedding, ad_embedding, 
                        self.dr_margin, self.dr_temperature, 
                        self.dr_weight, is_dr, self.dr_dim
                    )
                    from acclgr.ops import GlobalNegativeSampling

                    query_embedding, _, _ = GlobalNegativeSampling.apply(query_embedding)
                    ad_embedding, _, _ = GlobalNegativeSampling.apply(ad_embedding)
                    ood_embeding, _, _ = GlobalNegativeSampling.apply(ood_embeding)
                    is_dr, _, _ = GlobalNegativeSampling.apply(is_dr)
                    insample_energy = compute_energy(query_embedding, ad_embedding, self.ood_temperature)
                    outsample_energy = compute_energy(ood_embeding, ad_embedding, self.ood_temperature)
                    energy_loss = compute_hingle_loss(insample_energy, outsample_energy, self.m_in, self.m_out, is_dr)
                    energy_loss = energy_loss * self.energy_loss_weight
                    loss = loss + energy_loss + dr_celoss
                else:
                    dr_matmul = query_embedding @ ad_embedding.T
                    margin = torch.full(size=(batch_size,), fill_value=self.dr_margin, device=dr_matmul.device)
                    margin = torch.diag(margin)
                    dr_preds = dr_matmul - margin
                    dr_preds = dr_preds / self.dr_temperature
                    # build dr_labels
                    dr_labels = torch.arange(0, batch_size, 1, dtype=torch.int64, device=dr_logits.device)
                    dr_celoss = F.cross_entropy(dr_preds, dr_labels, reduction='none')
                    dr_celoss = (dr_celoss.squeeze() * is_dr.squeeze()).mean() * self.dr_weight

                    # ood loss
                    insample_energy = compute_energy(query_embedding, ad_embedding, self.ood_temperature)
                    outsample_energy = compute_energy(ood_embeding, ad_embedding, self.ood_temperature)
                    energy_loss = compute_hingle_loss(insample_energy, outsample_energy, self.m_in, self.m_out, is_dr)
                    energy_loss = energy_loss * self.energy_loss_weight

                    loss = loss + dr_celoss + energy_loss

        if labels is not None:
            if self.is_sft == 1:
                sft_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                loss = loss + sft_loss * self.is_sft

        if random.random() < 0.1:
            print(f'loss: {loss.item()}, sft_loss: {sft_loss.item()}, dr_loss: {dr_celoss.item()}, ood_loss: {energy_loss.item()}')

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
    model = TradeQwenDR.from_pretrained(model_path)
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

