from distutils.command.config import config
from gzip import _PaddedFile
from transformers import Qwen2PreTrainedModel, Qwen2Model, Qwen2ForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

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
        #init
        nn.init.trunc_normal_(self.dr_next_sent_feat_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.dr_next_sent_feat_linear.bias, 0.0)
        nn.init.trunc_normal_(self.dr_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.dr_linear.bias, 0.0)

    def forward(self, hidden_states):
        next_sent_feat = self.dr_next_sent_feat_linear(hidden_states)
        next_sent_feat = torch.tanh(next_sent_feat)

        next_sent_feat = self.dr_linear(next_sent_feat)
        next_sent_feat = torch.tanh(next_sent_feat)
        next_sent_feat = next_sent_feat.squeeze(1)
        next_sent_feat_query = next_sent_feat[0::2]
        next_sent_feat_agent = next_sent_feat[1::2]
        dr_logits = torch.matmul(next_sent_feat_query, next_sent_feat_agent.T)
        softmax_margin = torch.full(size=[dr_logits.shape[0]], fill_value=0.0, dtype=torch.float32)
        margin = torch.diag(softmax_margin)
        dr_logits = torch.subtract(dr_logits, margin.to(dr_logits.device))
        return dr_logits


class QwenWithDrInAgent(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size

        #ori lm head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #dr head
        self.context_feature_model = DenseRetrievalHead(config)

        #init
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        **kwargs
    ):
        #ori model output, 是否lm head 输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        sft_hidden_states = hidden_states[0::2]
        #ori lm head output
        logits = self.lm_head(sft_hidden_states)

        #extra output
        next_sent_feat = hidden_states[:, -1, :]
        dr_logits = self.context_feature_model(next_sent_feat)

        #probs, tw_probs, logits
        eps = 1e-10
        device = logits.device if hasattr(logits, 'device') else 'cpu'
        loss = torch.tensor(0.0, device=device)
        sft_loss = torch.tensor(0.0, device=device)
        dr_loss = torch.tensor(0.0, device=device)

        # 处理 lm_loss
        if 'labels' in kwargs:
            labels = kwargs['labels']
            labels = labels[0::2]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                reduction='none',
                ignore_index=-100  # 假设用-100表示padding
            )
            loss_per_token = loss_per_token.view(shift_labels.shape)    # 还原成 (bsz, seq_len)
            sft_loss = (loss_per_token.sum(dim=1) / (shift_labels != -100).sum(dim=1)).unsqueeze(-1)  # 对 seq_len 取 Mean, 忽略 ignore_index

        #处理 dr_loss
        dr_labels = torch.arange(0, dr_logits.shape[0], dtype=torch.int64)
        dr_labels.stop_graidient = True
        dr_labels = dr_labels.to(dr_logits.device)
        #dr_labels = torch.reshape(x=dr_labels, shape=[-1, 1])
        dr_loss = F.cross_entropy(dr_logits, dr_labels, reduction='none')

        loss = sft_loss + dr_loss
        loss = loss.mean()

        return CausalLMOutputWithPast(
            loss=loss if loss != 0 else None,  # 无损失时返回None保持兼容性
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )





