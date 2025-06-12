from distutils.command.config import config
from importlib.metadata import SelectableGroups
from transformers import Qwen2PreTrainedModel, Qwen2Model, Qwen2ForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class NluHead(nn.Module):
    def __init__(self, config, **kwargs):
        super.__init__()

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


class QwenWithTaskPlugin(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size

        #ori lm head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #nlu head
        self.nlu_head = NluHead(config)

        #init
        self.post_init()
       

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        is_use_sft_loss=None,
        cls_soft_label=None,
        is_use_cls_loss=None,
        tw_soft_label=None,
        is_use_tw_loss=None,
        **kwargs             #确认是否存在loss计算
    ):
        #ori model output, 是否lm head 输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        #ori lm head output
        logits = self.lm_head(hidden_states)

        #extra output
        next_sent_feat = hidden_states[:, -1, :]
        reward_logits, tw_logits = self.nlu_head(next_sent_feat)   
    
        #probs, tw_probs, logits
        eps = 1e-10
        device = logits.device if hasattr(logits, 'device') else 'cpu'
        loss = torch.tensor(0.0, device=device)
        lm_loss = torch.tensor(0.0, device=device)
        reward_loss = torch.tensor(0.0, device=device)
        tw_loss = torch.tensor(0.0, device=device)

        # 处理 lm_loss
        if 'labels' in kwargs:
            labels = kwargs['labels']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
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
        if cls_soft_label is not None:
            log_probs = F.log_softmax(reward_logits, dim=-1)
            reward_loss = F.kl_div(log_probs, cls_soft_label, reduction='none').sum(dim=1, keepdim=True)  # (bsz, 1)
            is_use_cls_loss = is_use_cls_loss if is_use_cls_loss is not None else torch.tensor(0.0)
            reward_loss = torch.multiply(reward_loss, is_use_cls_loss)

        # 处理 tw_loss
        if tw_soft_label is not None:
            log_tw_probs = F.log_softmax(tw_logits, dim=-1)
            tw_loss = F.kl_div(log_tw_probs, tw_soft_label, reduction='none').sum(dim=1, keepdim=True)  # (bsz, 1)
            is_use_tw_loss = is_use_tw_loss if is_use_tw_loss is not None else torch.tensor(0.0)
            tw_loss = torch.multiply(tw_loss, is_use_tw_loss)

        # 计算总损失
        loss = lm_loss + reward_loss + tw_loss
        loss = loss.mean()

        return CausalLMOutputWithPast(
            loss=loss if loss != 0 else None,  # 无损失时返回None保持兼容性
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )



