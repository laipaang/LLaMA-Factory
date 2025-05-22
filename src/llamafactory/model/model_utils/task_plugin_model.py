from distutils.command.config import config
from transformers import Qwen2PreTrainedModel, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenWithTaskPlugin(Qwen2ForCausalLM):  # 修改继承关系
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        
        # 原始LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 新增分类头（保持原始初始化）
        self.next_sent_feat_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.jm63_linear = nn.Linear(config.hidden_size, 4)
        self.tw_linear = nn.Linear(config.hidden_size, 2)
        
        self.post_init()
        nn.init.trunc_normal_(self.next_sent_feat_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.next_sent_feat_linear.bias, 0.0)
        nn.init.trunc_normal_(self.jm63_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.jm63_linear.bias, 0.0)
        nn.init.trunc_normal_(self.tw_linear.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.constant_(self.tw_linear.bias, 0.0)

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
        **kwargs
    ):
        # 原始模型前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)  # 生成所需的主logits 

        # 保持原有的损失计算逻辑
        loss = torch.tensor(0.0, device=hidden_states.device)
        batch_size = hidden_states.size(0)
        
        # 池化层计算
        next_sent_feat = hidden_states[:, -1, :]
        
        # 分类头计算
        next_sent_feat = torch.tanh(self.next_sent_feat_linear(next_sent_feat))
        reward_logits = self.jm63_linear(next_sent_feat)
        tw_logits = self.tw_linear(next_sent_feat)

        # 语言模型损失计算
        if 'labels' in kwargs:
            labels = kwargs['labels']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                reduction='none',
                ignore_index=-100
            ).view_as(shift_labels)
            
            lm_loss = (loss_per_token.sum(dim=1) / (shift_labels != -100).sum(dim=1)).unsqueeze(-1)
            is_use_sft_loss = is_use_sft_loss if is_use_sft_loss is not None else torch.tensor(1.0)
            loss += (lm_loss * is_use_sft_loss).mean()

        # 奖励损失计算
        if cls_soft_label is not None:
            log_probs = F.log_softmax(reward_logits, dim=-1)
            reward_loss = F.kl_div(log_probs, cls_soft_label, reduction='none').sum(dim=1, keepdim=True)
            is_use_cls_loss = is_use_cls_loss if is_use_cls_loss is not None else torch.tensor(0.0)
            loss += (reward_loss * is_use_cls_loss).mean()

        # TW损失计算
        if tw_soft_label is not None:
            log_twprobs = F.log_softmax(tw_logits, dim=-1)
            tw_loss = F.kl_div(log_twprobs, tw_soft_label, reduction='none').sum(dim=1, keepdim=True)
            is_use_tw_loss = is_use_tw_loss if is_use_tw_loss is not None else torch.tensor(0.0)
            loss += (tw_loss * is_use_tw_loss).mean()

        # 返回标准格式输出
        return CausalLMOutputWithPast(
            loss=loss if loss != 0 else None,  # 无损失时返回None保持兼容性
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
