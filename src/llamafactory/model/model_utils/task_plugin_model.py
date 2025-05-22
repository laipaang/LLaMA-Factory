from distutils.command.config import config
from transformers import Qwen2PreTrainedModel, Qwen2Model
import torch
import torch.nn as nn

class QwenWithTaskPlugin(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        
        #ori lm head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #added head
        self.next_sent_feat_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.jm63_linear = nn.Linear(config.hidden_size, 4)
        self.tw_linear = nn.Linear(config.hidden_size, 2)
        #init
        self.post_init()       
        #add init distribute
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
        **kwargs             #确认是否存在loss计算
    ):
        final_outputs = {}
        #ori model output, 是否lm head 输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs           
        )
        final_outputs['outputs'] = outputs
        
        hidden_states = outputs.last_hidden_state
        #ori lm head output
        logits = self.lm_head(hidden_states)
        final_outputs['logits'] = logits

        #extra output
        attention_mask_expanded = attention_mask.unsqueeze(-1)  # [bsz, seq_len, 1]
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)  # [bsz, hidden_size]
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9) 
        next_sent_feat = sum_embeddings / sum_mask  # [bsz, hidden_size]

        reward_logits = self.jm63_linear(next_sent_feat)  # [batch_size, 2]
        tw_logits = self.tw_linear(next_sent_feat)

        tw_prob = torch.sigmoid(tw_logits)  # [batch_size, 1]
        tw_prob = torch.squeeze(tw_prob, dim=-1)  # [batch_size]
        tw_probs = torch.cat([1.0 - tw_prob, tw_prob], dim=-1)  # [batch_size, 2]

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

            if is_use_sft_loss is None:
                is_use_sft_loss = torch.tensor(1.0, device=device)  # 默认启用
            lm_loss = torch.multiply(lm_loss, is_use_sft_loss)
    
        # 处理 reward_loss
        if cls_soft_label is not None:
            log_probs = F.log_softmax(reward_logits, dim=-1)
            reward_loss = F.kl_div(log_probs, cls_soft_label, reduction='none').sum(dim=1, keepdim=True)  # (bsz, 1)
            if is_use_cls_loss is None:
                is_use_cls_loss = torch.tensor(0.0, device=device)
            reward_loss = torch.multiply(reward_loss, is_use_cls_loss)
        

            probs_clip = torch.clip(x=probs, min=eps, max=1.0 - eps)  
            log_probs = torch.log(probs_clip)
            reward_loss = -torch.sum(torch.multiply(cls_soft_label, log_probs))
            if is_use_cls_loss is None:
                is_use_cls_loss = torch.tensor(0.0, device=device)
            reward_loss = torch.multiply(reward_loss, is_use_cls_loss)

        # 处理 tw_loss
        if tw_soft_label is not None:
            log_tw_probs = F.log_softmax(tw_probs, dim=-1)
            tw_loss = F.kl_div(log_twprobs, tw_soft_label, reduction='none').sum(dim=1, keepdim=True)  # (bsz, 1)
            if is_use_tw_loss is None:
                is_use_tw_loss = torch.tensor(0.0, device=device)
            tw_loss = torch.multiply(tw_loss, is_use_tw_loss)   


            tw_probs_clip = torch.clip(tw_probs, min=eps, max=1.0 - eps)
            log_twprobs = torch.log(tw_probs_clip)
            tw_loss = -torch.sum(torch.multiply(tw_soft_label, log_twprobs), axis=1)
            if is_use_tw_loss is None:
                is_use_tw_loss = torch.tensor(0.0, device=device)
            tw_loss = torch.multiply(tw_loss, is_use_tw_loss)

        # 计算总损失
        loss = lm_loss + reward_loss + tw_loss
        loss = loss.mean()
        final_outputs['loss'] = loss
        return final_outputs

    

