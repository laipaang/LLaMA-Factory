from transformers import Qwen2PreTrainedModel, Qwen2Model
import torch
import torch.nn as nn
import torch.nn.init as init

class QwenWithTaskPlugin(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = Qwen2Model(config)

        '''
        #ori lm head
        self.lm_head = nn.linear(config.hidden_size, config.vocab_size, bias=False)
        
        #added head
        self.next_sent_feat_linear = nn.Linear(hidden_size, hidden_size)
        self.jm63_linear = nn.Linear(hidden_size, 4)
        self.tw_linear = nn.Linear(hidden_size, 1)

        #init
        self.post_init()   
        '''     

        hidden_size = getattr(config, "hidden_size", None)
        self.next_sent_feat_linear = nn.Linear(hidden_size, hidden_size)
        init.trunc_normal_(self.next_sent_feat_linear.weight, std=0.02, a=-0.04, b=0.04)
        init.constant(self.next_sent_feat_linear.bias, 0.0)
        self.jm63_linear = nn.Linear(hidden_size, 4)    #cls_score_size
        init.trunc_normal_(self.jm63_linear.weight, std=0.02, a=-0.04, b=0.04)
        init.constant(self.jm63_linear.bias, 0.0)
        self.tw_linear = nn.Linear(hidden_size, 1)   #tw_score_size
        init.trunc_normal_(self.tw_linear.weight, std=0.02, a=-0.04, b=0.04)
        init.constant(self.tw_linear.bias, 0.0)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        is_use_sft_loss=None,
        cls_soft_label=None,
        is_use_cls_loss=None,
        tw_softlabel=None,
        is_use_tw_loss=None,
        **kwargs             #确认是否存在loss计算
    ):
        #ori model output, 是否lm head 输出
        outputs = super.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs           
        )
        
        hidden_states = outputs.last_hidden_state
        
        #ori lm head output
        logits = self.lm_head(hidden_states)

        #extra output
        hidden_states = outputs.hidden_states[-1]    #这里和zeus输入对齐
        last_token_hidden = hidden_states[:, -1, :]   #[batch, hideen_size]
        next_sent_feat = self.next_sent_feat_linear(last_token_hidden).squeeze(-1)  #[batch]
        next_sent_feat = torch.tanh(next_sent_feat) 
        reward_logits = self.jm63_linear(next_sent_feat)  # [batch_size, 2]
        probs = F.softmax(reward_logits, dim=-1) 
        tw_logits = self.tw_linear(next_sent_feat)

        tw_prob = torch.sigmoid(tw_logits)  # [batch_size, 1]
        tw_prob = torch.squeeze(tw_prob, dim=-1)  # [batch_size]
        tw_probs = torch.cat([1.0 - tw_prob, tw_prob], dim=-1)  # [batch_size, 2]

        #probs, tw_probs, logits
        eps = 1e-10
        loss = None
        if 'labels' in kwargs or cls_soft_label is not None or tw_softlabel is not None:
            lm_loss = None
            if 'labels' in kwargs:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = kwargs['labels'][..., 1:].contiguous()
                lm_loss_fct = nn.CrossEntropyLoss()
                lm_loss = lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                    shift_labels.view(-1))
            lm_loss = torch.multiply(lm_loss, is_use_sft_loss)
            
            reward_loss = None
            if cls_soft_label is not None:
                probs_clip = torch.clip(x=probs, min=eps, max=1.0 - eps)  
                log_probs = torch.log(probs_clip)
                reward_loss = -torch.sum(torch.multiply(cls_soft_label, log_probs))
                reward_loss = torch.multiply(reward_loss, is_use_cls_loss)
            
            tw_loss = None
            if tw_softlabel is not None:
                tw_probs = torch.clip(tw_probs, min=eps, max=1.0 - eps)
                log_twprobs = torch.log(tw_probs_clip)
                tw_loss = -torch.sum(torch.multiply(tw_softlabel, log_twprobs), axis=1)
                tw_loss = torch.multiply(tw_loss, is_use_tw_loss)
        
        loss = lm_loss + reward_loss + tw_loss
        return loss

    

