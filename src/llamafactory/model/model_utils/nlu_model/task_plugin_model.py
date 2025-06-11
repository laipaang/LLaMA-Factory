from distutils.command.config import config
from transformers import Qwen2PreTrainedModel, Qwen2Model, Qwen2ForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class QwenWithTaskPlugin(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
    
        # 共享原始LM Head（如果预训练权重需要）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
        # 分类头参数初始化优化
        self.next_sent_feat_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.jm63_classifier = nn.Linear(config.hidden_size, 4)
        self.tw_classifier = nn.Linear(config.hidden_size, 3)
    
        # 参数初始化
        for layer in [self.next_sent_feat_linear, self.jm63_classifier, self.tw_classifier]:
            nn.init.trunc_normal_(layer.weight, std=0.02, a=-0.04, b=0.04)
            nn.init.constant_(layer.bias, 0.0)
    
        self.post_init()

    def _compute_segment_features(self, hidden_states, position_ids):
        """向量化计算分段特征和掩码""" 
        batch_size, seq_len = position_ids.shape
        device = position_ids.device
        output = []

        for i in range(batch_size):
            # 找到所有0的位置（子样本分界点）
            zero_positions = (position_ids[i] == 0).nonzero().squeeze(-1)
            last_non_zero = (position_ids[i] != 0).nonzero().squeeze(-1) 
            last_non_zero = last_non_zero[-1]
            valid_zero_positions = zero_positions[zero_positions <= last_non_zero]
    
            # 计算每个子样本的最后一个有效位置的索引
            sub_sample_ends = []
            # 第一个子样本的结束位置是第一个0的前一个位置
            if valid_zero_positions[0] > 0:
                sub_sample_ends.append(valid_zero_positions[0] - 1)
            # 中间子样本的结束位置是下一个0的前一个位置
            for j in range(1, len(valid_zero_positions)):
                if valid_zero_positions[j] - valid_zero_positions[j-1] > 1:
                    sub_sample_ends.append(valid_zero_positions[j] - 1)
            # 最后一个子样本的结束位置是最后一个非0位置
            sub_sample_ends.append(last_non_zero)
            output.append(torch.tensor(sub_sample_ends, device=device))
        # 补齐到最大子样本数
        max_sub_samples = max([len(indices) for indices in output]) if batch_size > 0 else 0
        padded_output = torch.full((batch_size, max_sub_samples), -1, dtype=torch.long, device=device)
        for i in range(batch_size):
            if len(output[i]) > 0:
                padded_output[i, :len(output[i])] = output[i]
        #print(f"padded_output shape: {padded_output.shape}")

        batch_size, num_sub_samples = padded_output.shape
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_sub_samples)
        segment_features = hidden_states[batch_indices, padded_output]

        #segment_features = torch.gather(hidden_states, 1, padded_output)
        #print(f"segment features shape: {segment_features.shape}")
        #print(f"padded output: {padded_output}")
        return segment_features, padded_output

    def get_valid_mask(self, soft_labels, pad_token=-100):
        """生成有效位置的布尔掩码
        参数：
            soft_labels: (batch_size, cutoff_len, num_classes)
            pad_token: 无效填充位置的特征值
        返回：
            mask: (batch_size, cutoff_len), True表示有效位置
        """
        # 若soft_labels是概率分布，无效位置可能在某个特征维度为pad_token
        # 例如：填充位置的概率分布全为-100
        return (soft_labels[..., 0] != pad_token)  # 检查第一个特征是否无效


    def compute_masked_kl_loss(self, pred, target, task_mask=None, pad_token=-100):
        """
        参数：
            pred: (batch_size, cutoff_len, num_classes)
            target: (batch_size, cutoff_len, num_classes)
            task_mask: (batch_size, cutoff_len) 可选的任务权重
        """
        valid_mask = self.get_valid_mask(target, pad_token)

        # 整合任务权重
        #if task_mask is not None:
        #    combined_mask = valid_mask.float() * task_mask.float()
        #else:
        #    combined_mask = valid_mask.float()
        #print(f"valid mask: {valid_mask}")
        #print(f"pred shape: {pred.shape}")

        # 展平处理
        batch_size, seq_len, num_classes = pred.shape
        #print(f"pred flat shape: {pred.view(-1, num_classes).shape}")
        #print(f"valid mask flat shape: {valid_mask.view(-1).shape}")


        pred_flat = pred.view(-1, num_classes)[valid_mask.view(-1)]  # (total_valid, C)
        target_flat = target.reshape(-1, num_classes)[valid_mask.reshape(-1)]
        weights_flat = task_mask.reshape(-1)[valid_mask.reshape(-1)]
        #target_flat = target.view(-1, num_classes)[valid_mask.view(-1)]  # (total_valid, C)
        #weights_flat = combined_mask.view(-1)[valid_mask.view(-1)]  # (total_valid,)
        #print(f"target_flat shape: {target_flat.shape}, weights_flat shape: {weights_flat.shape}")

        # 计算加权KL散度
        log_probs = F.log_softmax(pred_flat, dim=-1)
        kl_elements = F.kl_div(log_probs, target_flat, reduction='none').sum(dim=1)
        weighted_loss = (kl_elements * weights_flat).sum()
        
        # 归一化
        if weights_flat.sum() > 0:
            return weighted_loss / weights_flat.sum()
        else:
            return torch.tensor(0.0, device=pred.device)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        is_use_sft_loss=None,
        cls_soft_label=None,
        is_use_cls_loss=None,
        tw_soft_label=None,
        is_use_tw_loss=None,
        sample_last_indices=None,
        **kwargs
    ):
        # 基础模型前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state
        segment_features = None
        if sample_last_indices is not None:
            mask = sample_last_indices != 0
            lengths = mask.sum(dim=1)
            trimmed = [sample_last_indices[i, :lengths[i]].tolist() for i in range(len(sample_last_indices))]
            max_len = max(lengths.tolist())
            padded_sample_last_indices = torch.full((len(sample_last_indices), max_len), -1, dtype=sample_last_indices.dtype)
            for i in range(len(sample_last_indices)):
                padded_sample_last_indices[i, :lengths[i]] = sample_last_indices[i, :lengths[i]]
            #print(f"sample last indices in model: {sample_last_indices}")
            #print(f"padded sample last indices: {padded_sample_last_indices}")

            batch_size, num_sub_samples = padded_sample_last_indices.shape
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_sub_samples)
            segment_features = hidden_states[batch_indices, padded_sample_last_indices]
            num_segments = segment_features.size(1)
            #print(f"num segments: {num_segments}")

        # ===== 分段特征提取 =====
        #segment_features, seg_mask = self._compute_segment_features(hidden_states, position_ids)
        #num_segments = segment_features.size(1)
        #print(f"num segments: {num_segments}")

        # ===== 语言模型头计算 =====
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)

        reward_logits, tw_logits = None, None
        # ===== 分类头计算 =====
        if segment_features != None:
            next_sent_feat = torch.tanh(self.next_sent_feat_linear(segment_features))
            reward_logits = self.jm63_classifier(next_sent_feat)  # (batch_size, num_segments, 4)
            tw_logits = self.tw_classifier(next_sent_feat)        # (batch_size, num_segments, 3)

        # ===== 损失计算 =====
        loss = 0.0

        # 语言模型损失
        if labels is not None:
            # 生成shifted labels掩码
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()

            # 计算每个token的损失
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            ).view_as(shift_labels)
            #print(f"lm loss shape: {lm_loss.shape}")

            # 应用分段掩码和任务权重
            valid_mask = (shift_labels != -100).float()
            #print(f"valid mask: {valid_mask}")
            if is_use_sft_loss is not None:
                is_use_sft_loss = is_use_sft_loss[:, :511].squeeze(-1)
                valid_mask *= is_use_sft_loss.expand_as(valid_mask)
            #print(f"after valid mask: {valid_mask}")

            lm_loss = (lm_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1e-6)
            loss += lm_loss
            #print(f"lm loss: %s" % lm_loss)
        
        # 奖励分类损失
        if cls_soft_label is not None and reward_logits is not None:
            #print(f"reward logtis shape: {reward_logits.shape}, target shape: {cls_soft_label.shape}, mask shape: {is_use_cls_loss.shape}")
            reward_loss = self.compute_masked_kl_loss(
                pred=reward_logits,  # (B, L, C)
                target=cls_soft_label[:, :num_segments],
                task_mask=is_use_cls_loss[:, :num_segments],  # (B, L)
                pad_token=-100
            )
            #print(f"reward loss: {reward_loss}")
            loss += reward_loss

        # TW分类损失
        if tw_soft_label is not None and tw_logits is not None:
            tw_loss = self.compute_masked_kl_loss(
                pred=tw_logits,
                target=tw_soft_label[:, :num_segments],
                task_mask=is_use_tw_loss[:, :num_segments],
                pad_token=-100
            )
            loss += tw_loss
            #print(f"tw loss: {tw_loss}")

        #print(f"final loss: {loss}")

        return CausalLMOutputWithPast(
            loss=loss if loss != 0 else None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

