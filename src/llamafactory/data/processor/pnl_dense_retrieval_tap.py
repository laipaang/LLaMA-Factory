from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen

@dataclass
class PNLDenseRetrievalTAPDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:

        model_inputs = defaultdict(list)

        for idx in range(len(examples["_src"])):
            src_list = examples["_src"][idx]
            tgt_list = examples["_tgt"][idx]
            # dr_src_list = examples["_dr_src"][idx]
            # dr_tgt_list = examples["_dr_tgt"][idx]
            # use_dr_loss_list = examples["_is_use_dr_loss"][idx]

            num_pairs = len(src_list)
            for pair_idx in range(num_pairs): # num_pairs其实为1
                # 处理单个 src-tgt 对
                src = src_list[pair_idx]
                tgt = tgt_list[pair_idx]

                #tgt, dr_src, dr_tgt, is_dr = tgt.split('\x01')
                #8       关键词：湿气重会影响男性性功能吗。扩展标题：湿气重可能导致男性性功能受影响。强相关改写：        l0235  l1032  l2216[SEP]1_0 0.0561000108719_0.85268098861_0.0912190005183 1[SEP]1[SEP]1[SEP]1   男士性功能下降  湿气重会影响男性性功能吗        徐州医健医院---公信立医院，徐州男科总院，徐州性协会，专注于男性病诊疗，致力于提供全面、专业的医疗服务。一。医院：老牌·正规·以男性病为诊疗中心二。专病：包皮包茎,阳痿早泄,生殖器感染,前列腺病等男科疾病三。专治：汇聚京沪浙苏教授团队,多年专注男病研究与诊疗，临床经验丰富四。专科：致力于男性健康的诊疗、康复及研究五。专研：积极参与学术交流，不断提高临床运用和技术水平创新能力，提升自身医疗质量六。隐私：为保障医疗质量和患者隐私，医院执行“一患、一医、一诊室”的诊疗模式，规范用药，透明收费。七。挂号：为提高看诊质量，缓解挂号难、排队久，我院特开通“网络预约”挂号就医服务。16至79岁人群。      关键词：猛大帅。扩展标题：猛大帅是变形金刚玩具中的角色。强相关改写：
                # tgt, tap_jm63_scores, tap_skynet_scores, click, is_sft, is_tap, is_dr, dr_tgt_pos, dr_tgt_hardneg_lp, dr_tgt_hardneg_bidword, dr_tgt_hardneg_cluster, dr_ood_query = tgt.split('[SEP]')
                tgt, tap_jm63_scores, tap_skynet_scores, click, is_sft, is_tap, is_dr, dr_tgt_positive, dr_tgt_hardneg_lp, dr_tgt_hardneg_bidword, dr_ood_query = tgt.split('[SEP]')

                # sft src and tgt
                src_msg = self.template.format_user.apply(content=src)
                tgt_msg = self.template.format_assistant.apply(content=tgt)
                source_ids = self.tokenizer.encode(src_msg[0], add_special_tokens=False)
                target_ids = self.tokenizer.encode(tgt_msg[0], add_special_tokens=False)
                # 截断到最大长度
                source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len)
                source_ids = source_ids[:source_len]
                target_ids = target_ids[:target_len]
                 # 构建输入和标签
                input_ids = source_ids + target_ids
                label_ids = [IGNORE_INDEX] * len(source_ids) + target_ids
                
                # dr sft src and tgt
                #dr_tgt_pos, dr_tgt_hardneg_lp, dr_tgt_hardneg_bidword, dr_ood_query
                dr_tgt_positive_msg = self.template.format_user.apply(content=dr_tgt_positive)
                dr_tgt_positive_ids = self.tokenizer.encode(dr_tgt_positive_msg[0], add_special_tokens=False)
                dr_tgt_positive_len, _ = infer_seqlen(len(dr_tgt_positive_ids), 0, self.data_args.cutoff_len)
                dr_tgt_positive_ids = dr_tgt_positive_ids[:dr_tgt_positive_len]
                dr_tgt_positive_label_ids = [IGNORE_INDEX] * len(dr_tgt_positive_ids)

                dr_tgt_hardneg_lp_msg = self.template.format_user.apply(content=dr_tgt_hardneg_lp)
                dr_tgt_hardneg_lp_ids = self.tokenizer.encode(dr_tgt_hardneg_lp_msg[0], add_special_tokens=False)
                dr_tgt_hardneg_lp_len, _ = infer_seqlen(len(dr_tgt_hardneg_lp_ids), 0, self.data_args.cutoff_len)
                dr_tgt_hardneg_lp_ids = dr_tgt_hardneg_lp_ids[:dr_tgt_hardneg_lp_len]
                dr_tgt_hardneg_lp_label_ids = [IGNORE_INDEX] * len(dr_tgt_hardneg_lp_ids)

                dr_tgt_hardneg_bidword_msg = self.template.format_user.apply(content=dr_tgt_hardneg_bidword)
                dr_tgt_hardneg_bidword_ids = self.tokenizer.encode(dr_tgt_hardneg_bidword_msg[0], add_special_tokens=False)
                dr_tgt_hardneg_bidword_len, _ = infer_seqlen(len(dr_tgt_hardneg_bidword_ids), 0, self.data_args.cutoff_len)
                dr_tgt_hardneg_bidword_ids = dr_tgt_hardneg_bidword_ids[:dr_tgt_hardneg_bidword_len]
                dr_tgt_hardneg_bidword_label_ids = [IGNORE_INDEX] * len(dr_tgt_hardneg_bidword_ids)

                dr_ood_query_msg = self.template.format_user.apply(content=dr_ood_query)
                dr_ood_query_ids = self.tokenizer.encode(dr_ood_query_msg[0], add_special_tokens=False)
                dr_ood_query_len, _ = infer_seqlen(len(dr_ood_query_ids), 0, self.data_args.cutoff_len)
                dr_ood_query_ids = dr_ood_query_ids[:dr_ood_query_len]
                dr_ood_query_label_ids = [IGNORE_INDEX] * len(dr_ood_query_ids)

                # not open
                if self.template.efficient_eos:
                    input_ids.append(self.tokenizer.eos_token_id)
                    label_ids.append(self.tokenizer.eos_token_id)
                    dr_tgt_positive_ids.append(self.tokenizer.eos_token_id)
                    dr_tgt_positive_label_ids.append(self.tokenizer.eos_token_id)
                    dr_tgt_hardneg_lp_ids.append(self.tokenizer.eos_token_id)
                    dr_tgt_hardneg_lp_label_ids.append(self.tokenizer.eos_token_id)
                    dr_tgt_hardneg_bidword_ids.append(self.tokenizer.eos_token_id)
                    dr_tgt_hardneg_bidword_label_ids.append(self.tokenizer.eos_token_id)
                    dr_ood_query_ids.append(self.tokenizer.eos_token_id)
                    dr_ood_query_label_ids.append(self.tokenizer.eos_token_id)

                model_inputs["input_ids"].append(input_ids + \
                                                 dr_tgt_positive_ids + \
                                                 dr_tgt_hardneg_lp_ids + \
                                                 dr_tgt_hardneg_bidword_ids + \
                                                 dr_ood_query_ids)
                model_inputs["attention_mask"].append([1] * len(input_ids) + \
                                                      [2] * len(dr_tgt_positive_ids) + \
                                                      [3] * len(dr_tgt_hardneg_lp_ids) + \
                                                      [4] * len(dr_tgt_hardneg_bidword_ids) + \
                                                      [5] * len(dr_ood_query_ids))
                model_inputs["labels"].append(label_ids + \
                                              dr_tgt_positive_label_ids + \
                                              dr_tgt_hardneg_lp_label_ids + \
                                              dr_tgt_hardneg_bidword_label_ids + \
                                              dr_ood_query_label_ids)
                model_inputs['dr_slice'].append([len(source_ids),
                                                 len(input_ids) + len(dr_tgt_positive_ids), 
                                                 len(input_ids) + len(dr_tgt_positive_ids) + len(dr_tgt_hardneg_lp_ids), 
                                                 len(input_ids) + len(dr_tgt_positive_ids) + len(dr_tgt_hardneg_lp_ids) + len(dr_tgt_hardneg_bidword_ids), 
                                                 len(input_ids) + len(dr_tgt_positive_ids) + len(dr_tgt_hardneg_lp_ids) + len(dr_tgt_hardneg_bidword_ids) + len(dr_ood_query_ids)])
                model_inputs['is_dr'].append(float(is_dr))
                # model_inputs['is_tap'].append(float(is_tap))
                model_inputs['is_use_cls_loss'].append(float(is_tap))
                model_inputs['is_use_tw_loss'].append(float(is_tap))
                model_inputs["is_use_sft_loss"].append(float(is_sft))
                model_inputs["cls_soft_label"].append([float(i) for i in tap_jm63_scores.split('_')])
                model_inputs["tw_soft_label"].append([float(i) for i in tap_skynet_scores.split('_')])
                model_inputs["sample_length"].append(len(input_ids) - 1)
                
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:{}\n".format(example["input_ids"]))
        ids = example["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        print("inputs:{}\n".format(self.tokenizer.decode(ids, skip_special_tokens=False)))
        print("label_ids:{}\n".format(example["labels"]))
        print(f'attention_mask: {example["attention_mask"]}')
        print(f'dr_slice: {example["dr_slice"]}')
        print(f'is_use_cls_loss: {example["is_use_cls_loss"]}')
        print(f'is_use_tw_loss: {example["is_use_tw_loss"]}')
        print(f'is_use_sft_loss: {example["is_use_sft_loss"]}')
        print(f'cls_soft_label: {example["cls_soft_label"]}')
        print(f'tw_soft_label: {example["tw_soft_label"]}')
        print(f'sample_length: {example["sample_length"]}')

        # print("position_ids:{}\n".format(example["position_ids"]))
        labels_list = example["labels"]
        if isinstance(labels_list[0], list):
            for i, labels in enumerate(labels_list):
                valid_labels = [x for x in labels if x != IGNORE_INDEX]
                if len(valid_labels) == 0:
                    print(f"labels[{i}]: <EMPTY>")
                else:
                    try:
                        print(f"labels[{i}]: {self.tokenizer.decode(valid_labels, skip_special_tokens=False)}\n")
                    except Exception as e:
                        print(f"labels[{i}]: decode error: {e}, valid_labels={valid_labels}")
        else:
            valid_labels = [x for x in labels_list if x != IGNORE_INDEX]
            if len(valid_labels) == 0:
                print("labels: <EMPTY>")
            else:
                try:
                    print(f"labels: {self.tokenizer.decode(valid_labels, skip_special_tokens=False)}\n")
                except Exception as e:
                    print(f"labels: decode error: {e}, valid_labels={valid_labels}")
