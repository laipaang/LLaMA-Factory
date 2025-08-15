from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen

@dataclass
class CustidDRDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:

        model_inputs = defaultdict(list)

        for idx in range(len(examples["_src"])):
            src_list = examples["_src"][idx]
            tgt_list = examples["_tgt"][idx]

            num_pairs = len(src_list)
            for pair_idx in range(num_pairs): # num_pairs其实为1
                # 处理单个 src-tgt 对
                src = src_list[pair_idx]
                tgt = tgt_list[pair_idx]
                # tgt, dr_src, dr_tgt, is_dr = tgt.split('\x01')
                tgt, bidword, fea, biz, is_dr, ood, tw_query, tw_bw, tw_label = tgt.split('\x01')
                dr_index_fea = bidword + ';' + fea + ';' + biz
                dr_index_fea = dr_index_fea.replace('[NO_CONTENT]', '')

                # sft src and tgt
                src_msg = self.template.format_user.apply(content=src)
                tgt_msg = self.template.format_assistant.apply(content=tgt)
                source_ids = self.tokenizer.encode(src_msg[0], add_special_tokens=False)
                target_ids = self.tokenizer.encode(tgt_msg[0], add_special_tokens=False)
                # 截断到最大长度
                source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len)
                source_ids = source_ids[:source_len]
                target_ids = target_ids[:target_len]
                
                # dr sft src and tgt
                dr_index_fea_msg = self.template.format_user.apply(content=dr_index_fea)
                dr_index_bidword_msg = self.template.format_user.apply(content=bidword)
                ood_msg = self.template.format_user.apply(content=ood)
                tw_query_msg = self.template.format_user.apply(content=tw_query)
                tw_bw_msg = self.template.format_user.apply(content=tw_bw)

                dr_index_fea_ids = self.tokenizer.encode(dr_index_fea_msg[0], add_special_tokens=False)
                dr_index_bidword_ids = self.tokenizer.encode(dr_index_bidword_msg[0], add_special_tokens=False)
                ood_ids = self.tokenizer.encode(ood_msg[0], add_special_tokens=False)
                tw_query_ids = self.tokenizer.encode(tw_query_msg[0], add_special_tokens=False)
                tw_bw_ids = self.tokenizer.encode(tw_bw_msg[0], add_special_tokens=False)

                # 截断到最大长度
                dr_index_fea_len, _ = infer_seqlen(len(dr_index_fea_ids), 0, self.data_args.cutoff_len)
                dr_index_bidword_len, _ = infer_seqlen(len(dr_index_bidword_ids), 0, self.data_args.cutoff_len)
                ood_len, _ = infer_seqlen(len(ood_ids), 0, self.data_args.cutoff_len)
                tw_query_len, _ = infer_seqlen(len(tw_query_ids), 0, self.data_args.cutoff_len)
                tw_bw_len, _ = infer_seqlen(len(tw_bw_ids), 0, self.data_args.cutoff_len)

                dr_index_fea_ids = dr_index_fea_ids[:dr_index_fea_len]
                dr_index_bidword_ids = dr_index_bidword_ids[:dr_index_bidword_len]
                ood_ids = ood_ids[:ood_len]
                tw_query_ids = tw_query_ids[:tw_query_len]
                tw_bw_ids = tw_bw_ids[:tw_bw_len]

                # 构建输入和标签
                input_ids = source_ids + target_ids
                label_ids = [IGNORE_INDEX] * len(source_ids) + target_ids

                dr_index_fea_label_ids = [IGNORE_INDEX] * len(dr_index_fea_ids)
                dr_index_bidword_label_ids = [IGNORE_INDEX] * len(dr_index_bidword_ids)
                ood_label_ids = [IGNORE_INDEX] * len(ood_ids)
                tw_query_label_ids = [IGNORE_INDEX] * len(tw_query_ids)
                tw_bw_label_ids = [IGNORE_INDEX] * len(tw_bw_ids)

                # if self.template.efficient_eos:
                #     input_ids.append(self.tokenizer.eos_token_id)
                #     label_ids.append(self.tokenizer.eos_token_id)
                #     dr_source_ids.append(self.tokenizer.eos_token_id)
                #     dr_target_ids.append(self.tokenizer.eos_token_id)
                #     dr_source_label_ids.append(self.tokenizer.eos_token_id)
                #     dr_tgt_label_ids.append(self.tokenizer.eos_token_id)

                model_inputs["input_ids"].append(input_ids + dr_index_fea_ids + dr_index_bidword_ids + ood_ids + tw_query_ids + tw_bw_ids)
                model_inputs["attention_mask"].append([1] * len(input_ids) + [2] * len(dr_index_fea_ids) + [3] * len(dr_index_bidword_ids) + [4] * len(ood_ids) + [5] * len(tw_query_ids) + [6] * len(tw_bw_ids))
                model_inputs["labels"].append(label_ids + dr_index_fea_label_ids + dr_index_bidword_label_ids + ood_label_ids + tw_query_label_ids + tw_bw_label_ids)
                model_inputs['dr_slice'].append([
                    len(source_ids),
                    len(input_ids) + len(dr_index_fea_ids),
                    len(input_ids) + len(dr_index_fea_ids) + len(dr_index_bidword_ids) + len(ood_ids),
                    ])
                model_inputs['tw_slice'].append([
                    len(source_ids),
                    len(input_ids) + len(dr_index_fea_ids) + len(dr_index_bidword_ids),
                    len(input_ids) + len(dr_index_fea_ids) + len(dr_index_bidword_ids) + len(ood_ids) + len(tw_query_ids), 
                    len(input_ids) + len(dr_index_fea_ids) + len(dr_index_bidword_ids) + len(ood_ids) + len(tw_query_ids) + len(tw_bw_ids)
                    ])
                model_inputs['is_dr'].append(float(is_dr))
                model_inputs['tw_label'].append(float(tw_label))
                
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:{}\n".format(example["input_ids"]))
        ids = example["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        print("inputs:{}\n".format(self.tokenizer.decode(ids, skip_special_tokens=False)))
        print("label_ids:{}\n".format(example["labels"]))
        print(f'attention_mask: {example["attention_mask"]}')

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
