#!/usr/bin/env python
# coding: utf-8

from torch.utils.data.dataset import Dataset
import helper.logger as logger
import json
import os
from transformers import BertTokenizer
import torch
from collections import defaultdict


def get_sample_position(corpus_filename, on_memory, corpus_lines, stage):
    """
    position of each sample in the original corpus File or on-memory List
    :param corpus_filename: Str, directory of the corpus file
    :param on_memory: Boolean, True or False
    :param corpus_lines: List[Str] or None, on-memory Data
    :param mode: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
    :return: sample_position -> List[int]
    """
    sample_position = [0]
    if not on_memory:
        print('Loading files for ' + stage + ' Dataset...')
        with open(corpus_filename, 'r') as f_in:
            sample_str = f_in.readline()
            while sample_str:
                sample_position.append(f_in.tell())
                sample_str = f_in.readline()
            sample_position.pop()
    else:
        assert corpus_lines
        sample_position = range(len(corpus_lines))
    return sample_position


class ClassificationDataset(Dataset):
    def __init__(self, config, vocab, stage='TRAIN', on_memory=True, corpus_lines=None, mode="TRAIN"):
        """
        Dataset for text classification based on torch.utils.data.dataset.Dataset
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param stage: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
        :param on_memory: Boolean, True or False
        :param corpus_lines: List[Str] or None, on-memory Data
        :param mode: TRAIN / PREDICT, for loading empty label
        """
        super(ClassificationDataset, self).__init__()
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file)}
        self.config = config
        self.vocab = vocab #need to add attribute vocab
        # need to load the label_to_tokenized_text dict here
        self.attribute_map, self.index2attribute = vocab.v2i['attribute'], vocab.i2v['attribute']
        self.attributes_list = []
        for i in range(0, len(self.attribute_map)):
            self.attributes_list.append(self.index2attribute[i])

        self.gating_dict = {"exist":0, "none":1}
        self.PAD_token_idx = self.vocab.v2i['token']['[PAD]']
        self.max_value_length = self.config.generator.max_resp_length # if set to 50, 32GB GPU would run out of GPU memory


        #self.cur_path = os.getcwd()
        #print('dataset.py path: ', self.cur_path)
        #self.model_name = os.path.join(self.cur_path, 'hf_sp_ichiba_from_scratch_512')
        self.model_name = 'bert-base-chinese'
        print('pretrained bert model path: ', self.model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.on_memory = on_memory
        self.data = corpus_lines
        self.max_input_length = self.config.text_encoder.max_length
        self.corpus_file = self.corpus_files[stage]
        self.sample_position = get_sample_position(self.corpus_file, self.on_memory, corpus_lines, stage)
        self.corpus_size = len(self.sample_position)
        self.mode = mode

        self.token_map = self.vocab.v2i['token']
        self.bert_token2idx = self.bert_tokenizer.vocab
        self.bert_idx2token = {}
        for k, v in self.bert_token2idx.items():
            self.bert_idx2token[v] = k


    def __len__(self):
        """
        get the number of samples
        :return: self.corpus_size -> Int
        """
        return self.corpus_size

    def __getitem__(self, index):
        """
        sample from the overall corpus
        :param index: int, should be smaller in len(corpus)
        :return: sample -> Dict{'token': List[Str], 'label': List[Str], 'token_len': int}
        """
        if index >= self.__len__():
            raise IndexError
        if not self.on_memory:
            position = self.sample_position[index]
            with open(self.corpus_file) as f_in:
                f_in.seek(position)
                sample_str = f_in.readline()
        else:
            sample_str = self.data[index]
        return self._preprocess_sample(sample_str)

    def _preprocess_sample(self, sample_str):
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
        """
        raw_sample = json.loads(sample_str)
        ## get tokenized text for each label and combine them to get generate_y for each attribute
        attributes = self.vocab.v2i['attribute']
        label_nestedlist = raw_sample['label_nestedlist']
        sample_labels_nogroup = []
        example_gatelabels_attribute = []
        generation_labels = defaultdict()
        for each in label_nestedlist:
            groupid = each[0]
            tagids = each[1:]
            # print('length of tagids: ', len(tagids))
            # sample_labels_nogroup += tagids
            example_gatelabels_attribute.append(groupid)
            generation_value_label_4each_attribute = tagids[0]
            firsttag_temp = self.bert_tokenizer.encode(generation_value_label_4each_attribute, add_special_tokens=False)
            y_tokenized_eachattr = [self.token_map[self.bert_idx2token[w_id]] for w_id in firsttag_temp]
            for j, each_tagid in enumerate(tagids):
               if j != 0:
                   eachtag_temp = self.bert_tokenizer.encode(each_tagid, add_special_tokens=False)
                   eachtag_tokenized = [self.token_map[self.bert_idx2token[w_id]] for w_id in eachtag_temp]
                   y_tokenized_eachattr += [self.token_map['[SEP]']] + eachtag_tokenized
            #for j in range(1, len(tagids)):
            #    generation_value_label_4each_attribute += ['[SEP]'] + tagids_tokenized_text_154[tagids[j]]
            generation_labels[groupid]=y_tokenized_eachattr+[self.token_map['[EOS]']]
            # print(generation_value_label_4each_attribute)
        ## tokenized generation_labels
        generate_y, generate_y_lengths, gating_label = [], [], []
        for each_attribute in self.attributes_list:
            if each_attribute in generation_labels.keys():
                #y_tokenized_temp = self.bert_tokenizer.encode(generation_labels[each_attribute], add_special_tokens=False)
                #y_tokenized = [self.token_map[self.bert_idx2token[w_id]] for w_id in y_tokenized_temp]
                y_tokenized = generation_labels[each_attribute]
                if len(y_tokenized) < self.max_value_length:
                    generate_y_lengths.append(len(y_tokenized))
                    padding_y = [self.PAD_token_idx for _ in range(0, self.max_value_length - len(y_tokenized))]
                    y_tokenized += padding_y
                else:
                    generate_y_lengths.append(self.max_value_length)
                    y_tokenized=y_tokenized[:self.max_value_length]
                generate_y.append(y_tokenized)
                gating_label.append(self.gating_dict["exist"])
            else:
                generate_y_lengths.append(2)
                tagert = [self.token_map['[NONE]'], self.token_map['[EOS]']]
                padding_y = [self.PAD_token_idx for _ in range(0, self.max_value_length-2)]
                generate_y.append(tagert+padding_y)
                gating_label.append(self.gating_dict["none"])

        sample = {'token': [], 'label': [], 'label_nestedlist':raw_sample['label_nestedlist'],'generate_y':generate_y,'y_lengths':generate_y_lengths,'gating_label':gating_label}
        for k in raw_sample.keys():
            if k == 'token':
                #sample[k] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_sample[k]]
                #sample[k] = self.bert_tokenizer.encode(raw_sample[k], add_special_tokens=True)
                sample[k] = [self.token_map[w] for w in raw_sample[k]]
                #print('sample token length: ', len(sample[k]))
            elif k == 'label':
                k = 'label'
                sample[k] = []
                for v in raw_sample['label']:
                    if v not in self.vocab.v2i[k].keys():
                        logger.warning('Vocab not in ' + k + ' ' + v)
                    else:
                        sample[k].append(self.vocab.v2i[k][v])
        if not sample['token']:
            #sample['token'].append(self.vocab.padding_index)
            sample['token'].append(0)
        if self.mode == 'TRAIN':
            assert sample['label'], 'Label is empty'
        else:
            sample['label'] = [0]
        sample['token_len'] = min(len(sample['token']), self.max_input_length)
        padding = [self.PAD_token_idx for _ in range(0, self.max_input_length - len(sample['token']))]
        #padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token']))]
        sample['token'] += padding
        sample['token'] = sample['token'][:self.max_input_length]
        return sample
