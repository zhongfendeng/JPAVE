#!/usr/bin/env python
# coding:utf-8

import numpy as np
import torch
import helper.logger as logger
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
from transformers import BertModel, BertTokenizer

INIT_FUNC = {
    'uniform': uniform_,
    'kaiming_uniform': kaiming_uniform_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_normal': kaiming_normal_
}


class EmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',
                 negative_slope=0, mode_fan='fan_in',
                 activation_type='linear',
                 ):
        """
        embedding layer
        :param vocab_map: vocab.v2i[filed] -> Dict{Str: Int}
        :param embedding_dim: Int, config.embedding.token.dimension
        :param vocab_name: Str, 'token' or 'label'
        :param config: helper.configure, Configure Object
        :param padding_index: Int, index of padding word
        :param pretrained_dir: Str,  file path for the pretrained embedding file
        :param model_mode: Str, 'TRAIN' or 'EVAL', for initialization
        :param initial_type: Str, initialization type
        :param negative_slope: initialization config
        :param mode_fan: initialization config
        :param activation_type: None
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = torch.nn.Dropout(p=config['embedding'][vocab_name]['dropout'])
        self.embedding = torch.nn.Embedding(len(vocab_map), embedding_dim, padding_index)

        #load bert model
        self.model_name = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name)
        bert_params = list(self.bert_model.base_model.parameters())
        self.bert_word_emb = bert_params[0]  # first Parameter from bert_params is the word embedding weight
        self.bert_vocab_v2i=self.tokenizer.vocab
        #self.bert_config = BertConfig.from_pretrained(self.model_name, output_hidden_states=True)
        #self.bert_model = BertModel.from_pretrained(self.model_name, config=self.bert_config)

        # initialize lookup table
        assert initial_type in INIT_FUNC
        if initial_type.startswith('kaiming'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=negative_slope,
                                                        mode=mode_fan,
                                                        nonlinearity=activation_type)
        elif initial_type.startswith('xavier'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        gain=torch.nn.init.calculate_gain(activation_type))
        else:
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=-0.25,
                                                        b=0.25)

        if model_mode == 'TRAIN' and config['embedding'][vocab_name]['type'] == 'pretrain':
            self.load_pretrained(embedding_dim, vocab_map, vocab_name, self.bert_vocab_v2i)

        #if padding_index is not None:
        #    self.lookup_table[padding_index] = 0.0
        self.embedding.weight.data.copy_(self.lookup_table)
        self.embedding.weight.requires_grad = True
        del self.lookup_table

    def load_pretrained(self, embedding_dim, vocab_map, vocab_name, bert_vocab_v2i):
        """
        load pretrained file
        :param embedding_dim: Int, configure.embedding.field.dimension
        :param vocab_map: vocab.v2i[field] -> Dict{v:id}
        :param vocab_name: field
        :param pretrained_dir: str, file path
        """
        logger.info('Loading {}-dimension {} embedding from bert pretrained word embedding parameter'.format(
            embedding_dim, vocab_name))
        num_pretrained_vocab = 0
        for k, v in vocab_map.items(): # k is index, v is token
            if v in bert_vocab_v2i:
                index_bertvocab = bert_vocab_v2i[v]
                curr_token_embedding = self.bert_word_emb[index_bertvocab]
                self.lookup_table[k] = curr_token_embedding
                num_pretrained_vocab += 1
        logger.info('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        logger.info('Pretrained vocab embedding has %d / %d' % (num_pretrained_vocab, len(vocab_map)))

    def forward(self, vocab_id_list):
        """
        :param vocab_id_list: torch.Tensor, (batch_size, max_length)
        :return: embedding -> torch.FloatTensor, (batch_size, max_length, embedding_dim)
        """
        embedding = self.embedding(vocab_id_list)
        return self.dropout(embedding)
