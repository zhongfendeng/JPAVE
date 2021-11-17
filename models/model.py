#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
import torch
import numpy as np
from models.text_encoder import TextEncoder
from models.generator import Generator
from models.embedding_layer import EmbeddingLayer
import models.label_embedding_layer
from models.utils.masked_cross_entropy import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import json



class JPAVE(nn.Module):
    def __init__(self, config, vocab, model_mode='TRAIN'):
        """
        JPAVE Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(JPAVE, self).__init__()
        self.config = config
        self.vocab = vocab
        #self.vocab = self.vocab.to(config.train.device_setting.device)
        self.device = config.train.device_setting.device

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        self.index2label = vocab.i2v['label']
        self.index2token = vocab.i2v['token']
        self.attribute_map, self.index2attribute = vocab.v2i['attribute'], vocab.i2v['attribute']
        self.attributes_list = []
        for i in range(0, len(self.attribute_map)):
            self.attributes_list.append(self.index2attribute[i])

        #self.token_map = self.token_map.to(config.train.device_setting.device)

        self.token_embedding = EmbeddingLayer(
            vocab_map=self.index2token,
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
            padding_index=vocab.padding_index,
            pretrained_dir=config.embedding.token.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.token.init_type
        )
        self.token_embedding.to(self.device)

        self.label_embedding = models.label_embedding_layer.EmbeddingLayer(
            vocab_map=self.index2label,
            embedding_dim=config.embedding.label.dimension,
            vocab_name='label',
            config=config,
            padding_index=None,
            pretrained_dir=config.embedding.label.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.label.init_type
        ).to(self.device)

        # value classifier
        self.cls_linear = nn.Linear(config.embedding.label.dimension, len(self.label_map))

        # dropout
        self.cls_dropout = nn.Dropout(p=config.model.classifier.dropout)

        self.text_encoder = TextEncoder(config)
        self.text_encoder.to(self.device)

        self.gating_dict = {"exist": 0, "none": 1}
        self.nb_gate = len(self.gating_dict)
        dropout = 0.05
        self.decoder = Generator(self.vocab, self.vocab.i2v['token'], self.token_embedding.embedding, len(self.vocab.i2v['token']), config.embedding.token.dimension, dropout,
                                 self.attributes_list, self.nb_gate, self.config)



        self.cross_entorpy = nn.CrossEntropyLoss()

        # load tagmaster
        self.cur_path = os.getcwd()
        self.tagmaster = {}
        self.tagmaster_file_path = os.path.join(self.cur_path, 'tagmaster.json')
        with open(self.tagmaster_file_path, 'r') as jf:
            for i, line in enumerate(jf):
                data = json.loads(line)
                attribute = data['attribute']
                values = data['values']
                self.tagmaster[attribute] = values
            #self.tagmaster = json.load(jf)
        
    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        #params.append({'params': self.token_embedding.parameters()})
        #params.append({'params': self.label_embedding.parameters()})
        #params.append({'params': self.cls_linear.parameters()})
        #params.append({'params': self.cls_dropout.parameters()})
        params.append({'params': self.decoder.parameters()})
        return params

    @staticmethod
    def _soft_attention(text_f, label_f):
        """
        soft attention module
        :param text_f -> torch.FloatTensor, (batch_size, K, dim)
        :param label_f ->  torch.FloatTensor, (N, dim)
        :return: label_align ->  torch.FloatTensor, (batch, N, dim)
        """
        att = torch.matmul(text_f, label_f.transpose(0, 1))
        weight_label = functional.softmax(att.transpose(1, 2), dim=-1)
        label_align = torch.matmul(weight_label, text_f)
        return label_align

    def forward(self, batch, mode):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return: 
        """

        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        #print('input shape: ', batch['token'].shape)#embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
        #print('input format: ', batch['token'])#embedding = self.token_embedding(batch['token'].to("cuda:7"))

        #print('input shape: ' batch['token'].shape)# get the length of sequences for dynamic rnn, (batch_size, 1)
        seq_len = batch['token_len']
        embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
        token_output, last_hidden = self.text_encoder(embedding, seq_len)

        ## using generator here to generate values for each attribute by taking the output from text encoder as
        #the input for decoder (generator)
        encoder_outputs = token_output # token_output shape [64, max_input_len(e.g. 256), 768]
        #encoder_hiddenstates = torch.mean(token_output, dim=1, keepdim=True) # [64, 1, 768]
        encoder_hiddenstates = last_hidden.unsqueeze(1) ## last_hidden shape is [64, 768]
        #print("token_output shape: ", token_output.shape)
        #print("last_hidden shape: ", last_hidden.shape)
        encoder_hidden = encoder_hiddenstates.transpose(0, 1) # [1, batchsize, hiddensize]

        # Get the words that can be copy from the memory
        if mode=='TRAIN':
            use_teacher_forcing = False
        else:
            use_teacher_forcing = False
        batch_size = len(batch['generate_y'])
        batch_input_max_len = encoder_outputs.shape[1]
        story_temp = batch['token'][:, :batch_input_max_len]
        input_tokens = story_temp.to(self.config.train.device_setting.device) # story is the context/input text tokens that decoder wants to attend on
        max_res_len = batch['generate_y'].size(2) if mode=='TRAIN' else 10
        #print('batch token_len, ', batch['token_len'])
        all_point_outputs, all_attrs_cls_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
            encoder_hidden, encoder_outputs, batch['token_len'], input_tokens, max_res_len, batch['generate_y'].to(self.config.train.device_setting.device), \
            use_teacher_forcing, self.attributes_list)


        loss_generator = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            batch["generate_y"].to(self.config.train.device_setting.device).contiguous(),  # [:,:len(self.point_slots)].contiguous(),
            batch["y_lengths"].to(self.config.train.device_setting.device))  # [:,:len(self.point_slots)])
        attrs_pred = all_attrs_cls_outputs
        loss_attr_pred = self.cross_entorpy(attrs_pred.transpose(0, 1).contiguous().view(-1, attrs_pred.size(-1)),
                                       batch["gating_label"].to(self.config.train.device_setting.device).contiguous().view(-1))

        ## value attention and value classification
        text_feature = token_output
        label_feature = self.label_embedding(torch.arange(0, len(self.label_map)).long().to(self.device))
        text_feature = torch.mean(text_feature, dim=1, keepdim=True)
        label_aware_text_feature_ = self._soft_attention(text_feature, label_feature)
        #label_aware_text_feature = text_feature#print("label_aware_text_feature shape", label_aware_text_feature_.shape)
        label_aware_text_feature = torch.mean(label_aware_text_feature_, dim=1)
        value_cls_logits = self.cls_dropout(self.cls_linear(label_aware_text_feature.view(label_aware_text_feature.shape[0], -1)))
        return value_cls_logits, loss_generator, all_point_outputs.transpose(0, 1).contiguous(), words_point_out, loss_attr_pred, attrs_pred.transpose(0, 1).contiguous()

