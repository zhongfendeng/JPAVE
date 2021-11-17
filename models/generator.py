#!/usr/bin/env python
# coding:utf-8
import os

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np
import json


class Generator(nn.Module):
    def __init__(self, vocab, lang, shared_emb, vocab_size, hidden_size, dropout, attributes, nb_gate, config):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang_index2word = lang
        #self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.attributes = attributes
        self.config = config

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # temporarliy random initializ the attribute emb, later can try to load the pre-trained emb
        self.cur_path = os.getcwd()
        self.pretrained_attr_emb_dir = os.path.join(self.cur_path, 'mepave_attribute_embeddings.json')

        self.attribute_map, self.index2attribute = vocab.v2i['attribute'], vocab.i2v['attribute']
        self.lookup_table = torch.normal(0, 0.1, (len(self.attribute_map), hidden_size))
        self.load_pretrained_attribute_emb(pretrained_dir=self.pretrained_attr_emb_dir)
        self.Attribute_emb = nn.Embedding(len(self.attributes), hidden_size)
        self.Attribute_emb.weight.data = self.lookup_table
        #self.Attribute_emb.weight.data.normal_(0, 0.1)

    def load_pretrained_attribute_emb(self, pretrained_dir):
        """
        load pretrained file
        :param embedding_dim: Int, configure.embedding.field.dimension
        :param vocab_map: vocab.v2i[field] -> Dict{v:id}
        :param vocab_name: field
        :param pretrained_dir: str, file path
        """
        #logger.info('Loading {}-dimension {} embedding from pretrained file: {}'.format(
        #    embedding_dim, vocab_name, pretrained_dir))
        with open(pretrained_dir) as f_in:
            data = json.load(f_in)
            #num_pretrained_vocab = 0
            for i in range(len(self.index2attribute)):
                self.lookup_table[i]=torch.FloatTensor(data[str(i)])


    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches,
                use_teacher_forcing, slot_temp):
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if 'cuda' in self.config.train.device_setting.device:
            USE_CUDA = True
        else:
            USE_CUDA = False
        if USE_CUDA:
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()

        # Get attribute embedding
        attribute_emb_dict = {}
        for i, attribute in enumerate(slot_temp):
            attribute_emb = self.Attribute_emb(torch.tensor([i]).cuda())
            attribute_emb_dict[attribute]=attribute_emb
            attribute_emb_exp = attribute_emb.expand_as(encoded_hidden)
            if i == 0:
                attribute_emb_arr = attribute_emb_exp.clone()
            else:
                attribute_emb_arr = torch.cat((attribute_emb_arr, attribute_emb_exp), dim=0)

        if self.config.generator.parallel_decode:
            # Compute pointer-generator output, puting all attributes in one batch
            decoder_input = self.dropout_layer(attribute_emb_arr).view(-1, self.hidden_size)  # (batch*|attr|) * emb
            hidden = encoded_hidden.repeat(1, len(slot_temp), 1)  # 1 * (batch*|attr|) * emb
            words_point_out = [[] for i in range(len(slot_temp))]
            words_class_out = []

            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0:
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size())
                if USE_CUDA: p_context_ptr = p_context_ptr.cuda()

                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words = [self.lang_index2word[w_idx.item()] for w_idx in pred_word]

                for si in range(len(slot_temp)):
                    words_point_out[si].append(words[si * batch_size:(si + 1) * batch_size])

                all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab,
                                                               (len(slot_temp), batch_size, self.vocab_size))

                if use_teacher_forcing:
                    decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1, 0)))
                else:
                    decoder_input = self.embedding(pred_word)

                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            # Compute pointer-generator output, decoding each attribute one-by-one
            words_point_out = []
            counter = 0
            for attribute in slot_temp:
                hidden = encoded_hidden
                #print('encoded hidden size: ', encoded_hidden.size())
                #bs, max_input_len, h_size = encoded_hidden.size(0),encoded_hidden.size(1),encoded_hidden.size(2) # 64, max_input_len (46), 768
                #print('batch size: ', bs)
                #print('max_input_len in generator: ', max_input_len)
                #print('hidden size: ', h_size)
                words = []
                attribute_emb = attribute_emb_dict[attribute]
                decoder_input = self.dropout_layer(attribute_emb).expand(batch_size, self.hidden_size)
                for wi in range(max_res_len):
                    #dec_state, hidden = self.gru(decoder_input, hidden)
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0:
                        all_gate_outputs[counter] = self.W_gate(context_vec)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    p_context_ptr = torch.zeros(p_vocab.size())
                    if USE_CUDA: p_context_ptr = p_context_ptr.cuda()
                    #print("story shape: ", story.shape)
                    #print("p_prob shape: ", prob.shape)
                    p_context_ptr.scatter_add_(1, story, prob)
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.lang_index2word[w_idx.item()] for w_idx in pred_word])
                    all_point_outputs[counter, :, wi, :] = final_p_vocab
                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, counter, wi])  # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)
                    if USE_CUDA: decoder_input = decoder_input.cuda()
                counter += 1
                words_point_out.append(words)

        return all_point_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        #max_len = max_input_len
        #scores_ = torch.zeros((scores_old.shape[0], max_len), dtype=torch.float).cuda()
        for i, l in enumerate(lens):
            #scores_.data[i, :l] = scores_old.data[i, :l]
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores

