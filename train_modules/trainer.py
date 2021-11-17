#!/usr/bin/env python
# coding:utf-8
import os

import helper.logger as logger
from train_modules.evaluation_metrics import evaluate, evaluate_value_generation_only, evaluate_value_generation_withcls
import torch
import tqdm
import json
import numpy as np


class Trainer(object):
    def __init__(self, model, criterion, optimizer, vocab, config):
        """
        :param model: Computational Graph
        :param criterion: train_modules.ClassificationLoss object
        :param optimizer: optimization function for backward pass
        :param vocab: vocab.v2i -> Dict{'token': Dict{vocabulary to id map}, 'label': Dict{vocabulary
        to id map}}, vocab.i2v -> Dict{'token': Dict{id to vocabulary map}, 'label': Dict{id to vocabulary map}}
        :param config: helper.Configure object
        """
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer

    def update_lr(self):
        """
        (callback function) update learning rate according to the decay weight
        """
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def run(self, data_loader, epoch, stage, mode='TRAIN'):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        predict_probs = []
        target_labels = []
        target_attributes = []
        target_generation_y = []
        generation_y_indices = []
        generation_words = []
        gate_pred_probs = []
        total_loss = 0.0
        total_loss_attr_pred=0.0
        num_batch = data_loader.__len__()

        for batch in tqdm.tqdm(data_loader):
            logits, loss_generator, generation_y, generation_y_word, loss_attr_predictor, attrs_pred = self.model(batch, mode)

            recursive_constrained_params = None
            loss_value_classifier = self.criterion(logits,
                    batch['label'].to(self.config.train.device_setting.device),
                                  recursive_constrained_params)
            
            loss = loss_generator + loss_attr_predictor
            total_loss += loss.item()
            total_loss_attr_pred += loss_attr_predictor.item()

            if mode == 'TRAIN':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])
            target_attributes.extend((batch['gating_label']))
            target_generation_y.extend((batch['generate_y'])) #List[List[List[int]]], shape: num_exmaples, num_attributes, max_resp_len
            generation_y_word_index = torch.argmax(generation_y, dim=3).cpu().tolist()
            generation_y_indices.extend(generation_y_word_index)
            #print('gate_pred: ', gate_pred)
            attr_pred_cls = torch.softmax(attrs_pred, dim=2).cpu().tolist()
            #print('gate_pred_cls: ', gate_pred_cls)
            gate_pred_probs.extend(attr_pred_cls)
            #generation_words.extend(generation_y_word)
        total_loss = total_loss / num_batch
        total_loss_attr_pred /= num_batch
        if mode == 'EVAL':
            generation_vocab = self.vocab.v2i['token']
            metrics, attribute_metrics, joint_acc_score_ptr, F1_score_ptr, example_acc_score_ptr, _, _, _ = evaluate_value_generation_only(self.vocab, generation_y_indices, target_generation_y, target_labels, gate_pred_probs, target_attributes)

            if stage == 'TEST':
                _, _, _, _, _, epoch_values_pred, epoch_attrs_pred, epoch_attrs_gold = evaluate_value_generation_only(
                    self.vocab, generation_y_indices, target_generation_y, target_labels, gate_pred_probs,
                    target_attributes)
                save_predictions_withgeneration(self.vocab, target_labels, epoch_values_pred, epoch_attrs_gold, epoch_attrs_pred)

            logger.info("%s value performance at epoch %d --- Precision: %f, "
                        "Recall: %f, Micro-F1: %f, Macro-F1: %f, Loss: %f.\n"
                        % (stage, epoch,
                           metrics['precision'], metrics['recall'], metrics['micro_f1'], metrics['macro_f1'],
                           total_loss))

            logger.info("%s attribute performance at epoch %d --- Precision: %f, "
                        "Recall: %f, Micro-F1: %f, Macro-F1: %f, Gate Loss: %f.\n"
                        % (stage, epoch,
                           attribute_metrics['precision'], attribute_metrics['recall'], attribute_metrics['micro_f1'], attribute_metrics['macro_f1'], total_loss_attr_pred))

            logger.info("%s value generation performance at epoch %d --- Joint ACC: %f, "
                        "Instance-level ACC: %f, Joint F1: %f.\n"
                        % (stage, epoch,
                           joint_acc_score_ptr, example_acc_score_ptr, F1_score_ptr))
            return metrics

    def train(self, data_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'Train', mode='TRAIN')

    def eval(self, data_loader, epoch, stage):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='EVAL')


def save_predictions_withgeneration(vocab, epoch_values_target, epoch_values_pred, epoch_attrs_target, epoch_attrs_pred):
    assert len(epoch_values_pred) == len(epoch_values_target), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    id2attribute = vocab.i2v['attribute']
    attribute2id = vocab.v2i['attribute']
    epoch_gold_values = []
    epoch_gold_value_ids = []
    epoch_pred_values = []
    epoch_pred_value_ids = []

    epoch_gold_attributes = []
    epoch_gold_attribute_ids = []
    epoch_pred_attribute_ids = []
    epoch_pred_attributes = []
    # get id label name of ground truth
    for ind, sample_labels in enumerate(epoch_values_target):
        epoch_gold_value_ids.append({ind: sample_labels})
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_values.append({ind: sample_gold})

        ## pred values
        sample_pred_value_ids = epoch_values_pred[ind]
        epoch_pred_value_ids.append({ind: epoch_values_pred[ind]})
        sample_pred_values = [id2label[each] for each in sample_pred_value_ids]
        epoch_pred_values.append({ind: sample_pred_values})

        ## gold attributes
        sample_gold_attr_ids = epoch_attrs_target[ind]
        epoch_gold_attribute_ids.append({ind: sample_gold_attr_ids})
        sample_gold_attrs = [id2attribute[each] for each in sample_gold_attr_ids]
        epoch_gold_attributes.append({ind: sample_gold_attrs})

        ## pred attributes
        sample_pred_attr_ids = epoch_attrs_pred[ind]
        epoch_pred_attribute_ids.append({ind: sample_pred_attr_ids})
        sample_pred_attrs = [id2attribute[each] for each in sample_pred_attr_ids]
        epoch_pred_attributes.append({ind: sample_pred_attrs})


    # save groudtruth value ids and predicted value ids (nested list)
    with open('mepave_test_value_ids_groundtruth.json', 'w', encoding='utf-8') as jf:
        for line in epoch_gold_value_ids:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')
    with open('mepave_test_value_ids_predictions.json', 'w', encoding='utf-8') as jf:
        for line in epoch_pred_value_ids:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')

    with open('mepave_test_attr_ids_groundtruth.json', 'w', encoding='utf-8') as jf:
        for line in epoch_gold_attribute_ids:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')
    with open('mepave_test_attr_ids_predictions.json', 'w', encoding='utf-8') as jf:
        for line in epoch_pred_attribute_ids:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')

    with open('mepave_test_values_groundtruth.json', 'w', encoding='utf-8') as jf:
        for line in epoch_gold_values:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')
    with open('mepave_test_values_predictions.json', 'w', encoding='utf-8') as jf:
        for line in epoch_pred_values:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')

    with open('mepave_test_attrs_groundtruth.json', 'w', encoding='utf-8') as jf:
        for line in epoch_gold_attributes:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')
    with open('mepave_test_attrs_predictions.json', 'w', encoding='utf-8') as jf:
        for line in epoch_pred_attributes:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')

