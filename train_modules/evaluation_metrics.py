#!/usr/bin/env python
# coding:utf-8

import numpy as np


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

def parse_values_for_attribute(each_attribute_values_list, token2id_map, id2token_map):
    #attr_name = id2attribute_map[attribute_index]
    attr_values = []
    one_value = []
    for each_word_idx in each_attribute_values_list:
        if int(each_word_idx) == token2id_map['[EOS]']:
            break
        if int(each_word_idx) == token2id_map['[SEP]']:  ## split token for values
            attr_values.append(one_value)
            one_value = []
        else:
            one_value.append(id2token_map[int(each_word_idx)])
    # add the last generated value (either exit above loop by EOS or just finish the loop)
    if len(one_value) != 0:
        attr_values.append(one_value)
    return attr_values


def concat_attribute_name_and_value(attr_name, attr_values):
    # concate attribute name and value together
    attr_value_pairs_withnone = []
    attr_value_pairs = []
    for each_value in attr_values:
        each_value_str = ""
        for w in each_value:
            each_value_str += w
        attr_value_pair = attr_name + "-" + each_value_str
        attr_value_pairs_withnone.append(attr_value_pair)
        if each_value_str != '[NONE]':
            attr_value_pairs.append(attr_value_pair)
    return attr_value_pairs, attr_value_pairs_withnone


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        if len(pred)==0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count



def evaluate_value_generation_only(vocab, epoch_generation_y, epoch_target_generation_y, epoch_labels, epoch_gate_pred, epoch_attributes):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_attribute: List[List[int]], shape: num_examples, num_attributes (12), each element in the inner list is either 0(none) or 1(ptr)
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param epoch_generation_y: List[List[List[int]]], generated word index, shape: num_examples, num_attributes, max_resp_length
    :param epoch_target_generation_y: List[List[List[int]]], target word index, shape: num_examples, num_attributes, max_resp_length
    :param epoch_gate_pred: List[List[List[Float]]], gate class label list, shape: num_examples, num_attributes, num_gate_classes (only 2 classes here, ptr-0 and none-1)
    :param generation_vocab: should use model's own token vocab in generator to predict the vaule word in each step, not the OrderedDict from tokenizer of Bert model anymore
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_generation_y) == len(epoch_target_generation_y), 'mismatch between prediction and ground truth for evaluation'
    token2id = vocab.v2i['token']
    id2token = vocab.i2v['token']
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    id2attribute = vocab.i2v['attribute']
    attribute2id = vocab.v2i['attribute']
    epoch_gold_label = list()
    # get id label name of ground truth
    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    epoch_predict_id_list = []
    epoch_pred_attribute_id_list = []
    epoch_target_attribute_id_list = []

    total, example_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for sample_gold, sample_generation_y, sample_target_generation_y, sample_gate_pred, sample_gold_attribute in zip(epoch_labels, epoch_generation_y, epoch_target_generation_y, epoch_gate_pred, epoch_attributes):
        sample_predict_id_list = []
        sample_generated_avpairs = []
        sample_target_avpairs = []

        example_pred_attribute_id_list = []
        example_target_attribute_id_list = []
        attr_index = 0
        for each_attribute_value_generated, each_attribute_target_value, each_attribute_gate_pred, gold_attribute in zip(sample_generation_y, sample_target_generation_y, sample_gate_pred, sample_gold_attribute):
            np_each_attribute_gate_pred = np.array(each_attribute_gate_pred, dtype=np.float32)
            if np_each_attribute_gate_pred[0] > np_each_attribute_gate_pred[1]:  # i.e., this attribute is classified as ptr (copying words from input for its vaule)
                example_pred_attribute_id_list.append(attr_index)
            if gold_attribute == 0:
                example_target_attribute_id_list.append(attr_index)

            attr_name = id2attribute[attr_index]
            generated_attr_values = parse_values_for_attribute(each_attribute_value_generated, token2id, id2token)
            for each_value in generated_attr_values:
                each_value_str = ""
                for word in each_value:
                    each_value_str += word
                if each_value_str != "" and each_value_str in label2id:
                    each_labelid = label2id[each_value_str]
                    sample_predict_id_list.append(each_labelid)
            target_attr_values = parse_values_for_attribute(each_attribute_target_value, token2id, id2token)
            # concate attribute name and value together
            generated_attr_value_pairs, generated_attr_value_pairs_withnone = concat_attribute_name_and_value(attr_name, generated_attr_values)
            target_attr_value_pairs, target_attr_value_pairs_withnone = concat_attribute_name_and_value(attr_name, target_attr_values)
            sample_generated_avpairs += generated_attr_value_pairs
            sample_target_avpairs += target_attr_value_pairs
            attr_index += 1

        epoch_predict_id_list.append(sample_predict_id_list)
        epoch_pred_attribute_id_list.append(example_pred_attribute_id_list)
        epoch_target_attribute_id_list.append(example_target_attribute_id_list)

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1



        ## compute acc or prf for value prediction of each example, input are a list of attribute name and generated value pair,
        # and a list of attribute name and ground truth value pair
        temp_acc = compute_acc(set(sample_target_avpairs), set(sample_generated_avpairs), id2attribute)
        example_acc += temp_acc

        if set(sample_target_avpairs) == set(sample_generated_avpairs):
            joint_acc += 1
        total += 1

        # Compute prediction joint F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(set(sample_target_avpairs), set(sample_generated_avpairs))
        F1_pred += temp_f1
        F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    example_acc_score = example_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    #return joint_acc_score, F1_score, example_acc_score


    attribute_metrics = compute_acc_pfr_forattribute(epoch_pred_attribute_id_list, epoch_target_attribute_id_list, id2attribute)
    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}, attribute_metrics, joint_acc_score, F1_score, example_acc_score, epoch_predict_id_list, epoch_pred_attribute_id_list, epoch_target_attribute_id_list


def evaluate_value_generation_withcls(vocab, epoch_generation_y, epoch_target_generation_y, epoch_predicts, epoch_labels, epoch_gate_pred, epoch_attributes, threshold=0.5):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_attribute: List[List[int]], shape: num_examples, num_attributes (12), each element in the inner list is either 0(none) or 1(ptr)
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param epoch_generation_y: List[List[List[int]]], generated word index, shape: num_examples, num_attributes, max_resp_length
    :param epoch_target_generation_y: List[List[List[int]]], target word index, shape: num_examples, num_attributes, max_resp_length
    :param epoch_gate_pred: List[List[List[Float]]], gate class label list, shape: num_examples, num_attributes, num_gate_classes (only 2 classes here, ptr-0 and none-1)
    :param generation_vocab: should use model's own token vocab in generator to predict the vaule word in each step, not the OrderedDict from tokenizer of Bert model anymore
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_generation_y) == len(epoch_target_generation_y), 'mismatch between prediction and ground truth for evaluation'
    token2id = vocab.v2i['token']
    id2token = vocab.i2v['token']
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    id2attribute = vocab.i2v['attribute']
    attribute2id = vocab.v2i['attribute']
    epoch_gold_label = list()
    # get id label name of ground truth
    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    epoch_predict_id_list = []
    epoch_pred_attribute_id_list = []
    epoch_target_attribute_id_list = []

    total, example_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for sample_predict, sample_gold, sample_generation_y, sample_target_generation_y, sample_gate_pred, sample_gold_attribute in zip(epoch_predicts, epoch_labels, epoch_generation_y, epoch_target_generation_y, epoch_gate_pred, epoch_attributes):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list_cls = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list_cls.append(sample_predict_descent_idx[j])

        sample_predict_id_list = []
        sample_generated_avpairs = []
        sample_target_avpairs = []

        example_pred_attribute_id_list = []
        example_target_attribute_id_list = []
        attr_index = 0
        for each_attribute_value_generated, each_attribute_target_value, each_attribute_gate_pred, gold_attribute in zip(sample_generation_y, sample_target_generation_y, sample_gate_pred, sample_gold_attribute):
            np_each_attribute_gate_pred = np.array(each_attribute_gate_pred, dtype=np.float32)
            if np_each_attribute_gate_pred[0] > np_each_attribute_gate_pred[1]:  # i.e., this attribute is classified as ptr (copying words from input for its vaule)
                example_pred_attribute_id_list.append(attr_index)
            if gold_attribute == 0:
                example_target_attribute_id_list.append(attr_index)

            attr_name = id2attribute[attr_index]
            generated_attr_values = parse_values_for_attribute(each_attribute_value_generated, token2id, id2token)
            for each_value in generated_attr_values:
                each_value_str = ""
                for word in each_value:
                    each_value_str += word
                if each_value_str != "" and each_value_str in label2id:
                    each_labelid = label2id[each_value_str]
                    sample_predict_id_list.append(each_labelid)
            target_attr_values = parse_values_for_attribute(each_attribute_target_value, token2id, id2token)
            # concate attribute name and value together
            generated_attr_value_pairs, generated_attr_value_pairs_withnone = concat_attribute_name_and_value(attr_name, generated_attr_values)
            target_attr_value_pairs, target_attr_value_pairs_withnone = concat_attribute_name_and_value(attr_name, target_attr_values)
            sample_generated_avpairs += generated_attr_value_pairs
            sample_target_avpairs += target_attr_value_pairs
            attr_index += 1

        # join the sample_predict_id_list_cls from classification and sample_predict_id_list from generation
        sample_predict_id_list_joined = list(set(sample_predict_id_list_cls + sample_predict_id_list))
        epoch_predict_id_list.append(sample_predict_id_list_joined)
        epoch_pred_attribute_id_list.append(example_pred_attribute_id_list)
        epoch_target_attribute_id_list.append(example_target_attribute_id_list)

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1



        ## compute acc or prf for value prediction of each example, input are a list of attribute name and generated value pair,
        # and a list of attribute name and ground truth value pair
        temp_acc = compute_acc(set(sample_target_avpairs), set(sample_generated_avpairs), id2attribute)
        example_acc += temp_acc

        if set(sample_target_avpairs) == set(sample_generated_avpairs):
            joint_acc += 1
        total += 1

        # Compute prediction joint F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(set(sample_target_avpairs), set(sample_generated_avpairs))
        F1_pred += temp_f1
        F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    example_acc_score = example_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    #return joint_acc_score, F1_score, example_acc_score


    attribute_metrics = compute_acc_pfr_forattribute(epoch_pred_attribute_id_list, epoch_target_attribute_id_list, id2attribute)
    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}, attribute_metrics, joint_acc_score, F1_score, example_acc_score


def compute_acc_pfr_forattribute(epoch_pred, epoch_gold, id2attribute):
    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2attribute))] for _ in range(len(id2attribute.keys()))]
    right_count_list = [0 for _ in range(len(id2attribute.keys()))]
    gold_count_list = [0 for _ in range(len(id2attribute.keys()))]
    predicted_count_list = [0 for _ in range(len(id2attribute.keys()))]
    for sample_pred, sample_gold in zip(epoch_pred, epoch_gold):
        for i in range(len(confusion_count_list)):
            for predict_id in sample_pred:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for attr in sample_pred:
                if gold == attr:
                    right_count_list[gold] += 1

        # count for the predicted items
        for attr in sample_pred:
            predicted_count_list[attr] += 1
    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2attribute.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                    precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}



def post_process_labels_from_generated_values(generated_value, SEP_token_index, texttoken2labelids_map, original_predict_id_list):
    """
        :param generated_value: List[int]], max_resp_length (i.e., max_value_length)
        :param SEP_token_index: SEP_token_index of the special token [SEP] in the vocab of bert tokenizer
        :param texttoken2labelids_map: Dict, key is token index (str not int), value is a set of label ids whose text include this token
        :param original_predict_id_list: List[int], the predicted label id list by the classifier.
        :return:  new_predict_id_list -> List[int], adding extra labels obtained from the generated value.
    """
    new_predict_id_list = original_predict_id_list
    tokens_for_onelabel = []
    prev_labels_ids_set = set(texttoken2labelids_map[str(generated_value[0])])
    for i in range(1,len(generated_value)):
        if generated_value[i] == SEP_token_index:
            # add the label ids obtained from intersection of previous tokens into the new list and clear the prev_labels_ids_set
            for each in prev_labels_ids_set:
                if each not in new_predict_id_list:
                    new_predict_id_list.append(each)
            prev_labels_ids_set = set()
        else:
            curr_label_ids_set = set(texttoken2labelids_map[str(generated_value[i])])
            if len(prev_labels_ids_set) == 0:
                prev_labels_ids_set = curr_label_ids_set
            else:
                prev_labels_ids_set = prev_labels_ids_set.intersection(curr_label_ids_set)
    return new_predict_id_list



def evaluate(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}
