#!/usr/bin/env python
# coding:utf-8
import json

import helper.logger as logger
from models.model import JPAVE
import torch
import sys
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss
from train_modules. trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
import time


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")


def train(config):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=1,
                         max_size=50000)

    # get data
    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab)

    # build up model
    jpave = JPAVE(config, corpus_vocab, model_mode='TRAIN')
    jpave.to(config.train.device_setting.device)
    # define training objective & optimizer
    criterion = ClassificationLoss()
    optimize = set_optimizer(config, jpave)

    # get epoch trainer
    trainer = Trainer(model=jpave,
                      criterion=criterion,
                      optimizer=optimize,
                      vocab=corpus_vocab,
                      config=config)

    # set origin log
    best_epoch = [-1, -1]
    best_performance = [0.0, 0.0]
    model_checkpoint = config.train.checkpoint.dir
    model_name = config.model.type
    wait = 0
    if not os.path.isdir(model_checkpoint):
        os.mkdir(model_checkpoint)
    else:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
        latest_model_file = ''
        for model_file in dir_list[::-1]:
            if model_file.startswith('best_micro'):
                latest_model_file = model_file
                break
                #continue
            else:
                continue
                #latest_model_file = model_file
                #break
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            logger.info('Loading Previous Checkpoint...')
            logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
            best_performance, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                       model=jpave,
                                                       config=config,
                                                       optimizer=optimize)
            logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                best_performance[0], best_performance[1]))

    # train
    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        start_time = time.time()
        trainer.train(train_loader,
                      epoch)
        trainer.eval(train_loader, epoch, 'TRAIN')
        performance = trainer.eval(dev_loader, epoch, 'DEV')
        # saving best model and check model
        if not (performance['micro_f1'] >= best_performance[0] or performance['macro_f1'] >= best_performance[1]):
            wait += 1
            if wait % config.train.optimizer.lr_patience == 0:
                logger.warning("Performance has not been improved for {} epochs, updating learning rate".format(wait))
                trainer.update_lr()
            if wait == config.train.optimizer.early_stopping:
                logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
                               .format(wait))
                break

        if performance['micro_f1'] > best_performance[0]:
            wait = 0
            logger.info('Improve Micro-F1 {}% --> {}%'.format(best_performance[0], performance['micro_f1']))
            best_performance[0] = performance['micro_f1']
            best_epoch[0] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': jpave.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
        if performance['macro_f1'] > best_performance[1]:
            wait = 0
            logger.info('Improve Macro-F1 {}% --> {}%'.format(best_performance[1], performance['macro_f1']))
            best_performance[1] = performance['macro_f1']
            best_epoch[1] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': jpave.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, 'best_macro_' + model_name))

        if epoch % 80 == 1:
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': jpave.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))

        logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_micro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=jpave,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[0], 'TEST')

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=jpave,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[1], 'TEST')

    return


if __name__ == "__main__":
    configs = Configure(config_json_file='config/jpave-mepave.json')

    if 'cuda' in configs.train.device_setting.device:
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    logger.Logger(configs)

    if not os.path.isdir(configs.train.checkpoint.dir):
        os.mkdir(configs.train.checkpoint.dir)

    train(configs)
