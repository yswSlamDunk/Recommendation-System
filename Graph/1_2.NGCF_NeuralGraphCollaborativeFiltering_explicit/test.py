import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_laoder', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, the print to console
    model = config.init_obj('arch', module_arch)
    logger.ingo(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParaller(model, device_ids=device_ids)

    # get fuction handles of loss and metris
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimzier, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trinable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimzier = config.init_obj('optimzier', torch.optim, trinable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimzier)

    trainer = Trainer(model, criterion, metrics, optimzier,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(descroption='NGCF_Explicit')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config gile path (default : None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default : all)')

    config = ConfigParser.from_args(args)
    main(config)
