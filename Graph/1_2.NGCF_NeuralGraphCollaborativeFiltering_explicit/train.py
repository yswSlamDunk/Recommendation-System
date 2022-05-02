from re import S
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


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # config는 ConfigParser 클래스의 객체

    logger = config.get_logger('train')
    # logger는 logging의 객체로, 코드에서 발생하는 이벤트를 추적하기 위한 패키지의 클래스
    # 이때, logging의 level은 Debug로 설정됨

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    # config.init_obj('data_loader', module_data)를 통해서 data_loader를 생성하고, valid_data_loader를 생성함
    # 이 부분에서 코드를 수정할 필요 있음(user, item index 유무에 따른 분류 등)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    # model을 생성했고, console에 출력 및 logging에 기록

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimzier, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimzer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_schduler', torch.optim.lr_scheduler, optimzer)

    trainer = Trainer(model, criterion, metrics, optimzer,
                      config=config,
                      device=device,
                      data_laoder=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='NGCF_explicit')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    # 기존에 돌린 적이 있는지 없는지에 해당하는 값
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from defualt values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float,
                   target='optimzier; args; lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader; agrs; batch_size')
    ]

    config = ConfigParser.from_args(args, options)
    # optios에 있는 값들은 args.add_argument을 통해 argparse의 값으로 들어가게 됨
    # config.json 파일을 읽어 앞서 생성된 argparse와 합쳐지는 과정이 존재???
    # ConfigParser.from_args(args, options)를 통해 ConfigParser의 생성자에 필요한 config, resume, modification을 생성하고, 생성자를 호출하여 ConfigParser의 객체를 생성함
    main(config)
