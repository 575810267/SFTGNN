import logging
import os
import os.path as osp
import random
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Optional
from tqdm import tqdm
from functools import total_ordering
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_dataloader
from data.config import Config as data_config, AtomFeatureType
from data.config import CrystalDataset, JarvisTarget, MPTarget
from model.net import SFTGNN
from model.config import Config as model_config


def getLogger() -> logging.Logger:
    """
        Set up logger
    :return:
    """
    logger = logging.getLogger(data_config.getTargetName())
    logger.setLevel(logging.INFO)
    log_root = r'./logs'
    if not osp.exists(log_root):
        os.makedirs(log_root)
    log_file = osp.join(log_root, f'{model_config.model_name}_{data_config.getTargetName()}.log')
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(filename=log_file, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger


@total_ordering
class ModelPara:
    """
        Save the model parameters, as well as the epoch and evaluation metrics at that time.
        The .pth binary actually holds that object
    """

    def __init__(self, model_para, score: float, epoch: int):
        self.model_para = model_para
        self.score = score
        self.epoch = epoch

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f'model epoch: {self.epoch:0>3}, Valid MAE: {self.score:.8f}'


class TopKModelSaver:
    def __init__(self, model_para, k: int, early_stop: bool = False, patience: int = 15):
        """
            The best k models are saved during training
        :param model_para: model.state_dict()
        :param k:  The number of the best models saved
        :param early_stop: Whether to use an early stop mechanism
        :param patience: When the early stop mechanism is enabled, if the model evaluation indicators do not have better parameters after consecutive iterations, the early stop will be triggered and the training will be stopped
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.top_k = [ModelPara(model_para, float('inf'), 0) for _ in range(k)]
        self.early_stop = early_stop
        self.patience = patience
        self.counter = 0

    def step(self, model, score: float, epoch: int):
        """
        called once per epoch
        :param score: current validation metrics
        :param model: current model
        :param epoch: current epoch number for file naming
        """
        # update k optimal models
        temp_para = ModelPara(model.state_dict(), score, epoch)
        if temp_para < self.top_k[-1]:
            self.top_k.append(temp_para)
            self.top_k.sort()
            self.top_k.pop()
        else:
            self.counter += 1

    @property
    def stop(self) -> bool:
        if self.early_stop and self.counter >= self.patience:
            return True
        else:
            return False

    def state_dict(self) -> dict:
        return self.__dict__

    def load_state_dict(self, state_dict: dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

    def info(self) -> str:
        s = f'The best {self.k}:\n'
        for i in self.top_k:
            s += str(i) + '\n'
        return s.removesuffix('\n')


class TrainConfigManager:
    """
        If the hyperparameter settings for each task are different during training, you can use this type to make special modifications,
        __enter__ adjust the hyperparameters, __exit__ restore the adjusted hyperparameters to their default values
    """

    def __init__(self, target: Union[JarvisTarget, MPTarget, None]):
        if target in [*JarvisTarget, *MPTarget, None]:
            self.target = target
        else:
            raise ValueError(f"target needs to be an enum value of JarvisTarget or MPTarget, or None. target:{target}")

    def __enter__(self):
        if self.target in (JarvisTarget.Bandgap_MBJ, MPTarget.BulkModuli, MPTarget.ShearModuli):
            model_config.num_epoch = 100

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.target in (JarvisTarget.Bandgap_MBJ, MPTarget.BulkModuli, MPTarget.ShearModuli):
            model_config.num_epoch = 500


def train(model, logger, train_loader, valid_loader, mean, std, model_saver: TopKModelSaver):
    begin_epoch = model_config.begin_epoch
    num_epoch = model_config.num_epoch

    train_dataset_len = len(train_loader)
    valid_dataset_len = len(valid_loader)

    loss = nn.L1Loss().cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=model_config.lr, weight_decay=model_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=model_config.max_lr, epochs=num_epoch,
                                                    steps_per_epoch=len(train_loader), pct_start=0.3)

    if begin_epoch != 0:
        logger.info(f'Continue training from epoch:{begin_epoch}')
        para_file_name = f'{model_config.model_name}_{data_config.getTargetName()}_epoch{begin_epoch}_para.pth'
        model.load_state_dict(torch.load(osp.join('parameter', para_file_name)))
        checkpoint = torch.load(osp.join('parameter', 'checkpoint.pt'))
        model_saver.load_state_dict(checkpoint['model_saver'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("Training...")

    loss_train_trace = []
    loss_valid_trace = []
    tensorboard_path = './logs/tensorboard'
    if not osp.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    start_time = time.time()
    for epoch in range(begin_epoch, num_epoch):
        train_mae = 0
        loss_sum = 0
        model.train()
        with tqdm(train_loader, leave=False, desc=f'epoch{epoch}:train') as pbar:
            for batch in pbar:
                batch = batch.cuda()
                outputs = model(batch)
                result_loss = loss(outputs, batch.y)
                loss_sum += result_loss.detach()
                train_mae += torch.absolute(outputs - batch.y).mean()
                optimizer.zero_grad()
                result_loss.backward()
                optimizer.step()
                scheduler.step()

        train_mae = (train_mae.item() / train_dataset_len) * std
        loss_train_trace.append(loss_sum.item() / len(train_loader))

        valid_mae = 0
        loss_sum = 0
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader, leave=False, desc=f'epoch{epoch}:valid') as pbar:
                for batch in pbar:
                    batch = batch.cuda()
                    outputs = model(batch)
                    result_loss = loss(outputs, batch.y)
                    loss_sum += result_loss
                    valid_mae += torch.absolute(outputs - batch.y).mean()

        valid_mae = (valid_mae.item() / valid_dataset_len) * std
        loss_valid_trace.append(loss_sum.item() / len(valid_loader))
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/valid', valid_mae, epoch)
        log_str = (
            f'epoch:{epoch:0>3}; train:mae:{train_mae:.8f};'
            f'valid:mae:{valid_mae:.8f};'
        )
        logger.info(log_str)
        model_saver.step(model, valid_mae, epoch)
        if model_saver.stop and num_epoch - epoch < 50:
            logger.info("Early stopping!")
            break

        try:
            if (epoch + 1) % 50 == 0:
                if not osp.exists('parameter'):
                    os.mkdir('parameter')
                para_file_name = f'{model_config.model_name}_{data_config.getTargetName()}_epoch{epoch + 1}_para.pth'
                torch.save(model.state_dict(), osp.join('parameter', para_file_name))
                checkpoint = {
                    'model_saver': model_saver.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(checkpoint, osp.join('parameter', 'checkpoint.pt'))

        except Exception:
            logger.error('Checkpoint save failed!')
        # end each epoch

    writer.close()
    end_time = time.time()
    logger.info(f'Train time:{(end_time - start_time):.3f}s')
    logger.info('\n' + model_saver.info())
    return model_saver


def test(model, logger, test_loader, mean, std, model_saver: Optional[TopKModelSaver] = None):
    test_mae_list = []
    model.eval()
    para_file_name = f'{model_config.model_name}_{data_config.getTargetName()}_para.pth'
    para_file_path = osp.join('parameter', para_file_name)
    logger.info("Testing...")
    with torch.no_grad():
        if model_saver:
            model_paras = model_saver.top_k
        else:
            logger.info("Test with a pre-trained model")
            model_paras = []
            if osp.exists(para_file_path):
                pretrain_para = torch.load(para_file_path)
                model_paras.append(pretrain_para)
            else:
                raise RuntimeError(
                    f"Tested with a pre-trained model, but no relevant files were found {para_file_path}")

        for model_para in model_paras:
            targets = []
            predictions = []
            model.load_state_dict(model_para.model_para)
            start_time = time.time()
            with tqdm(test_loader, leave=False, desc=f'test') as pbar:
                for batch in pbar:
                    batch = batch.cuda()
                    outputs = model(batch)
                    origin_target = batch.y.item() * std + mean
                    origin_pred = outputs.item() * std + mean
                    targets.append(origin_target)
                    predictions.append(origin_pred)
            end_time = time.time()
            test_mae = np.abs(np.array(targets) - np.array(predictions)).mean()
            test_mae_list.append(test_mae)
            log_str = (
                f"Test time:{(end_time - start_time):.3f}s, "
                + str(model_para) if model_saver else "Pre-trained models"
                                                      + f', Test MAE:{test_mae:.8f};'
            )
            logger.info(log_str)

    logger.info('\n' + data_config.info())
    logger.info('\n' + model_config.info())
    if model_saver:
        logger.info(model_config.message)
        best_mae_index, best_mae_value = min(enumerate(test_mae_list), key=lambda x: x[1])
        logger.info(f"The best Test MAE:{best_mae_value}\n")
        try:
            if not (osp.exists('parameter')):
                os.mkdir('parameter')
            torch.save(model_saver.top_k[best_mae_index], para_file_path)
        except Exception:
            logger.error('The best model save failed!')


def train_and_test():
    random.seed(model_config.random_seed)
    np.random.seed(model_config.random_seed)
    torch.manual_seed(model_config.random_seed)
    torch.cuda.manual_seed(model_config.random_seed)
    torch.cuda.manual_seed_all(model_config.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    match model_config.model_name:
        case "SFTGNN":
            model = SFTGNN(num_layers=5, in_dim=data_config.getAtomDim())
        case _:
            raise ValueError(f'Invalid model name: {model_config.model_name}')

    model.cuda()
    logger = getLogger()
    train_loader, valid_loader, test_loader, mean, std = get_dataloader()
    model_saver = TopKModelSaver(model.state_dict(), k=5)

    train(model, logger, train_loader, valid_loader, mean, std, model_saver)
    test(model, logger, test_loader, mean, std, model_saver)


def test_with_pretrained_model():
    match model_config.model_name:
        case "SFTGNN":
            model = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
        case _:
            raise ValueError(f'Invalid model name: {model_config.model_name}')

    model.cuda()
    logger = getLogger()
    train_loader, valid_loader, test_loader, mean, std = get_dataloader()
    test(model, logger, test_loader, mean, std, model_saver=None)


if __name__ == '__main__':
    # If you don't want to set up training configurations using command-line parameters, you can set up custom configurations here. However, this will work for all training.
    # If you want to use different training configurations for each task, you can use TrainConfigManager
    # data_config.batch_size = 32
    # data_config.atom_features = AtomFeatureType.CGCNN
    # model_config.message = "SFTGNN Standard Model"
    # model_config.lr = 1e-3
    # model_config.max_lr = 1e-3
    # model_config.begin_epoch = 0
    # model_config.num_epoch = 500

    # You can add the tasks you want to train within targets
    # targets = [*JarvisTarget, *MPTarget]  #All tasks for training both datasets
    # targets = [*JarvisTarget] #All tasks to train the Jarvis dataset
    # targets = [*MPTarget] #All tasks to train the MP dataset
    targets = [MPTarget.BulkModuli]
    for t in targets:
        data_config.target = t
        with TrainConfigManager(t):
            train_and_test()
            # If you just want to test with a pre-trained model, you can replace it with a function like this
            # test_with_pretrained_model()
