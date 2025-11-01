import json
import math
import pickle
import time
from pathlib import Path
import random
import os.path as osp
import numpy as np
import torch

import pandas as pd
from typing import Literal

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from jarvis.db.figshare import data as jdata
from jarvis.core.specie import get_node_attributes

from pandarallel import pandarallel
from tqdm import tqdm

from data.config import *
from data.graph import atoms2graph


class _CrysAtomVec:
    """
        From CrysAtom: Distributed Representation of Atoms for Crystal Property Prediction
        https://github.com/shrimonmuke0202/CrysAtom
    """
    def __init__(self):
        with open(osp.join(str(Path(__file__).parent), 'CrysAtom_vector.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        self.all_features = torch.zeros(size=(95, 200), dtype=torch.float)
        for k, v in self.data.items():
            self.all_features[k] = v.squeeze()

    def get(self, atomic_number: int | torch.Tensor) -> torch.Tensor:
        """
        :param atomic_number:
        :return: tensor shape[95,200]
        """
        if isinstance(atomic_number, torch.Tensor):
            atomic_number = int(atomic_number)
        return self.data[atomic_number]

    def getAllFeatures(self) -> torch.Tensor:
        """
        :return: tensor shape[95,200]
        """
        return self.all_features


class GraphDataset(InMemoryDataset):
    """
        The Dataset will further process the edge features of the node features on the basis of rawdata,
        and the processed data will be cached in the ./Dataset/processed path
    """
    def __init__(self,
                 rawdata,
                 split_type: Literal['train', 'valid', 'test'],
                 root=r'Dataset',
                 mean: float = 0.0,
                 std: float = 1.0,
                 ):
        self.rawdata = rawdata
        self.root = root
        self.split_type = split_type
        self.atom_features = Config.atom_features
        self.whole_name = Config.getDatasetWholeName() + f'-{self.split_type}-{Config.atom_features}'
        self.normalize = Config.normalize
        self.mean = mean
        self.std = std
        with open(osp.join(str(Path(__file__).parent), 'elements.json'), 'rt') as f:
            self.element_dict = json.load(f)  # key: str of atomic number,value: str of symbol

        super().__init__(self.root)
        del self.rawdata
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> str:
        return self.whole_name + '.pt'

    def process(self) -> None:
        match self.atom_features:
            case AtomFeatureType.AtomicNumber:
                with tqdm(self.rawdata, desc=f'processing {self.split_type} data') as t:
                    for data in t:
                        # To adapt to the embedding in the model, use this as the index and adjust the range to [0,93]
                        data.x = data.x - 1
                        if len(data.x.shape) == 1:
                            data.x = data.x.reshape(-1, 1)
                        data.x = data.x.reshape(-1)

            case AtomFeatureType.CGCNN | AtomFeatureType.CrysAtomVec:
                if self.atom_features == AtomFeatureType.CGCNN:
                    features = np.zeros(shape=(1 + 94, 92))
                    for z, symbol in self.element_dict.items():
                        features[int(z)] = get_node_attributes(symbol, self.atom_features.value)
                    features = torch.tensor(features, dtype=torch.float)
                elif self.atom_features == AtomFeatureType.CrysAtomVec:
                    features = _CrysAtomVec().getAllFeatures()

                with tqdm(self.rawdata, desc=f'processing {self.split_type} data') as t:
                    for data in t:
                        data.x = features[data.x]
                        if len(data.x.shape) == 1:  # If there is only one atom in the crystal unit cell
                            data.x = data.x.unsqueeze(0)
                        if self.normalize:
                            data.y = (data.y - self.mean) / self.std

            case _:
                raise ValueError(f'Unknown atom_features: {self.atom_features}')

        data, slices = self.collate(self.rawdata)
        torch.save((data, slices), self.processed_paths[0])


def get_dataloader():
    """
        Generate a dataset and dataloader for a single task based on data.config.config,
        and save the cache file in the ./dataset path during the generation process,
        which can be deleted if it is no longer trained or tested
    :return: train_loader, valid_loader, test_loader, mean, std
    """
    if Config.target in JarvisTarget:
        Config.dataset_name = CrystalDataset.Jarvis
    else:
        Config.dataset_name = CrystalDataset.MP

    dataset_target_pair = {CrystalDataset.Jarvis: JarvisTarget, CrystalDataset.MP: MPTarget}
    if Config.target not in dataset_target_pair[Config.dataset_name]:
        raise ValueError(f'The dataset:{Config.dataset_name} does not match the target:{Config.target}!')
    del dataset_target_pair[Config.dataset_name]

    if Config.target in [MPTarget.BulkModuli, MPTarget.ShearModuli]:
        if not osp.isfile(Config.raw_data_cache_path()):
            filename_dict = {MPTarget.BulkModuli: 'bulk_megnet', MPTarget.ShearModuli: 'shear_megnet'}
            with open(osp.join('data', 'MPElasticModulus', filename_dict[Config.target] + '_train.pkl'), 'rb') as f:
                data_train = pickle.load(f)
            with open(osp.join('data', 'MPElasticModulus', filename_dict[Config.target] + '_val.pkl'), 'rb') as f:
                data_valid = pickle.load(f)
            with open(osp.join('data', 'MPElasticModulus', filename_dict[Config.target] + '_test.pkl'), 'rb') as f:
                data_test = pickle.load(f)
            pandarallel.initialize(progress_bar=True, verbose=0)
            tqdm.pandas()
            torch.set_printoptions(precision=10)
            all_data = []
            for original_data in [data_train, data_valid, data_test]:
                graph_data = [data['atoms'] for data in original_data]
                graph_data = pd.DataFrame(graph_data).parallel_apply(atoms2graph, axis=1)
                # graph_data = pd.DataFrame(graph_data).apply(atoms2graph, axis=1)
                graph_data = graph_data.values.tolist()
                if Config.target == MPTarget.BulkModuli:
                    for idx, data in enumerate(graph_data):
                        data.y = original_data[idx][Config.target.value].float()
                elif Config.target == MPTarget.ShearModuli:
                    for idx, data in enumerate(graph_data):
                        data.y = torch.tensor(original_data[idx][Config.target.value], dtype=torch.float)
                all_data.append(graph_data)
            torch.save(all_data, Config.raw_data_cache_path())
            data_train, data_valid, data_test = all_data
        else:
            data_train, data_valid, data_test = torch.load(Config.raw_data_cache_path())

    else:
        if not osp.isfile(Config.raw_data_cache_path()):
            data = jdata(Config.dataset_name.value)
            clean_data = []
            targets = []
            data_id = []
            id_tag = 'jid' if Config.dataset_name == CrystalDataset.Jarvis else 'id'
            for i in data:
                if isinstance(i[Config.target.value], list):
                    clean_data.append(i['atoms'])
                    targets.append(torch.tensor(i[Config.target.value]))
                    data_id.append(i[id_tag])

                elif (
                        i[Config.target.value] is not None
                        and i[Config.target.value] != "na"
                        and not math.isnan(i[Config.target.value])
                ):
                    clean_data.append(i['atoms'])
                    targets.append(i[Config.target.value])
                    data_id.append(i[id_tag])

            # clean_data = clean_data[:1000]
            pandarallel.initialize(progress_bar=True, verbose=0)
            tqdm.pandas()
            torch.set_printoptions(precision=10)
            start_time = time.time()
            clean_data = pd.DataFrame(clean_data).parallel_apply(atoms2graph, axis=1)
            #clean_data = pd.DataFrame(clean_data).apply(atoms2graph, axis=1)
            end_time = time.time()
            print(f'\nTime consumption for constructing graph data: {end_time - start_time}s')

            clean_data = clean_data.values.tolist()
            for idx, data in enumerate(clean_data):
                data.y = torch.tensor(targets[idx], dtype=torch.float)

            torch.save(clean_data, Config.raw_data_cache_path())
        else:
            clean_data = torch.load(Config.raw_data_cache_path())

        random.seed(Config.split_seed)
        random.shuffle(clean_data)
        num_train = 0
        num_valid = 0
        if Config.dataset_name == CrystalDataset.Jarvis:
            total_len = len(clean_data)
            num_train, num_valid = int(Config.train_ratio * total_len), int(Config.valid_ratio * total_len)
        elif Config.dataset_name == CrystalDataset.MP:
            num_train, num_valid = Config.num_train, Config.num_valid
        else:
            raise ValueError(f'Unknown dataset_name: {Config.dataset_name}')

        data_train = clean_data[:num_train]
        data_valid = clean_data[num_train:num_train + num_valid]
        data_test = clean_data[num_train + num_valid:]

    train_max = max(data_train, key=lambda d: float(d.y)).y.item()
    train_min = min(data_train, key=lambda d: float(d.y)).y.item()
    valid_max = max(data_valid, key=lambda d: float(d.y)).y.item()
    valid_min = min(data_valid, key=lambda d: float(d.y)).y.item()
    test_max = max(data_test, key=lambda d: float(d.y)).y.item()
    test_min = min(data_test, key=lambda d: float(d.y)).y.item()
    print(f'train range:({train_min},{train_max})')
    print(f'valid range:({valid_min},{valid_max})')
    print(f'test range:({test_min},{test_max})')

    if Config.normalize:
        train_targets = torch.tensor([data.y for data in data_train])
        mean = torch.mean(train_targets).item()
        std = torch.std(train_targets).item()
        print(f'mean:{mean}')
        print(f'std:{std}')
    else:
        mean = 0.0
        std = 1.0
    train_dataset = GraphDataset(data_train, 'train', mean=mean, std=std)
    valid_dataset = GraphDataset(data_valid, 'valid', mean=mean, std=std)
    test_dataset = GraphDataset(data_test, 'test', mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers,
                              pin_memory=Config.pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False,
                              num_workers=Config.num_workers, pin_memory=Config.pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=Config.num_workers,
                             pin_memory=Config.pin_memory, drop_last=False)

    return train_loader, valid_loader, test_loader, mean, std
