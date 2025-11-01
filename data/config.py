import enum
import os
import os.path as osp
from typing import Union


def modify_enum(cls):
    """
        Decorate the StrEnum so that it defaults to name instead of value
    """
    cls.__str__ = lambda self: self.name  # Dynamically replace __str__ methods
    return cls


@modify_enum
class CrystalDataset(enum.StrEnum):
    Jarvis = 'dft_3d'
    MP = 'megnet'


@modify_enum
class JarvisTarget(enum.StrEnum):
    FormationEnergy = 'formation_energy_peratom'
    Bandgap_OPT = 'optb88vdw_bandgap'
    Bandgap_MBJ = 'mbj_bandgap'
    TotalEnergy = 'optb88vdw_total_energy'
    Ehull = 'ehull'


@modify_enum
class MPTarget(enum.StrEnum):
    """
        For bulk and shear datasets, these are derived from Matformer.
        https://github.com/YKQ98/Matformer/
        https://figshare.com/projects/Bulk_and_shear_datasets/165430
    """
    FormationEnergy = 'e_form'
    Bandgap = 'gap pbe'
    BulkModuli = 'bulk modulus'
    ShearModuli = 'shear modulus'


@modify_enum
class AtomFeatureType(enum.StrEnum):
    AtomicNumber = 'atomic_number'
    CGCNN = 'cgcnn'
    CrysAtomVec = 'crys_atom'


class Config:
    dataset_name: CrystalDataset = CrystalDataset.Jarvis
    target: Union[JarvisTarget, MPTarget] = JarvisTarget.FormationEnergy
    atom_features: AtomFeatureType = AtomFeatureType.CGCNN
    num_train: int = 60000
    num_valid: int = 5000
    num_test: int = None
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 123
    normalize: bool = True

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    cutoff: float = 4.0
    max_neighbors: int = 16
    use_sh: bool = True
    spherical_harmonics_l: int = 4
    use_crystalNN: bool = False

    @staticmethod
    def info() -> str:
        """
        :return: 'dataset_name', 'target', 'atom_features','split_seed', 'batch_size', 'normalize', 'use_group',
                 'pin_memory', 'cutoff','max_neighbors', 'usd_crystalNN','spherical_harmonics_l'
        """
        attrs = ['dataset_name', 'target', 'atom_features', 'split_seed', 'batch_size', 'normalize',
                 'pin_memory', 'cutoff', 'max_neighbors', 'use_sh', 'use_crystalNN', 'spherical_harmonics_l']
        info = 'data_config:\n{\n'
        for attr in attrs:
            info += ' '*4 + f'{attr}={getattr(Config, attr)}\n'
        info += '}'
        return info

    @staticmethod
    def getAtomDim() -> int:
        match Config.atom_features:
            case AtomFeatureType.AtomicNumber:
                return 1
            case AtomFeatureType.CGCNN:
                return 92
            case AtomFeatureType.CrysAtomVec:
                return 200
            case _:
                raise ValueError('Unknown AtomFeatureType')

    @staticmethod
    def getTargetName() -> str:
        """
            format: dataset_name-target
        :return:
        """
        targetName = f'{Config.dataset_name.name}-{Config.target.name}'
        return targetName

    @staticmethod
    def getDatasetWholeName() -> str:
        """
            format: dataset_name-target-cutoff-maxNeibNum-spheHarmL-splitSeed
        :return:
        """
        filename = f'{Config.dataset_name.name}-{Config.target.name}-cutoff{Config.cutoff}-maxNeibNum{Config.max_neighbors}-spheHarmL{Config.spherical_harmonics_l}-splitSeed{Config.split_seed}'
        if Config.use_crystalNN:
            filename += f'-withcrystalNN'
        return filename

    @staticmethod
    def raw_data_cache_path() -> str:
        """
            format: dataset_name-target-cutoff-maxNeibNum-spheHarmL-rawData.bin
        :return:
        """
        filename = f'{Config.dataset_name.name}-{Config.target.name}-cutoff{Config.cutoff}-maxNeibNum{Config.max_neighbors}-spheHarmL{Config.spherical_harmonics_l}-rawData.bin'
        if not osp.exists(osp.join('Dataset', 'raw')):
            os.makedirs(osp.join('Dataset', 'raw'))
        return osp.join('Dataset', 'raw', filename)
