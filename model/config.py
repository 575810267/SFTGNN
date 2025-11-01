from typing import Literal

class Config:
    message: str = 'SFTGNN Standard Model'
    model_name: Literal['SFTGNN'] = 'SFTGNN'
    device: Literal['cuda'] = 'cuda'
    random_seed: int = 123
    node_feature: int = 128
    num_layers: int = 5
    dist_dim: int = 64
    sh_dim: int = 64
    use_sh: bool = True
    use_dist: bool = True

    lr: float = 1e-3
    max_lr: float = 1e-3
    weight_decay: float = 1e-4

    begin_epoch = 0
    num_epoch = 5

    @staticmethod
    def info():
        attrs = ['model_name', 'node_feature', 'num_layers', 'dist_dim', 'sh_dim', 'use_sh',
                 'lr', 'max_lr', 'weight_decay']
        info = 'model_config:\n{\n'
        for attr in attrs:
            if isinstance(getattr(Config, attr), float):
                info += ' '*4 + f'{attr}={getattr(Config, attr):.1e}\n'
            else:
                info += ' '*4 + f'{attr}={getattr(Config, attr)}\n'
        info+='}'
        return info
