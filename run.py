import argparse
from data.config import JarvisTarget, MPTarget, Config as data_config
from model.config import Config as model_config
from train_and_test import train_and_test, test_with_pretrained_model, ModelPara, TrainConfigManager


def getArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['train_and_test', 'test'],
                        default='train_and_test')
    parser.add_argument("--target",
                        type=str,
                        required=True,
                        choices=('All', 'Jarvis', 'MP', *['Jarvis-' + t.name for t in JarvisTarget],
                                 *['MP-' + t.name for t in MPTarget]),
                        help="train target"
                        )
    parser.add_argument("--begin_epoch", type=int, required=False, default=None,
                        help="The number of starting epoches when resuming training at breakpoints")
    parser.add_argument("--num_epoch", type=int, required=False, default=None,
                        help="The total number of epochs trained, if it is not set, the default scheme is used, and 500 or 100 is selected according to the dataset size of the task")
    parser.add_argument("--batch_size", type=int, required=False, default=None, help="The batch size for training")
    parser.add_argument("--lr", type=float, required=False, default=None, help="The parameter lr of AdamW optimizer")
    parser.add_argument("--max_lr", type=float, required=False, default=None,
                        help="The parameter max_lr of OneCircle learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, required=False, default=None,
                        help="The parameter weight_decay of the AdamW optimizer")
    return parser


def parseArgs(args):
    targets = []
    match args.target:
        case 'All':
            targets = [*JarvisTarget, *MPTarget]
        case 'Jarvis':
            targets = [*JarvisTarget]
        case 'MP':
            targets = [*MPTarget]
        case _:
            # targets = [eval(str(args.target).replace('-','Target.'))]
            dataset_name, target_name = str(args.target).split('-')
            if dataset_name == 'Jarvis':
                targets.append(JarvisTarget[target_name])
            elif dataset_name == 'MP':
                targets.append(MPTarget[target_name])
            else:
                raise ValueError('unknown dataset')
    if args.begin_epoch:
        if len(targets) == 1:
            model_config.begin_epoch = args.begin_epoch
        else:
            raise ValueError(
                f'When using breakpoint resumption, only one task is allowed to be trained, targets:{targets}')
    if args.num_epoch:
        model_config.num_epoch = args.num_epoch
    if args.batch_size:
        data_config.batch_size = args.batch_size
    if args.lr:
        model_config.lr = args.lr
    if args.max_lr:
        model_config.max_lr = args.max_lr
    if args.weight_decay:
        model_config.weight_decay = args.weight_decay

    for t in targets:
        data_config.target = t
        # When num_epoch parameter is manually passed in on the command line, None is passed here,
        # invalidating the TrainConfigManager, i.e. all tasks use the same num_epoch
        with TrainConfigManager(t if not args.num_epoch else None):
            match args.task:
                case 'train_and_test':
                    train_and_test()
                case 'test':
                    test_with_pretrained_model()
                case _:
                    raise ValueError('unknown task')


if __name__ == '__main__':
    parser = getArgParser()
    args = parser.parse_args()
    parseArgs(args)
