from argparse import ArgumentParser
import torch
from models.trainer import *
import numpy as np
import random

torch.backends.cudnn.deterministic = True

print('Using CUDA:', torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    # model.train_models(args.pretrain)
    model.train_models_dp(args.pretrain)


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test', dataset=args.dataset)
    model = CDEvaluator(args=args, dataloader=dataloader)

    # model.eval_models()
    model.eval_models_dp()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--seed', default=None, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='CSST_Siam_T', type=str,
                        help='CSST_Siam_T | CSST_Siam_R | '
                             'CSST_Siam_PR etc. ')
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--pretrain', default=None, type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    # parser.add_argument('--lr_decay_iters', default=10, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print('using GPU:', args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
