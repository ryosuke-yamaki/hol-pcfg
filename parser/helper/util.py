import time
import os
import logging
from distutils.dir_util import copy_tree

from parser.model import NeuralPCFG, CompoundPCFG, TNPCFG, NeuralBLPCFG, NeuralLPCFG, FastTNPCFG, FastNBLPCFG, Simple_N_PCFG, Simple_C_PCFG, HN_PCFG

import torch


def get_model(args, dataset):
    if args.model_name == 'NPCFG':
        return NeuralPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'CPCFG':
        return CompoundPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'TNPCFG':
        return TNPCFG(args, dataset).to(dataset.device)


    elif args.model_name == 'NLPCFG':
        return NeuralLPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'NBLPCFG':
        return NeuralBLPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'FastTNPCFG':
        return FastTNPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'FastNBLPCFG':
        return FastNBLPCFG(args, dataset).to(dataset.device)
    
    elif args.model_name == "SNPCFG":
        return Simple_N_PCFG(args, dataset).to(dataset.device)
    
    elif args.model_name == "SCPCFG":
        return Simple_C_PCFG(args, dataset).to(dataset.device)

    elif args.model_name == "HNPCFG":
        return HN_PCFG(args, dataset).to(dataset.device)

    else:
        raise KeyError


def get_optimizer(args, model):
    # Separate param groups for special parameters
    relation_wd = getattr(args, 'relation_weight_decay', 0.0)
    lr_scale_c = getattr(args, 'lr_scale_c', None)

    special_ids = set()
    param_groups = []

    # scale_c with separate lr
    if lr_scale_c is not None and hasattr(model, 'scale_c'):
        param_groups.append({'params': [model.scale_c], 'lr': lr_scale_c})
        special_ids.add(id(model.scale_c))

    # tau parameters with optional separate lr (multi-head, multi_tau, or single)
    lr_tau = getattr(args, 'lr_tau', None)
    if lr_tau is not None:
        tau_params = []
        for name in ('log_tau', 'log_tau_root', 'log_tau_term', 'log_tau_rule'):
            if hasattr(model, name):
                tau_params.append(getattr(model, name))
        if tau_params:
            special_ids.update(id(p) for p in tau_params)
            param_groups.append({'params': tau_params, 'lr': lr_tau})

    # relation vectors with weight decay
    if relation_wd > 0 and hasattr(model, 'v_left'):
        relation_params = [model.v_left, model.v_right]
        if hasattr(model, 'v_term'):
            relation_params.append(model.v_term)
        special_ids.update(id(p) for p in relation_params)
        param_groups.append({'params': relation_params, 'weight_decay': relation_wd})

    # everything else
    other_params = [p for p in model.parameters() if id(p) not in special_ids]
    param_groups.insert(0, {'params': other_params})

    if args.name == 'adam':
        return torch.optim.Adam(params=param_groups, lr=args.lr, betas=(args.mu, args.nu))
    elif args.name == 'adamw':
        return torch.optim.AdamW(params=param_groups, lr=args.lr, betas=(args.mu, args.nu), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

def get_logger(args, log_name='train',path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):
    model_name = args.model.model_name
    suffix = "/{}".format(model_name) + time.strftime("%Y-%m-%d-%H_%M_%S",
                                                                             time.localtime(time.time()))
    from pathlib import Path
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")
    return  saved_name

