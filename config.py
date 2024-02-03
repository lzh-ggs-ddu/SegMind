
import logging
import platform
import argparse
from pathlib import Path

from utils.general import all_label_name_dict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

dataset_name_dict={1: 'LoveDA',2:'DeepGlobe_LandCover',3:'postdam',4:'japan_airport'}
dateset_short_name_dict={1: 'lda',2:'dland',3:'po',4:'ja'}

def parse_opt():
    parser = argparse.ArgumentParser(description='parameters of draft project')
    parser.add_argument('--device',default="cuda:0")

    parser.add_argument('--dataset_i',default=1, type=int)
    parser.add_argument('--label_rate',default=0.2, type=float)
    parser.add_argument('--net_i',default=1)
    parser.add_argument('--extra_word_half_half',default=1)

    parser.add_argument('--epoch_start',default=0, type=int)
    parser.add_argument('--epoch_end',default=100, type=int)
    parser.add_argument('--epoch_pre', default=50, type=int)
    parser.add_argument('--epoch_gap',default=50, type=int)
    parser.add_argument('--epoch_test',default='best')
    parser.add_argument('--batch_size',default=2)

    parser.add_argument('--lambda_l', default=1.0, type=float)
    parser.add_argument('--lambda_e', default=1.0, type=float)
    parser.add_argument('--lambda_r', default=1.0, type=float)
    parser.add_argument('--lambda_rsc', default=1.0, type=float)
    parser.add_argument('--lambda_c', default=1.0, type=float)

    parser.add_argument('--learn_rate',default=2e-4, type=float)
    parser.add_argument('--weight_delay',default=1e-6, type=float)
    parser.add_argument('--num_workers',default=2, type=int)
    parser.add_argument('--resize_length',default=512, type=int)
    parser.add_argument('--feature_per_class',default=1000, type=int)

    parser.add_argument('--query_threshold',default=0.97, type=float)
    parser.add_argument('--pseudo_threshold',default=0.7, type=float)
    parser.add_argument('--temperature',default=0.5, type=float)
    parser.add_argument('--bank_size',default=10000, type=int)
    parser.add_argument('--num_query',default=256, type=int)
    parser.add_argument('--num_negative',default=512, type=int)

    parser.add_argument('--alpha_ema',default=0.9, type=float)
    parser.add_argument('--mask_rate_start',default=0.25, type=float)
    parser.add_argument('--mask_rate_end',default=0.25, type=float)
    parser.add_argument('--mask_gap',default=4, type=int)

    parser.add_argument('--drop_opt',default=0.0, type=float)

    opt=parser.parse_args()

    opt.root=ROOT
    opt.dataset_root='D:/dataset'
    opt.dataset_name=dataset_name_dict[opt.dataset_i]
    opt.short_name=dateset_short_name_dict[opt.dataset_i]
    opt.extra_word_half=str(opt.net_i)+str(opt.extra_word_half_half)
    opt.extra_word=opt.short_name+str(opt.extra_word_half)
    opt.subset_from_dict={'train': 'train', 'validate': 'validate', 'test': 'test'}
    opt.ignore_class_num={1:1, 2:1, 3:0}[opt.dataset_i]
    opt.ignore_index=-1
    opt.class_num=len(all_label_name_dict[opt.dataset_i])-opt.ignore_class_num
    opt.dataset_txt_root=opt.root / 'datasets'
    opt.save_result_root=opt.root / 'results'

    logger = logging.getLogger(opt.extra_word)
    if platform.system() == 'Windows':
        for fn in logger.info, logger.warning:
            setattr(logger, fn.__name__, lambda x: fn(x.encode().decode('ascii', 'ignore')))
    opt.logger=logger

    opt.path_test_model=None

    return opt






