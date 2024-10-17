import os
import torch
import torch.optim as optim
import itertools
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import time
import argparse

from yolo2 import load_data
from yolo2 import utils
from utils import *
from cfg_img800 import get_cfgs
from tps_grid_gen import TPSGridGen
from load_models import load_models
from generator_dim import GAN_dis

import torchvision.transforms as T
import torchvision

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

from PIL import Image
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmcv.ops import batched_nms
from yolo2 import utils

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid

from scipy.interpolate import interp1d
import fnmatch
import math
import sys
from operator import itemgetter
import gc
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from median_pool import MedianPool2d
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core.visualization import imshow_det_bboxes

import shutil

from physical_testloader2 import PhysicalLoader

from models.common import DetectMultiBackend

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov2', help='target net name')
parser.add_argument('--method', default='TCEGA', help='method name')
parser.add_argument('--suffix', default=None, help='suffix name')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--epoch', type=int, default=None, help='')
parser.add_argument('--z_epoch', type=int, default=None, help='')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--config', help='train config file path')
parser.add_argument('--checkpoint', help='checkpoint file')
parser.add_argument('--config2', help='train config file path')
parser.add_argument('--checkpoint2', help='checkpoint file')
parser.add_argument('--config3', help='train config file path')
parser.add_argument('--checkpoint3', help='checkpoint file')
parser.add_argument('--work-dir', help='the dir to save logs and models')
parser.add_argument(
    '--resume-from', help='the checkpoint file to resume from')
parser.add_argument(
    '--auto-resume',
    action='store_true',
    help='resume from the latest checkpoint automatically')
parser.add_argument(
    '--no-validate',
    action='store_true',
    help='whether not to evaluate the checkpoint during training')
group_gpus = parser.add_mutually_exclusive_group()
group_gpus.add_argument(
    '--gpus',
    type=int,
    help='(Deprecated, please use --gpu-id) number of gpus to use '
    '(only applicable to non-distributed training)')
group_gpus.add_argument(
    '--gpu-ids',
    type=int,
    nargs='+',
    help='(Deprecated, please use --gpu-id) ids of gpus to use '
    '(only applicable to non-distributed training)')
group_gpus.add_argument(
    '--gpu-id',
    type=int,
    default=0,
    help='id of gpu to use '
    '(only applicable to non-distributed training)')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument(
    '--diff-seed',
    action='store_true',
    help='Whether or not set different seeds for different ranks')
parser.add_argument(
    '--deterministic',
    action='store_true',
    help='whether to set deterministic options for CUDNN backend.')

parser.add_argument(
    '--options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file (deprecate), '
    'change to --cfg-options instead.')
parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. If the value to '
    'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    'Note that the quotation marks are necessary and that no white space '
    'is allowed.')

parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument(
    '--auto-scale-lr',
    action='store_true',
    help='enable automatically scaling LR.')


parser.add_argument('--prepare_data', default=False, action='store_true', help='')

parser.add_argument('--batch_size', type=int, default=7, help='')

args = parser.parse_args()

seed_value=0
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if args.options and args.cfg_options:
    raise ValueError(
        '--options and --cfg-options cannot be both '
        'specified, --options is deprecated in favor of --cfg-options')
if args.options:
    warnings.warn('--options is deprecated in favor of --cfg-options')
    args.cfg_options = args.options


pargs, kwargs = get_cfgs(args.net, args.method)
device = torch.device(args.device)



batch_size=args.batch_size
image_size=800
num_anchors=9
class_num=80
nms_thresh = 0.4
conf_thresh = 0.5
iou_thresh=0.5

loss_eval_epoch=20
ap_eval_epoch=100

from mmdet.core.bbox.iou_calculators import BboxOverlaps2D
iou_calc=BboxOverlaps2D


class MaxProbExtractorYoloV5(nn.Module):

    def __init__(self):
        super(MaxProbExtractorYoloV5, self).__init__()

    def forward(self, output: torch.Tensor):
        assert output.size(-1) == (5 + 80)

        class_confs = output[:, :, 5 : 5 + 80]
        objectness_score = output[:, :, 4]
        class_confs = torch.nn.Softmax(dim=2)(class_confs)
        max_class = torch.argmax(class_confs,dim=2)
        is_person = max_class==0
        is_person = is_person*1
        class_confs = class_confs[:, :, 0]

        max_conf, _ = torch.max(class_confs, dim=1)
        max_obj, _ = torch.max(objectness_score, dim=1)
        return max_conf, max_obj, class_confs, objectness_score, is_person
img_dir_train = './data/INRIAPerson/Train/pos'
lab_dir_train = './data/train_labels'
if args.net=='yolov2':
    train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=10)

    model = load_models(**kwargs)
    model = model.eval().to(device)
else:
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    rank, _ = get_dist_info()
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')



    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    model1type=0
    cfg2 = Config.fromfile(args.config2)
    cfg2 = replace_cfg_vals(cfg2)

    if args.cfg_options is not None:
        cfg2.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg2)
    if cfg2.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg2.model:
        cfg2.model.pretrained = None
    elif 'init_cfg' in cfg2.model.backbone:
        cfg2.model.backbone.init_cfg = None

    if cfg2.model.get('neck'):
        if isinstance(cfg2.model.neck, list):
            for neck_cfg in cfg2.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg2.model.neck.get('rfp_backbone'):
            if cfg2.model.neck.rfp_backbone.get('pretrained'):
                cfg2.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg2.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg2.gpu_ids = [args.gpu_id]
    cfg2.device = get_device()
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg2.dist_params)


    rank, _ = get_dist_info()
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')
    model2 = build_detector(cfg2.model, test_cfg=cfg2.get('test_cfg'))
    fp16_cfg = cfg2.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model2)
    checkpoint2 = load_checkpoint(model2, args.checkpoint2, map_location='cpu')
    if 'CLASSES' in checkpoint2.get('meta', {}):
        model2.CLASSES = checkpoint2['meta']['CLASSES']
    else:
        model2.CLASSES = dataset.CLASSES

    model2 = build_dp(model2, cfg2.device, device_ids=cfg2.gpu_ids)
    cfg3 = Config.fromfile(args.config3)
    cfg3 = replace_cfg_vals(cfg3)

    if args.cfg_options is not None:
        cfg3.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg3)
    if cfg3.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg3.model:
        cfg3.model.pretrained = None
    elif 'init_cfg' in cfg3.model.backbone:
        cfg3.model.backbone.init_cfg = None

    if cfg3.model.get('neck'):
        if isinstance(cfg3.model.neck, list):
            for neck_cfg in cfg3.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg3.model.neck.get('rfp_backbone'):
            if cfg3.model.neck.rfp_backbone.get('pretrained'):
                cfg3.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg3.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg3.gpu_ids = [args.gpu_id]
    cfg3.device = get_device()
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg3.dist_params)


    rank, _ = get_dist_info()
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')



    prob_extractor_yolov5 = MaxProbExtractorYoloV5().to(device)
    model3 = DetectMultiBackend('./yolov5_params/yolov5m.pt', device=device, dnn=False, data=None, fp16=False)

    model.eval()
    model2.eval()
    model3.eval()

    for param in model3.parameters():
        param.requires_grad = False

    model_list=[model,model2,model3]
    model_type_list=[0,1,2]
    model_name_list=['pvt-retina','res-maskrcnn','YOLO-V5']
    conf_thresh_list=[0.3,0.3,0.1]
    nms_thresh_list=[0.7,0.7,0.5]
    iou_thresh_list=[0.5,0.5,0.5]
    model_imgsize_list=[800,416,640]
    optimize_index=2


    dataset = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=dataset[0].CLASSES)
    if args.net=='pvt_tiny_retina':
        model.CLASSES = 80

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    print('dataset:',dataset[0])
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    print('data_loaders: ',len(data_loaders),'------if greater than 1, need to change code')
    train_loader = data_loaders[0]



    dataset = [build_dataset(cfg.data.train_test)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=dataset[0].CLASSES)
    if args.net=='pvt_tiny_retina':
        model.CLASSES = 80

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    print('dataset:',dataset[0])
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    print('data_loaders: ',len(data_loaders),'------if greater than 1, need to change code')
    train_loader2 = data_loaders[0]
    valdataset = [build_dataset(cfg.data.val)]
    if args.net=='pvt_tiny_retina':
        model.CLASSES = 80

    valdataset = valdataset if isinstance(valdataset, (list, tuple)) else [valdataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    val_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        runner_type=runner_type,
        persistent_workers=False)

    val_loader_cfg = {
        **val_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    print('valdataset:',valdataset[0])
    valdata_loaders = [build_dataloader(ds, **val_loader_cfg) for ds in valdataset]
    print('valdata_loaders: ',len(valdata_loaders),'------if greater than 1, need to change code')
    val_data_loader = valdata_loaders[0]

sub_data_dirct_general='test_data_directory_here'
sub_data_dirct_general_train='2024_3_5_internet_ped'
sub_data_dirct_general_background='2023_5_11_diverse_person_background/background'


class PatchTransformerPVT(nn.Module):

    def __init__(self):
        super(PatchTransformerPVT, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.clampmin=-2.999999
        self.clampmax=2.99999
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

        ksize = 5
        half = (ksize - 1) * 0.5
        sigma = 0.3 * (half - 1) + 0.8
        x = np.arange(-half, half + 1)
        x = np.exp(- np.square(x / sigma) / 2)
        x = np.outer(x, x)
        x = x / x.sum()
        x = torch.from_numpy(x).float()
        kernel = torch.zeros(3, 3, ksize, ksize)
        for i in range(3):
            kernel[i, i] = x
        self.register_buffer('kernel', kernel)
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True, lc_scale=0.1, pooling='median', rand_sub=False, old_fasion=True,ratio=1,scale_ratio=1.15, rotate_strength=1, shift=0.07):

        ratio=ratio/1.2

        if adv_patch.dim() == 3:
            adv_patch = adv_patch.unsqueeze(0)

        B, L, _ = lab_batch.shape
        _, C, H, W = adv_patch.shape
        SBS = B * L
        if pooling is 'median':
            adv_patch = self.medianpooler(adv_patch)
        elif pooling is 'avg':
            adv_patch = F.avg_pool2d(adv_patch, 7, 3)
        elif pooling is 'gauss':
            adv_patch = F.conv2d(adv_patch, self.kernel, padding=2)
        elif pooling is not None:
            raise ValueError
        adv_patch = adv_patch.unsqueeze(1)
        adv_batch = adv_patch.expand(B, L, -1, -1, -1)
        batch_size = torch.Size((B, L))
        contrast = adv_patch.new(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = adv_patch.new(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noise = adv_patch.new(adv_batch.shape).uniform_(-1, 1) * self.noise_factor
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, self.clampmin, self.clampmax)
        msk_batch = adv_patch.new(adv_batch.shape).fill_(1).logical_and((lab_batch[:, :, 0] == 0).view(B, L, 1, 1, 1))
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = adv_patch.new(anglesize).uniform_(self.minangle, self.maxangle)
            angle = rotate_strength*angle
        else:
            angle = adv_patch.new(anglesize).fill_(0)
        target_size = torch.sqrt(((lab_batch[:, :, 3].mul(0.2)) ** 2) + ((lab_batch[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if rand_loc:
            off_x = targetoff_x * (adv_patch.new(targetoff_x.size()).uniform_(-lc_scale, lc_scale))
            target_x = target_x + off_x
            off_y = targetoff_y * (adv_patch.new(targetoff_y.size()).uniform_(-lc_scale, lc_scale))
            target_y = target_y + off_y
        cur_shift=shift+(0.03*(np.random.rand()-0.5))
        off_y = targetoff_y * (adv_patch.new(targetoff_y.size()).uniform_(-cur_shift-0.01, -cur_shift+0.01))
        target_y = target_y + off_y

        if old_fasion:
            target_y = target_y
        else:
            target_y = target_y

        scale = target_size * scale_ratio
        scale = scale.view(anglesize)

        adv_batch = adv_batch.view(SBS, C, H, W)
        msk_batch = msk_batch.view(SBS, C, H, W)

        if rand_sub is True:
            width = adv_batch.new(size=[SBS, 1]).uniform_(0.5, 1)
            height = adv_batch.new(size=[SBS, 1]).uniform_(0.8, 1)
            wst = adv_batch.new(size=[SBS, 1]).uniform_(0, 1) * (1 - width)
            hst = adv_batch.new(size=[SBS, 1]).uniform_(0, 1) * (1 - height)
            W_msk = torch.arange(W, device=adv_batch.device).expand(SBS, W) < (wst * W)
            W_msk.logical_xor_(torch.arange(W, device=adv_batch.device).expand(SBS, W) < ((wst + width) * W))
            W_msk = W_msk.view(SBS, 1, 1, W)
            H_msk = torch.arange(H, device=adv_batch.device).expand(SBS, H) < (hst * H)
            H_msk.logical_xor_(torch.arange(H, device=adv_batch.device).expand(SBS, H) < ((hst + height) * H))
            H_msk = H_msk.view(SBS, 1, H, 1)
            msk_batch = msk_batch.logical_and(W_msk.logical_and(H_msk))

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle).to(adv_patch)
        cos = torch.cos(angle).to(adv_patch)
        theta = adv_patch.new(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale*ratio
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale*ratio
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale*ratio
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale*ratio
        grid = F.affine_grid(theta, [SBS, C, img_size, img_size])

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch.to(adv_batch), grid)
        adv_batch_t = adv_batch_t.view(B, L, C, img_size, img_size)

        msk_batch_t = msk_batch_t.view(B, L, C, img_size, img_size)
        adv_batch_t = torch.clamp(adv_batch_t, self.clampmin, self.clampmax)

        return adv_batch_t * msk_batch_t


target_func = lambda obj, cls: obj
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = PatchTransformerPVT().to(device)
resize_transform = T.Resize(size = (image_size,image_size)).to(device)
resize_transform_list=[]
for i in range(len(model_imgsize_list)):
    resize_transform_list.append(T.Resize(size = (model_imgsize_list[i],model_imgsize_list[i])).to(device))
softmax=torch.nn.Softmax(1)
if kwargs['name'] == 'ensemble':
    prob_extractor_yl2 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov2').to(device)
    prob_extractor_yl3 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov3').to(device)
else:
    prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
total_variation = load_data.TotalVariation().to(device)

target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)

target_func = lambda obj, cls: obj
prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)

result_dir = './results/result_' + args.suffix

print(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

loader = train_loader
epoch_length = len(loader)
print(f'One epoch is {len(loader)}')

true_lab_dir = './data/temp_train_labels_pvt_retina'+'_2023_4_17_'+args.suffix


std_out=open(args.suffix+'_stdout'+'.txt','w+')
std_out.close()

if os.path.exists('./sample_patched')==False:
    os.makedirs('./sample_patched')

if os.path.exists('./sample_patched/'+args.suffix)==False:
    os.makedirs('./sample_patched/'+args.suffix)




def truths_length(truths):
    for i in range(50):
        if truths[i][1] == -1:
            return i


def label_filter(truths, labels=None):
    if labels is not None:
        new_truths = truths.new(truths.shape).fill_(-1)
        c = 0
        for t in truths:
            if t[0].item() in labels:
                new_truths[c] = t
                c = c + 1
        return new_truths

def whxy2xminyminxmaxymax(whxy):
    w=whxy[2]
    h=whxy[3]
    xm=whxy[0]-w/2
    ym=whxy[1]-h/2
    xM=whxy[0]+w/2
    yM=whxy[1]+h/2
    return [xm,ym,xM,yM]

def whxy2xminyminxmaxymax_batch(whxy):
    w=whxy[:,2:3]
    h=whxy[:,3:4]
    xm=whxy[:,0:1]-w/2
    ym=whxy[:,1:2]-h/2
    xM=whxy[:,0:1]+w/2
    yM=whxy[:,1:2]+h/2
    if torch.is_tensor(xm)==True:
        return np.concatenate([xm.detach().cpu().numpy(),ym.detach().cpu().numpy(),xM.detach().cpu().numpy(),yM.detach().cpu().numpy()],1)
    else:
        return np.concatenate([xm,ym,xM,yM],1)



if args.prepare_data:
    conf_thresh = 0.5
    nms_thresh = 0.4
    img_dir = './data/test_padded'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    lab_dir = './data/test_lab_%s' % args.net
    if not os.path.exists(lab_dir):
        os.mkdir(lab_dir)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    for batch_idx, data_ori in enumerate(val_data_loader):


        data=data_ori['img'][0]

        file_label=data_ori['img_metas'][0]._data[0][0]['ori_filename']
        file_label=file_label[0:len(file_label)-4]+'.txt'
        labs=[file_label]
        data_ori['img'][0]=data
        result = model(return_loss=False, rescale=True, **data_ori)
        w=data_ori['img_metas'][0]._data[0][0]['ori_shape'][1]
        h=data_ori['img_metas'][0]._data[0][0]['ori_shape'][0]
        all_boxes=[]
        for i in range(len(result)):
            batch_list=[]
            for j in range(len(result[i])):
                for k in range(result[i][j].shape[0]):
                    current=result[i][j][k,:]
                    x_w=current[0]/w
                    y_h=current[1]/h
                    w_w=current[2]/w
                    h_h=current[3]/h
                    det_confs=current[4]
                    cls_max_confs=current[4]
                    cls_max_ids=j
                    if det_confs>conf_thresh:
                        batch_list.append([x_w,y_h,w_w,h_h,det_confs,cls_max_confs,cls_max_ids])

            batch_list=torch.tensor(np.asarray(batch_list))
            all_boxes.append(batch_list)

        for i in range(data.size(0)):
            boxes = all_boxes[i]
            boxes = utils.nms(boxes, nms_thresh)
            if 1!=len(boxes.shape):
                new_boxes = boxes[:, [6, 0, 1, 2, 3]]
                new_boxes = new_boxes[new_boxes[:, 0] == 0]
                new_boxes = new_boxes.detach().cpu().numpy()
            else:
                new_boxes=boxes
            if lab_dir is not None:
                save_dir = os.path.join(lab_dir, labs[i])
                np.savetxt(save_dir, new_boxes, fmt='%f')
                img = unloader(data[i].detach().cpu())
            if img_dir is not None:
                save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
                img.save(save_dir)
    print('preparing done')




def get_bboxes_single(model,
                        cls_score_list,
                        bbox_pred_list,
                        score_factor_list,
                        mlvl_priors,
                        img_meta,
                        cfg,
                        rescale=False,
                        with_nms=True,
                        **kwargs):
    if score_factor_list[0] is None:
        with_score_factors = False
    else:
        with_score_factors = True
    img_shape = img_meta[0]['img_shape']
    nms_pre = 1000

    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_labels = []
    if with_score_factors:
        mlvl_score_factors = []
    else:
        mlvl_score_factors = None
    for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
            enumerate(zip(cls_score_list, bbox_pred_list,
                            score_factor_list, mlvl_priors)):

        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        if with_score_factors:
            score_factor = score_factor.permute(1, 2,
                                                0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2,
                                        0).reshape(-1, 80)
        if model.module.bbox_head.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)[:, :-1]
        results = filter_scores_and_topk(
            scores, conf_thresh, nms_pre,
            dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results

        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']

        if with_score_factors:
            score_factor = score_factor[keep_idxs]

        bboxes = model.module.bbox_head.bbox_coder.decode(
            priors, bbox_pred, max_shape=img_shape)

        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_labels.append(labels)
        if with_score_factors:
            mlvl_score_factors.append(score_factor)

    return bbox_post_process(model,mlvl_scores, mlvl_labels, mlvl_bboxes,
                                    img_meta[0]['scale_factor'], cfg, rescale,
                                    with_nms, mlvl_score_factors, **kwargs)

def bbox_post_process(model,
                        mlvl_scores,
                        mlvl_labels,
                        mlvl_bboxes,
                        scale_factor,
                        cfg,
                        rescale=False,
                        with_nms=True,
                        mlvl_score_factors=None,
                        **kwargs):
    assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

    mlvl_bboxes = torch.cat(mlvl_bboxes)
    if rescale:
        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
    mlvl_scores = torch.cat(mlvl_scores)
    mlvl_labels = torch.cat(mlvl_labels)

    if mlvl_score_factors is not None:
        mlvl_score_factors = torch.cat(mlvl_score_factors)
        mlvl_scores = mlvl_scores * mlvl_score_factors
    return mlvl_bboxes, mlvl_scores, mlvl_labels
def get_det_loss_ddetr(data,model,input,pargs,p_img):
    mlvl_feats = model.module.extract_feat(p_img)

    if type(data['img_metas'])!=list:
        img_metas=data['img_metas']._data[0]
    elif type(data['img_metas'])==list:
        img_metas=data['img_metas'][0]._data[0]

    num_levels = len(mlvl_feats)
    img_metas_list = [img_metas for _ in range(num_levels)]





    batch_size = mlvl_feats[0].size(0)
    input_img_h, input_img_w = p_img.shape[2],p_img.shape[3]
    img_masks = mlvl_feats[0].new_ones(
        (batch_size, input_img_h, input_img_w))
    for img_id in range(batch_size):
        img_h, img_w, _ = img_metas[img_id]['img_shape']
        img_masks[img_id, :img_h, :img_w] = 0

    mlvl_masks = []
    mlvl_positional_encodings = []
    for feat in mlvl_feats:
        mlvl_masks.append(
            F.interpolate(img_masks[None],
                            size=feat.shape[-2:]).to(torch.bool).squeeze(0))
        mlvl_positional_encodings.append(
            model.module.bbox_head.positional_encoding(mlvl_masks[-1]))

    query_embeds = None
    if not model.module.bbox_head.as_two_stage:
        query_embeds = model.module.bbox_head.query_embedding.weight
    hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = model.module.bbox_head.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                reg_branches=model.module.bbox_head.reg_branches if model.module.bbox_head.with_box_refine else None,
                cls_branches=model.module.bbox_head.cls_branches if model.module.bbox_head.as_two_stage else None
        )
    hs = hs.permute(0, 2, 1, 3)
    outputs_classes = []
    outputs_coords = []

    for lvl in range(hs.shape[0]):
        if lvl == 0:
            reference = init_reference
        else:
            reference = inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = model.module.bbox_head.cls_branches[lvl](hs[lvl])
        tmp = model.module.bbox_head.reg_branches[lvl](hs[lvl])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)

    outputs_classes = torch.stack(outputs_classes)
    outputs_coords = torch.stack(outputs_coords)

    class_list=[]
    for i in range(outputs_classes.shape[0]):
        cur=torch.squeeze(outputs_classes[i])
        class_list.append(cur)
    all_classes=torch.cat(class_list,0)
    max_index=torch.argmax(all_classes,1)
    person_probs=softmax(all_classes[max_index==0])[:,0]

    if person_probs.numel()!=0:
        return torch.mean(person_probs),0
    else:
        return None,0

def get_det_loss_retina(label_true,data,model,input,pargs,p_img,weight=1):

    valid_num = 0
    det_loss = p_img.new_zeros([])

    cls_scores=input[0]
    bbox_preds=input[1]
    with_score_factors = False
    score_factors=None

    if type(data['img_metas'])!=list:
        img_metas=data['img_metas']._data
    elif type(data['img_metas'])==list:
        img_metas=data['img_metas'][0]._data
    cfg=None
    rescale=False
    with_nms=True

    num_levels = len(cls_scores)

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = model.module.bbox_head.prior_generator.grid_priors(
        featmap_sizes,
        dtype=cls_scores[0].dtype,
        device=cls_scores[0].device)

    result_list = []

    score_list=[]
    boxes_list=[]
    label_list=[]

    temp_sum=torch.zeros([])
    count=0
    all_boxes_t=[]
    for img_id in range(len(img_metas)):
        img_meta = img_metas[img_id]
        cls_score_list = select_single_mlvl(cls_scores, img_id,detach=False)
        bbox_pred_list = select_single_mlvl(bbox_preds, img_id,detach=False)
        if with_score_factors:
            score_factor_list = select_single_mlvl(score_factors, img_id)
        else:
            score_factor_list = [None for _ in range(num_levels)]

        mlvl_bboxes, mlvl_scores, mlvl_labels = get_bboxes_single(model,cls_score_list, bbox_pred_list,
                                                    score_factor_list, mlvl_priors,
                                                    img_meta, cfg, rescale, with_nms)
        if type(data['img_metas'])!=list:
            w=data['img_metas']._data[0][img_id]['ori_shape'][1]
            h=data['img_metas']._data[0][img_id]['ori_shape'][0]
        elif type(data['img_metas'])==list:
            w=data['img_metas'][0]._data[0][img_id]['ori_shape'][1]
            h=data['img_metas'][0]._data[0][img_id]['ori_shape'][0]

        mlvl_bboxes=mlvl_bboxes.detach().cpu().numpy()

        batch_list=[]
        for j in range(mlvl_bboxes.shape[0]):
            current=mlvl_bboxes[j]
            xs_w=current[0]/w
            ys_h=current[1]/h
            ws_w=current[2]/w
            hs_w=current[3]/h
            batch_list.append([ws_w,hs_w,xs_w,ys_h])
        batch_list=torch.tensor(np.asarray(batch_list))

        if mlvl_labels.shape[0]!=0:
            batch_list=batch_list.to(device)
            all_boxes_t.append(batch_list[mlvl_labels==0,:])

            score_list.append(mlvl_scores[mlvl_labels==0])
        else:
            all_boxes_t.append(batch_list)

            score_list.append(mlvl_scores)
    if len(label_true)==0 or len(all_boxes_t)==0:
        return det_loss, valid_num
    if len(label_true[0].shape)==0 or len(all_boxes_t[0].shape)==0:
        return det_loss, valid_num
    if label_true[0].shape[0]==0 or all_boxes_t[0].shape[0]==0:
        return det_loss, valid_num
    all_boxes_np=whxy2xminyminxmaxymax_batch(all_boxes_t[0])
    all_lab_np=whxy2xminyminxmaxymax_batch(label_true[0])
    iou_list=[]
    for i in range(len(all_boxes_np)):
        iou=utils.bbox_iou(all_boxes_np[i], all_lab_np[0],x1y1x2y2=True)
        iou_list.append(iou)
    iou_list=np.asarray(iou_list)

    if len(iou_list)!=0:
        if np.max(iou_list)>0:
            iou_non_zero_indexes=[]
            for u in range(len(iou_list)):
                if iou_list[u]>0:
                    iou_non_zero_indexes.append(u)
            score_list_overlap=score_list[0][iou_non_zero_indexes]
            det_loss=det_loss+torch.log(torch.mean(score_list_overlap))
            return det_loss*weight, valid_num
        else:
            all_boxes_xy=all_boxes_t[0][:,0:2]
            all_labels_xy=label_true[0][:,0:2]
            all_labels_xy=torch.tensor(all_labels_xy).float().to(device)
            distance=torch.norm(all_boxes_xy-all_labels_xy,dim=1).cuda()
            det_loss = det_loss+torch.mean(score_list[0]/(distance*distance))
            return det_loss*weight, valid_num
    else:
        return det_loss, valid_num


def get_det_loss_mrcnn(data,model,pargs,p_img):
    x = model.module.extract_feat(p_img)

    proposal_cfg = model.module.train_cfg.get('rpn_proposal',
                                       model.module.test_cfg.rpn)
    outs = model.module.rpn_head(x)
    if type(data['img_metas'])!=list:
        img_metas=data['img_metas']._data[0]
    else:
        img_metas=data['img_metas'][0]._data[0]
    proposal_list = model.module.rpn_head.get_bboxes(
        *outs, img_metas=img_metas, cfg=proposal_cfg)

    if type(data['gt_bboxes'])!=list:
        gt_bboxes=data['gt_bboxes']._data[0]
    else:
        gt_bboxes=data['gt_bboxes'][0]._data[0]
    for i in range(len(gt_bboxes)):
        gt_bboxes[i]=gt_bboxes[i].to(device)

    if type(data['gt_labels'])!=list:
        gt_labels=data['gt_labels']._data[0]
    else:
        gt_labels=data['gt_labels'][0]._data[0]
    for i in range(len(gt_labels)):
        gt_labels[i]=gt_labels[i].to(device)


    num_imgs = len(img_metas)
    gt_bboxes_ignore = [None for _ in range(num_imgs)]
    sampling_results = []
    for i in range(num_imgs):
        assign_result = model.module.roi_head.bbox_assigner.assign(
            proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
            gt_labels[i])
        sampling_result = model.module.roi_head.bbox_sampler.sample(
            assign_result,
            proposal_list[i],
            gt_bboxes[i],
            gt_labels[i],
            feats=[lvl_feat[i][None] for lvl_feat in x])
        sampling_results.append(sampling_result)

    rois = bbox2roi([res.bboxes for res in sampling_results])
    bbox_results = model.module.roi_head._bbox_forward(x, rois)

    max_index=torch.argmax(softmax(bbox_results['cls_score'][:,0:80]),1)
    person_prob=softmax(bbox_results['cls_score'][:,0:80])[max_index==0,:][:,0:1]
    person_conf=bbox_results['cls_score'][:,80:][max_index==0,:]
    det_loss=torch.sum(person_prob)
    valid_num=person_conf.shape[0]
    return det_loss, valid_num



def get_det_loss_yolov5(output,label_true_forloss,cur_model_index):

    person_index=torch.argmax(output[0][:,5:],dim=1)
    person_boxes=output[0][:,0:4][person_index==0,:]
    person_obj=output[0][:,4:5][person_index==0,:]
    person_conf=output[0][:,5:6][person_index==0,:]
    labels = model_imgsize_list[cur_model_index]*torch.tensor(label_true_forloss[0]).to(device)

    iou = utils.bbox_iou_mat(person_boxes, labels, x1y1x2y2=False)
    iou = torch.squeeze(iou)
    ignore_thresh=0.3
    iou = torch.clamp(iou,ignore_thresh,1.0)-ignore_thresh

    det_loss=torch.mean(torch.squeeze(person_conf)*iou*iou)
    valid_num = 0
    if det_loss==None:
        det_loss=torch.mean(adv_patch)*0

    return det_loss, valid_num




def inspect_output(img,result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
    img=np.concatenate([img[:,:,2:3],img[:,:,1:2],img[:,:,0:1]],2)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    segms = None
    if segm_result is not None and len(labels) > 0:
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    if out_file is not None:
        show = False

    class_names=['person']
    for i in range(79):
        class_names.append('other')

    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
def test(cur_model_counter,sat_minmax,val_minmax,blur_layer_list_rand,physical_loader,data_subdircts,data_base,blur_layer_list,patch_name,resolution,min_area,max_area,model, loader, adv_cloth=None, gan=None, z=None, type_=None,conf_thresh=0.5, nms_thresh=0.5, iou_thresh=0.5,num_of_samples=100,
         old_fasion=True,patchname='None', test_crop_size=[200,133],test_crop_size_min=[180,113],counter=0):
    print('---point3-----')

    RGB_EOT=0.05
    SV_EOT_atm=[[0,0],
                [0,0],
                [0,0],
                [0.1,0.1],
                [0.1,0.1],
                [0.1,0.1],
                [0.1,0.1]]
    SV_EOT_rand=[[0,0],
                [0,0],
                [0,0],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15]]

    test_rand_thresh=0

    sharpen_1x=get_sharpen_kernel('cross',3)
    sharpen_1x2x=get_sharpen_kernel('cross',3)

    vis2turb={}
    vis2turb['71']=2.7
    vis2turb['86']=3.1
    vis2turb['101']=3.1
    vis2turb['116']=3.6
    vis2turb['131']=3.6

    sky_image_folder='./sky_images/images'
    sky_colors=precalc_sky_avgs(sky_image_folder)


    network_width=300
    color_network = MappingNet(network_width).to(device)
    color_network.load_state_dict(torch.load('./patches_to_load/2023_3_1_color_mapping_network/version_2023_2_23_temp.pth'))

    model.eval()
    total = 0.0
    proposals = 0.0
    correct = 0.0
    batch_num = len(loader)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    positives = []
    print('len(loader)',len(loader),'--------------------------')


    total_labels=0
    successed=0


    img_list=physical_loader.get_all(sub_data_dirct_general,data_subdircts[counter])

    for i_batch_, data_ori in enumerate(loader):
        break

    for i_batch in range(len(img_list)):

        data_ori['img_metas'][0]._data[0][0]['ori_shape']=(img_list[i_batch][0].shape[2],img_list[i_batch][0].shape[3],3)
        data_ori['img_metas'][0]._data[0][0]['img_shape']=(img_list[i_batch][0].shape[2],img_list[i_batch][0].shape[3],3)
        data_ori['img_metas'][0]._data[0][0]['pad_shape']=(img_list[i_batch][0].shape[2],img_list[i_batch][0].shape[3],3)
        data_ori['img_metas'][0]._data[0][0]['scale_factor']=np.asarray([1,1,1,1])
        data_ori['img_metas'][0]._data[0][0]['flip']=False
        data_ori['img_metas'][0]._data[0][0]['flip_direction']=None
        if type(data_ori['img_metas'])==list:
            w=data_ori['img_metas'][0]._data[0][0]['ori_shape'][1]
            h=data_ori['img_metas'][0]._data[0][0]['ori_shape'][0]
        elif type(data_ori['img_metas'])!=list:
            w=data_ori['img_metas']._data[0][0]['ori_shape'][1]
            h=data_ori['img_metas']._data[0][0]['ori_shape'][0]
        whwh=torch.tensor([w,h,w,h])
        whwh=torch.reshape(whwh,[1,1,4])

        img_batch=img_list[i_batch][0]

        label_name=true_lab_dir+'_temp_test' + '/' + img_list[i_batch][1]+'/'+ img_list[i_batch][2] + '/' +img_list[i_batch][3]
        label_name=label_name[0:len(label_name)-4]+'.txt'

        full_true_boxes=np.loadtxt(label_name, dtype=float)
        full_label_data=np.loadtxt(label_name, dtype=float)
        if len(full_true_boxes.shape)>1:
            cur_iter=full_true_boxes.shape[0]
        elif len(full_true_boxes.shape)==1 and len(full_true_boxes)==0:
            cur_iter=0
        elif len(full_true_boxes.shape)==1 and len(full_true_boxes)!=0:
            cur_iter=1
        else:
            cur_iter=0
        for extract_i in range(cur_iter):
            if len(full_true_boxes.shape)>1:
                true_boxes = [full_true_boxes[extract_i]]
                label_data = [full_label_data[extract_i]]
            elif len(full_true_boxes.shape)==1 and len(full_true_boxes)!=0:
                true_boxes = full_true_boxes
                label_data = full_label_data
            true_boxes=np.asarray(true_boxes)
            label_data=np.asarray(label_data)

            true_boxes=torch.tensor(true_boxes)

            if len(true_boxes.shape)>1:
                true_labels=true_boxes[:,:1]
                true_boxes=true_boxes[:,1:]
                label_true=torch.cat([true_labels,true_boxes],1)
                label_true=np.expand_dims(label_true,0)
            elif len(true_boxes.shape)==1 and len(true_boxes)==0:
                continue
            else:
                label_true=np.expand_dims(np.expand_dims(true_boxes,0),0)
            if data_ori['img_metas'][0]._data[0][0]['flip']==True:
                temp_x=label_true[:,:,1:2]
                temp_x=1-temp_x
                label_true[:,:,1:2]=temp_x

            if len(label_data.shape)>1:
                label_data=np.expand_dims(label_data,0)
            elif len(label_data.shape)==1 and len(label_data)==0:
                continue
            else:
                label_data=np.expand_dims(np.expand_dims(label_data,0),0)
            data=img_batch




            if data.shape[0]==1:
                target=label_data
                target=torch.tensor(target)
            else:
                print('need to update code')

            resize_transform_back = T.Resize(size = (data.shape[2],data.shape[3])).to(cfg.device)
            data=resize_transform(data)


            cur_crop_size=[]
            for i in range(len(test_crop_size)):
                cur_cs=int(np.random.rand()*(test_crop_size[i]-test_crop_size_min[i])+test_crop_size_min[i])
                cur_crop_size.append(cur_cs)
            if type_ == 'gan':
                z = torch.randn(1, 128, *([9] * 2), device=cfg.device)
                cloth = gan.generate(z)
                adv_patch, x, y = random_crop(cloth, cur_crop_size, pos=pargs.pos, crop_type=pargs.crop_type)
            elif type_ =='z':
                z_crop, _, _ = random_crop(z, [9] * 2, pos=None, crop_type='recursive')
                cloth = gan.generate(z_crop)
                adv_patch, x, y = random_crop(cloth, cur_crop_size, pos=pargs.pos, crop_type=pargs.crop_type)
            elif type_ == 'patch':
                adv_patch=adv_cloth
            elif type_ is not None:
                raise ValueError

            if adv_patch is not None:
                label_true = torch.tensor(label_true).to(cfg.device)

                img_batch_temp=img_batch+3
                img_batch_temp=img_batch_temp/6
                img_batch_temp=rgb2hsv_torch(img_batch_temp)
                img_avg_val=torch.mean(img_batch_temp[:,2:3,:,:].clone().detach()).detach()


                adv_patch_t=adv_patch+3
                adv_patch_t=adv_patch_t/6


                adv_patch_t=rgb2hsv_torch(adv_patch_t)
                adv_patch_t_avg_val=torch.mean(adv_patch_t[:,2:3,:,:].clone().detach())
                adv_patch_t[:,2:3,:,:]=torch.clamp(adv_patch_t[:,2:3,:,:]*(img_avg_val/adv_patch_t_avg_val),0,1)
                adv_patch_t=hsv2rgb_torch(adv_patch_t)

                adv_patch_t=torch.squeeze(adv_patch_t)
                apts=adv_patch_t.shape
                adv_patch_t=torch.transpose(adv_patch_t,0,1)
                adv_patch_t=torch.transpose(adv_patch_t,1,2)
                adv_patch_t=torch.reshape(adv_patch_t,[apts[1]*apts[2],apts[0]])

                adv_patch_t=color_network(adv_patch_t)

                adv_patch_t=torch.reshape(adv_patch_t,[apts[1],apts[2],apts[0]])
                adv_patch_t=torch.transpose(adv_patch_t,1,2)
                adv_patch_t=torch.transpose(adv_patch_t,0,1)
                adv_patch_t=torch.unsqueeze(adv_patch_t,0)

                cur_filter_rand=np.random.rand()
                adv_patch_t=filter_1x(adv_patch_t,counter,SV_EOT_atm[counter],RGB_EOT)

                adv_patch_t=adv_patch_t*6
                adv_patch_t=adv_patch_t-3

                all_vis=list(sky_colors.keys())
                cur_vis=all_vis[int(len(all_vis)*np.random.rand())]
                cur_index=int(len(sky_colors[cur_vis])*np.random.rand())
                cur_turb=vis2turb[cur_vis]
                cur_turb=torch.tensor(cur_turb).to(device)
                cur_sky=sky_colors[cur_vis][cur_index][2]
                cur_sky=torch.tensor(cur_sky).to(device)
                resize_transform_patch=T.Resize(size = (466,466)).to(cfg.device)
                resize_transform_back_patch = T.Resize(size = (adv_patch_t.shape[2],adv_patch_t.shape[3])).to(cfg.device)
                adv_patch_t=resize_transform_patch(adv_patch_t)

                if cur_filter_rand<test_rand_thresh:
                    cur_filter_rand2=int(np.random.rand()*len(blur_layer_list_rand[counter]))
                    adv_patch_t=filter_image(blur_layer_list_rand[counter][cur_filter_rand2],adv_patch_t)
                else:
                    adv_patch_t=adv_patch_t+3
                    adv_patch_t=adv_patch_t/6
                    adv_patch_t=filter_image(blur_layer_list[counter],adv_patch_t,turbidity=cur_turb, sky_rbg=cur_sky, useblur=True)
                    adv_patch_t=adv_patch_t*6
                    adv_patch_t=adv_patch_t-3

                adv_patch_t=resize_transform_back_patch(adv_patch_t)
                adv_patch_t=run_kernel_sharpen(adv_patch_t,sharpen_1x,1,'normal')

                if cur_filter_rand<test_rand_thresh:
                    adv_patch_t=adv_patch_t+3
                    adv_patch_t=adv_patch_t/6
                    datahsv=rgb2hsv_torch(adv_patch_t)

                    cur_hue=datahsv[:,0:1,:,:]
                    cur_sat_x=(sat_minmax[1][counter]-sat_minmax[0][counter])*np.random.rand()+sat_minmax[0][counter]
                    cur_sat=datahsv[:,1:2,:,:]
                    cur_sat=cur_sat*cur_sat_x
                    cur_sat=torch.clamp(cur_sat,0,1)

                    cur_val_x=(val_minmax[1][counter]-val_minmax[0][counter])*np.random.rand()+val_minmax[0][counter]
                    cur_val=datahsv[:,2:3,:,:]
                    cur_val=cur_val*cur_val_x
                    cur_val=torch.clamp(cur_val,0,1)

                    new_hsv=[cur_hue,cur_sat,cur_val]
                    new_hsv=torch.cat(new_hsv,dim=1)

                    adv_patch_t=hsv2rgb_torch(new_hsv)
                    adv_patch_t=adv_patch_t*6
                    adv_patch_t=adv_patch_t-3


                adv_batch_t = patch_transformer(adv_patch_t, label_true, model_imgsize_list[0], do_rotate=False, rand_loc=False,
                                            pooling=pargs.pooling, old_fasion=old_fasion,ratio=cur_crop_size[1]/cur_crop_size[0], shift=0.07)
                pre_data=data.to(cfg.device)


                data = patch_applier(pre_data, adv_batch_t)

            full_res_clone=data.clone()

            data=resize_transform_back(data)

            resize_transform_cur = T.Resize(size = (model_imgsize_list[cur_model_counter],model_imgsize_list[cur_model_counter])).to(cfg.device)
            data=resize_transform_cur(data)

            data_ori['img_metas'][0]._data[0][0]['ori_shape']=(model_imgsize_list[cur_model_counter],model_imgsize_list[cur_model_counter],3)
            data_ori['img_metas'][0]._data[0][0]['img_shape']=(model_imgsize_list[cur_model_counter],model_imgsize_list[cur_model_counter],3)
            data_ori['img_metas'][0]._data[0][0]['pad_shape']=(model_imgsize_list[cur_model_counter],model_imgsize_list[cur_model_counter],3)
            if type(data_ori['img_metas'])==list:
                w=data_ori['img_metas'][0]._data[0][0]['ori_shape'][1]
                h=data_ori['img_metas'][0]._data[0][0]['ori_shape'][0]
            elif type(data_ori['img_metas'])!=list:
                w=data_ori['img_metas']._data[0][0]['ori_shape'][1]
                h=data_ori['img_metas']._data[0][0]['ori_shape'][0]
            whwh=torch.tensor([w,h,w,h])
            whwh=torch.reshape(whwh,[1,1,4])

            if type(data_ori['img'])==list:
                data_ori['img'][0]=data
            elif type(data_ori['img'])!=list:
                data_ori['img']=[data]

            if model_type_list[cur_model_counter]==0 or model_type_list[cur_model_counter]==1:
                result = model_list[cur_model_counter](return_loss=False, rescale=True, **data_ori)
            else:
                data_temp0=data+3
                data_temp1=data_temp0/6
                output = model_list[cur_model_counter](data_temp1)[0]
                pred = non_max_suppression(output, conf_thres=conf_thresh, iou_thres=iou_thresh, classes=[0])

                result = pred
            if type(data_ori['img_metas'])==list:
                w=data_ori['img_metas'][0]._data[0][0]['ori_shape'][1]
                h=data_ori['img_metas'][0]._data[0][0]['ori_shape'][0]
            elif type(data_ori['img_metas'])!=list:
                w=data_ori['img_metas']._data[0][0]['ori_shape'][1]
                h=data_ori['img_metas']._data[0][0]['ori_shape'][0]
            all_boxes=[]

            if model_type_list[cur_model_counter]==0:
                for i in range(len(result)):
                    batch_list=[]
                    for j in range(len(result[i])):
                        for k in range(result[i][j].shape[0]):
                            current=result[i][j][k,:]
                            x_w=current[0]/w
                            y_h=current[1]/h
                            w_w=current[2]/w
                            h_h=current[3]/h
                            det_confs=current[4]
                            cls_max_confs=current[4]
                            cls_max_ids=j
                            if det_confs>conf_thresh:
                                batch_list.append([x_w,y_h,w_w,h_h,det_confs,cls_max_confs,cls_max_ids])
                    batch_list=torch.tensor(np.asarray(batch_list))
                    all_boxes.append(batch_list)
            elif model_type_list[cur_model_counter]==1:
                for i in range(len(result)):
                    batch_list=[]
                    for j in range(len(result[i][0])):
                        for k in range(result[i][0][j].shape[0]):
                            current=result[i][0][j][k,:]
                            x_w=current[0]/w
                            y_h=current[1]/h
                            w_w=current[2]/w
                            h_h=current[3]/h
                            det_confs=current[4]
                            cls_max_confs=current[4]
                            cls_max_ids=j
                            if det_confs>conf_thresh:
                                batch_list.append([x_w,y_h,w_w,h_h,det_confs,cls_max_confs,cls_max_ids])
                    batch_list=torch.tensor(np.asarray(batch_list))
                    all_boxes.append(batch_list)
            elif model_type_list[cur_model_counter]==2:
                batch_list=[]
                for i in range(len(result)):
                    for j in range(result[i].shape[0]):
                        cur_res=result[i].clone().detach().cpu().numpy()
                        x_w=cur_res[j][0]/w
                        y_h=cur_res[j][1]/h
                        w_w=cur_res[j][2]/w
                        h_h=cur_res[j][3]/h
                        det_confs=cur_res[j][4]
                        cls_max_confs=cur_res[j][4]
                        cls_max_ids=cur_res[j][5]
                        if det_confs>conf_thresh:
                            batch_list.append([x_w,y_h,w_w,h_h,det_confs,cls_max_confs,cls_max_ids])
                batch_list=torch.tensor(np.asarray(batch_list))
                all_boxes.append(batch_list)

            if i_batch<30:
                img=255*(data.detach().cpu().numpy()+3)/6
                img=img[0]
                img=np.transpose(img,[1,2,0])

                image_whwh=torch.tensor([[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]])

                temp_list=[]
                if model_type_list[cur_model_counter]==0:
                    for i in range(len(result[0])):
                        if i==0:
                            matrix=torch.tensor(np.asarray(result[0][i]))
                            matrix=matrix[matrix[:,4]>0.5,:]
                            matrix[:,0:4]=matrix[:,0:4]/whwh
                            matrix[:,0:4]=matrix[:,0:4]*image_whwh
                            temp_list.append(matrix)
                        else:
                            temp_list.append(torch.rand([0,5]))
                elif model_type_list[cur_model_counter]==1:
                    for i in range(len(result[0])):
                        if i==0:
                            matrix=torch.tensor(np.asarray(result[0][0][i]))
                            matrix=matrix[matrix[:,4]>0.5,:]
                            matrix[:,0:4]=matrix[:,0:4]/whwh
                            matrix[:,0:4]=matrix[:,0:4]*image_whwh
                            temp_list.append(matrix)
                        else:
                            temp_list.append(torch.rand([0,5]))
                elif model_type_list[cur_model_counter]==2:
                    for i in range(80):
                        if i==0:
                            matrix=result[0].clone().cpu().detach()
                            matrix=matrix[:,0:5]
                            matrix=matrix[matrix[:,4]>0.5,:]
                            matrix[:,0:4]=matrix[:,0:4]/whwh
                            matrix[:,0:4]=matrix[:,0:4]*image_whwh
                            temp_list.append(matrix)
                        else:
                            temp_list.append(torch.rand([0,5]))
                if os.path.exists('./inspect_output')==False:
                    os.makedirs('./inspect_output')
                if os.path.exists('./inspect_output/'+args.suffix)==False:
                    os.makedirs('./inspect_output/'+args.suffix)
                if os.path.exists('./inspect_output/'+args.suffix+'/'+str(resolution)+patch_name)==False:
                    os.makedirs('./inspect_output/'+args.suffix+'/'+str(resolution)+patch_name)
                out_file='./inspect_output/'+args.suffix+'/'+str(resolution)+patch_name+'/'+'type_'+str(cur_model_counter)+'_'+str(i_batch)+patchname+'.jpg'

                inspect_output(img=img,result=temp_list,out_file=out_file)



            for i in range(len(all_boxes)):
                boxes = all_boxes[i]
                boxes = utils.nms(boxes, nms_thresh)
                truths = target[i].view(-1, 5)
                truths = label_filter(truths, labels=[0])
                num_gts=truths.shape[0]
                truths = truths[:num_gts, 1:]
                truths = truths.tolist()
                total = total + num_gts

                truths_filtered=[]
                found_list=[]
                for k in range(len(truths)):
                    current=truths[k]
                    area=current[2]*current[3]
                    if area>=min_area and area<max_area:
                        truths_filtered.append(current)
                        found_list.append(0)

                for j in range(len(boxes)):
                    boxes_temp=boxes[j].numpy()[0:4]
                    if boxes[j][6].item() == 0:
                        best_iou = 0
                        best_index = 0
                        for ib, box_gt in enumerate(truths_filtered):
                            box_gt_temp=whxy2xminyminxmaxymax(box_gt)
                            iou = utils.bbox_iou(box_gt_temp, boxes_temp,x1y1x2y2=True)

                            if iou > iou_thresh:
                                found_list[ib]=1

                total_labels+=len(found_list)
                for q in range(len(found_list)):
                    if found_list[q]==0:
                        successed+=1
                print(i_batch,'found_list',found_list)
                if total_labels!=0:
                    print('rate',successed/total_labels,'successed',successed,'total_labels',total_labels)



    return successed,total_labels
def form_label(physical_loader,sub_directs,data_base_dir,counter,resolution,model, loader, adv_cloth=None, gan=None, z=None, type_=None, conf_thresh=0.5, nms_thresh=0.5, iou_thresh=0.5, num_of_samples=100,
         old_fasion=True,train_test='train',saturations=[1,1],brightness=[1,1],val1xmin=0,val1xmax=1,sat1xmin=0,sat1xmax=1,keep_min=0.03):
    if train_test=='train':
        lab_dir=true_lab_dir
    else:
        lab_dir=true_lab_dir+'_temp_test'
        if os.path.exists(lab_dir):
            shutil.rmtree(lab_dir)
    if not os.path.exists(lab_dir):
        os.mkdir(lab_dir)
    print('---point3-----')
    model.eval()
    total = 0.0
    proposals = 0.0
    correct = 0.0
    batch_num = len(loader)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    positives = []
    print('len(loader)',len(loader),'--------------------------')

    if train_test=='train':
        img_list=physical_loader.get_all(sub_data_dirct_general_train,sub_directs[counter])
    else:
        img_list=physical_loader.get_all(sub_data_dirct_general,sub_directs[counter])

    for i_batch, data_ori in enumerate(loader):
        break

    for data_index in range(len(img_list)):

        data_ori['img_metas'][0]._data[0][0]['ori_shape']=(img_list[data_index][0].shape[2],img_list[data_index][0].shape[3],3)
        data_ori['img_metas'][0]._data[0][0]['img_shape']=(img_list[data_index][0].shape[2],img_list[data_index][0].shape[3],3)
        data_ori['img_metas'][0]._data[0][0]['pad_shape']=(img_list[data_index][0].shape[2],img_list[data_index][0].shape[3],3)
        data_ori['img_metas'][0]._data[0][0]['scale_factor']=np.asarray([1,1,1,1])
        data_ori['img_metas'][0]._data[0][0]['flip']=False
        data_ori['img_metas'][0]._data[0][0]['flip_direction']=None
        data = img_list[data_index][0]

        w=data_ori['img_metas'][0]._data[0][0]['ori_shape'][1]
        h=data_ori['img_metas'][0]._data[0][0]['ori_shape'][0]
        whwh=torch.tensor([w,h,w,h])
        whwh=torch.reshape(whwh,[1,1,4])

        img_batch=img_list[data_index][0]

        if type(data_ori['img'])==list:
            data_ori['img'][0]=data
        elif type(data_ori['img'])!=list:
            data_ori['img']=[data]

        resize_transform_back = T.Resize(size = (data.shape[2],data.shape[3])).to(cfg.device)
        data=resize_transform(data)

        data=resize_transform_back(data)

        resize_transform_cur = T.Resize(size = (model_imgsize_list[0],model_imgsize_list[0])).to(cfg.device)
        data=resize_transform_cur(data)

        data_ori['img_metas'][0]._data[0][0]['ori_shape']=(model_imgsize_list[0],model_imgsize_list[0],3)
        data_ori['img_metas'][0]._data[0][0]['img_shape']=(model_imgsize_list[0],model_imgsize_list[0],3)
        data_ori['img_metas'][0]._data[0][0]['pad_shape']=(model_imgsize_list[0],model_imgsize_list[0],3)
        if type(data_ori['img_metas'])==list:
            w=data_ori['img_metas'][0]._data[0][0]['ori_shape'][1]
            h=data_ori['img_metas'][0]._data[0][0]['ori_shape'][0]
        elif type(data_ori['img_metas'])!=list:
            w=data_ori['img_metas']._data[0][0]['ori_shape'][1]
            h=data_ori['img_metas']._data[0][0]['ori_shape'][0]
        whwh=torch.tensor([w,h,w,h])
        whwh=torch.reshape(whwh,[1,1,4])


        data_ori['img'][0]=data

        result = model(return_loss=False, rescale=True, **data_ori)
        if type(data_ori['img_metas'])==list:
            w=data_ori['img_metas'][0]._data[0][0]['ori_shape'][1]
            h=data_ori['img_metas'][0]._data[0][0]['ori_shape'][0]
        elif type(data_ori['img_metas'])!=list:
            w=data_ori['img_metas']._data[0][0]['ori_shape'][1]
            h=data_ori['img_metas']._data[0][0]['ori_shape'][0]
        all_boxes=[]
        label_name=img_list[data_index][3]
        label_name=label_name[0:len(label_name)-4]+'.txt'

        if model_type_list[0]==0:
            for i in range(len(result)):
                batch_list=[]
                for j in range(len(result[i])):
                    for k in range(result[i][j].shape[0]):
                        current=result[i][j][k,:]
                        x_w=current[0]/w
                        y_h=current[1]/h
                        w_w=current[2]/w
                        h_h=current[3]/h
                        det_confs=current[4]
                        cls_max_confs=current[4]
                        cls_max_ids=j
                        if det_confs>conf_thresh:
                            batch_list.append([x_w,y_h,w_w,h_h,det_confs,cls_max_confs,cls_max_ids])
                batch_list=torch.tensor(np.asarray(batch_list))
                all_boxes.append(batch_list)
        else:
            for i in range(len(result)):
                batch_list=[]
                for j in range(len(result[i][0])):
                    for k in range(result[i][0][j].shape[0]):
                        current=result[i][0][j][k,:]
                        x_w=current[0]/w
                        y_h=current[1]/h
                        w_w=current[2]/w
                        h_h=current[3]/h
                        det_confs=current[4]
                        cls_max_confs=current[4]
                        cls_max_ids=j
                        if det_confs>conf_thresh:
                            batch_list.append([x_w,y_h,w_w,h_h,det_confs,cls_max_confs,cls_max_ids])
                batch_list=torch.tensor(np.asarray(batch_list))
                all_boxes.append(batch_list)


        for i in range(len(all_boxes)):
            boxes = all_boxes[i]
            boxes = utils.nms(boxes, nms_thresh)
            if 1!=len(boxes.shape):
                new_boxes = boxes[:, [6, 0, 1, 2, 3]]
                new_boxes = new_boxes[new_boxes[:, 0] == 0]
                new_boxes = new_boxes.detach().cpu().numpy()


                lab=new_boxes[:,0:1]
                xm=new_boxes[:,1:2]
                ym=new_boxes[:,2:3]
                xM=new_boxes[:,3:4]
                yM=new_boxes[:,4:5]
                w=(xM-xm)
                h=(yM-ym)
                x=xm+w/2
                y=ym+h/2
                new_boxes=np.concatenate([lab,x,y,w,h],1)

                new_boxes_=[]
                for box_i in range(len(new_boxes)):
                    if new_boxes[box_i][3]*new_boxes[box_i][4]>keep_min:
                        new_boxes_.append(new_boxes[box_i])
                new_boxes=np.asarray(new_boxes_)


            else:
                new_boxes=boxes

            cur_dirct=lab_dir+ '/'+ img_list[data_index][1]
            if os.path.exists(cur_dirct)==False:
                os.makedirs(cur_dirct)
            cur_dirct=cur_dirct+'/'+ img_list[data_index][2]
            if os.path.exists(cur_dirct)==False:
                os.makedirs(cur_dirct)
            save_dir = cur_dirct + '/' + label_name
            np.savetxt(save_dir, new_boxes, fmt='%f')


    print('preparing done')

def random_init_pixels(size,colors):
    interval_size=1/len(colors)
    output=[]
    for i in range(size):
        list2=[]
        for j in range(size):
            cur=np.random.rand()
            cur_index=int(cur/interval_size)
            list2.append(colors[cur_index])
        output.append(list2)
    output=np.asarray(output)
    output=torch.tensor(output)
    output=torch.transpose(output,0,2)
    resize_temp = T.Resize(size = (300,300))
    output=resize_temp(output)
    output_=torch.unsqueeze(output,dim=0)
    output_=output_*6-3
    output_=output_.float()
    return output_


def get_adjustable_kernel(k_size,channels,temperature,height,width):
    x_start=-int(k_size/2)
    x_end=int(k_size/2)
    y_start=-int(k_size/2)
    y_end=int(k_size/2)

    result=[]
    sigmoid_func=torch.nn.Sigmoid()
    for x in range(k_size):
        temp_list=[]
        for y in range(k_size):
            curx=x_start+x
            cury=y_start+y

            cur_sigx=np.sqrt(curx*curx+cury*cury)
            forward_x=cur_sigx+(width/2)
            backward_x=-1*cur_sigx+(width/2)
            forward_x=torch.tensor(forward_x)
            backward_x=torch.tensor(backward_x)
            sigmoid_func=torch.nn.Sigmoid()
            forward_y=sigmoid_func(forward_x*temperature)
            backward_y=sigmoid_func(backward_x*temperature)

            sigy=height*(forward_y+backward_y-1).numpy()

            temp_list.append(sigy)
        result.append(temp_list)

    result=torch.tensor(result).float()
    result=torch.unsqueeze(result,0)
    result=torch.unsqueeze(result,0)
    result=result/torch.sum(result)
    result=result.repeat(1,channels,1,1)

    result=torch.transpose(result,0,1)

    return result

def get_adjustable_kernelv4(k_size,channels,temperature,height,width):
    x_start=-int(k_size/2)
    x_end=int(k_size/2)
    y_start=-int(k_size/2)
    y_end=int(k_size/2)

    result=[]
    sigmoid_func=torch.nn.Sigmoid()
    for x in range(k_size):
        temp_list=[]
        for y in range(k_size):
            curx=x_start+x
            cury=y_start+y
            cur_sigx=np.max([np.abs(curx),np.abs(cury)])
            forward_x=cur_sigx+(width/2)
            backward_x=-1*cur_sigx+(width/2)
            forward_x=torch.tensor(forward_x)
            backward_x=torch.tensor(backward_x)
            sigmoid_func=torch.nn.Sigmoid()
            forward_y=sigmoid_func(forward_x*temperature)
            backward_y=sigmoid_func(backward_x*temperature)

            sigy=height*(forward_y+backward_y-1).numpy()

            temp_list.append(sigy)
        result.append(temp_list)

    result=torch.tensor(result).float()
    result=torch.unsqueeze(result,0)
    result=torch.unsqueeze(result,0)
    result=result/torch.sum(result)
    result=result.repeat(channels,1,1,1)

    return result

def run_filter_adjustablev4(input,kernel,transform,stride):
    in_s=input.shape
    ker_s=kernel.shape

    total_step=(in_s[2]+1)//stride
    pad_num=int(((total_step)*(stride)+ker_s[2]-1-in_s[2]))

    filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_num,pad_num],groups=3)
    if transform!=None:
        filtered=transform(filtered)

    return filtered

def run_filter_adjustable(input,kernel):
    in_s=input.shape
    ker_s=kernel.shape

    total_step=in_s[2]-(ker_s[2]-1)
    pad_num=int((in_s[2]-total_step)/2)

    filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=1,padding=pad_num,groups=3)

    return filtered

def get_blur_layers(cur_config):
    if cur_config==None:
        return None
    elif cur_config!=None:
        if cur_config[0]=='gaussian_blur':
            blur_layer=torchvision.transforms.GaussianBlur(cur_config[1], sigma=cur_config[2]).to(device)
            return [cur_config[0],blur_layer]
        elif cur_config[0]=='adjustablev1':
            channels=3
            height=1
            k_size=cur_config[1]
            temperature=cur_config[2]
            width=cur_config[3]
            adjustable_kernel=get_adjustable_kernel(k_size,channels,temperature,height,width)
            adjustable_kernel=adjustable_kernel.to(device)
            return [cur_config[0],adjustable_kernel]
        elif cur_config[0]=='adjustablev2':
            channels=3
            height=1
            k_size=cur_config[1]
            temperature=cur_config[2]
            width=cur_config[3]
            stride=cur_config[4]
            adjustable_kernel=get_adjustable_kernelv4(k_size,channels,temperature,height,width)
            adjustable_kernel=adjustable_kernel.to(device)
            return [cur_config[0],adjustable_kernel,stride]
        elif cur_config[0]=='styled':
            return cur_config

def filter_image(cur_blurlayer,p_img_batch, turbidity='self', sky_rbg='self', useblur=True,use_style=True):
    if cur_blurlayer==None:
        return p_img_batch
    else:
        if cur_blurlayer[0]=='gaussian_blur':
            p_img_batch=cur_blurlayer[1](p_img_batch)
        elif cur_blurlayer[0]=='adjustablev1':
            p_img_batch=run_filter_adjustable(p_img_batch,cur_blurlayer[1])
            p_img_batch=torch.clamp(p_img_batch,-3,3)
        elif cur_blurlayer[0]=='adjustablev2':
            resize_transform_ = T.Resize(size = (p_img_batch.shape[2],p_img_batch.shape[3]))
            p_img_batch=run_filter_adjustablev4(p_img_batch,cur_blurlayer[1],resize_transform_,cur_blurlayer[2])
            p_img_batch=torch.clamp(p_img_batch,-3,3)
        elif cur_blurlayer[0]=='styled':
            blur_module=cur_blurlayer[1]
            filters=cur_blurlayer[2]
            cur_counter=cur_blurlayer[3]
            blur_eot_c=cur_blurlayer[4]

            resize_final = T.Resize(size = [p_img_batch.shape[2],p_img_batch.shape[3]])
            p_img_batch=blur_module.run_blurring(p_img_batch,cur_counter,eot_size=blur_eot_c, turbidity=turbidity, sky_rbg=sky_rbg, useblur=useblur)
            p_img_batch = filters.run_manual_crop(p_img_batch,cur_counter)
            if use_style==True:
                p_img_batch = filters.run_style_filter(p_img_batch,blur_eot_c)
            p_img_batch = resize_final(p_img_batch)



        return p_img_batch

class MappingNet(nn.Module):
    def __init__(self,width):
        super(MappingNet,self).__init__()

        self.fc1 = nn.Linear(3, width)
        self.bn1 = torch.nn.BatchNorm1d(width)

        self.bn1 = torch.nn.BatchNorm1d(width)

        self.block1 = nn.Sequential(
                        nn.Linear(width, width),
                        nn.ReLU(),
                        )

        self.fcout = nn.Linear(width, 3)
        self.out_sig=nn.Sigmoid()

    def forward(self,x):
        x_ori=x
        x=self.fc1(x)
        x=self.bn1(x)
        x=x+self.block1(x)
        x=self.fcout(x)
        x=self.out_sig(x)*2-1+x_ori
        x=torch.clamp(x,0,1)
        return x


def rgb2hsv_torch(img):
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + 0.0001 ) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + 0.0001 ) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + 0.0001 ) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + 0.0001 )
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value],dim=1)
    return hsv

def hsv2rgb_torch(hsv):
    h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
    h = h%1
    s = torch.clamp(s,0,1)
    v = torch.clamp(v,0,1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    hi0 = hi==0
    hi1 = hi==1
    hi2 = hi==2
    hi3 = hi==3
    hi4 = hi==4
    hi5 = hi==5

    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]

    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]

    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]

    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]

    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]

    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    return rgb

def get_satval_minmax(saturation_data,brightness_data,min_eot_range):
    sat_minmax=[[],[]]
    val_minmax=[[],[]]
    for i in range(len(saturation_data[0])):
        cur_sats=[]
        cur_vals=[]
        for j in range(len(saturation_data)):
            cur_sats.append(saturation_data[j][i])
            cur_vals.append(brightness_data[j][i])
        cur_sats=np.asarray(cur_sats)
        cur_vals=np.asarray(cur_vals)

        cur_smin=np.min(cur_sats)
        cur_smax=np.max(cur_sats)
        cur_vmin=np.min(cur_vals)
        cur_vmax=np.max(cur_vals)

        if cur_smax-cur_smin<min_eot_range:
            cur_savg=(cur_smax+cur_smin)/2
            cur_smin=cur_savg-(min_eot_range/2)
            cur_smax=cur_savg+(min_eot_range/2)
        if cur_vmax-cur_vmin<min_eot_range:
            cur_vavg=(cur_vmax+cur_vmin)/2
            cur_vmin=cur_vavg-(min_eot_range/2)
            cur_vmax=cur_vavg+(min_eot_range/2)

        sat_minmax[0].append(cur_smin)
        sat_minmax[1].append(cur_smax)
        val_minmax[0].append(cur_vmin)
        val_minmax[1].append(cur_vmax)
    return sat_minmax,val_minmax

def adjust_val_sat_in_range(val1xmin,val1xmax,cur_val):
    cur_val_mean=torch.mean(cur_val).clone().detach().cpu().numpy()
    if cur_val_mean<val1xmin or cur_val_mean>val1xmax:
        val_mult=(val1xmax-val1xmin)*np.random.rand()+val1xmin
        val_mult=val_mult/cur_val_mean
        cur_val=cur_val*val_mult
    return cur_val


class BlurModule:
    def __init__(self, strides, widths, temperatures,widths2,temperatures2, N_exp, c_exp, lambda_divide, rbg_lambda, sky_rbg, turbidity,  sv_shift, device, trainable=True,load=True, model_name='BlurModule_default',num_days=1):

        self.variable_list=[]
        load_index=0

        self.trainable=trainable
        self.device=device
        self.load=load
        self.model_name=model_name

        self.sigmoid_func=torch.nn.Sigmoid()
        self.strides=strides
        self.kernel_sizes=[]
        for i in range(len(self.strides)):
            cur_ksize=int(self.strides[i]+2)
            if cur_ksize<3:
                cur_ksize=3
            elif cur_ksize%3!=0:
                cur_ksize=int(cur_ksize+1)
            self.kernel_sizes.append(cur_ksize)
        self.widths=widths
        self.temperatures=temperatures
        self.widths2=widths2
        self.temperatures2=temperatures2
        self.height=1
        self.channels=3

        self.widths=torch.tensor(widths).to(device)
        self.widths,load_index,self.variable_list=self.make_variable_differentiable(self.widths,self.load,load_index,self.variable_list)
        self.temperatures=torch.tensor(temperatures).to(device)

        self.widths2=torch.tensor(widths2).to(device)
        self.widths2,load_index,self.variable_list=self.make_variable_differentiable(self.widths2,self.load,load_index,self.variable_list)
        self.temperatures2=torch.tensor(temperatures2).to(device)
        self.N_exp=torch.tensor(N_exp).to(device)
        self.N_exp,load_index,self.variable_list=self.make_variable_differentiable(self.N_exp,self.load,load_index,self.variable_list)
        self.c_exp=torch.tensor(c_exp).to(device)
        self.c_exp,load_index,self.variable_list=self.make_variable_differentiable(self.c_exp,self.load,load_index,self.variable_list)
        self.lambda_divide=torch.tensor(lambda_divide).to(device)
        self.lambda_divide,load_index,self.variable_list=self.make_variable_differentiable(self.lambda_divide,self.load,load_index,self.variable_list)
        self.rbg_lambda=torch.tensor(rbg_lambda).to(device)
        self.rbg_lambda,load_index,self.variable_list=self.make_variable_differentiable(self.rbg_lambda,self.load,load_index,self.variable_list)
        self.sky_rbg=torch.tensor(sky_rbg).to(device)
        temp_turb=[]
        for i in range(num_days):
            temp_turb.append(turbidity)
        self.turbidity=torch.tensor(temp_turb).to(device)
        self.turbidity,load_index,self.variable_list=self.make_variable_differentiable(self.turbidity,self.load,load_index,self.variable_list)
        sv_temp=[]
        for i in range(num_days):
            sv_temp.append(sv_shift)
        self.sv_shift=torch.tensor(sv_temp).to(device)
        self.sv_shift,load_index,self.variable_list=self.make_variable_differentiable(self.sv_shift,self.load,load_index,self.variable_list)
        self.sv_shift0=torch.tensor([0,0]).to(device)

        self.blur_base=[]
        self.blur_base_v4=[]
        for k in range(len(self.kernel_sizes)):
            cur_blur_base=[]
            cur_blur_base_v4=[]
            for i in range(int(self.kernel_sizes[k])):
                cur=[]
                for j in range(int(self.kernel_sizes[k])):
                    cur.append([i,j])
                cur_blur_base.append(cur)
                cur_blur_base_v4.append(cur)
            cur_blur_base=torch.tensor(np.asarray(cur_blur_base)).to(self.device)
            self.blur_base.append(cur_blur_base)
            self.blur_base_v4.append(cur_blur_base)
        self.EOT_temp=1
        self.EOT_N_exp=0.4
        self.EOT_c_exp=0.4
        self.EOT_lambda_divide=2
        self.EOT_rbg_lambda=10
        self.EOT_sky_rbg=0.05
        self.EOT_turbidity=0.3
        self.EOT_blur_width=[0.1,0.9]
        self.EOT_blur_width2=[0.1,0.9]



        self.dist_list=[4,8,14,20,26,34,42]


    def make_variable_differentiable(self,variable,load,load_index,variable_list):
        if load==True:
            value=np.load('./model/'+self.model_name+'/'+str(load_index)+'.npy')
            variable=value
            load_index+=1
        variable = torch.tensor(variable).float()
        variable = variable.to(self.device)
        if self.trainable==True:
            variable.requires_grad_(True)
        variable_list.append(variable)
        return variable,load_index,variable_list


    def save(self,epoch):
        save_dir='./model/'+self.model_name
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)
        for i in range(len(self.variable_list)):
            cur_name=save_dir+'/'+str(epoch)+'_'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())
            cur_name=save_dir+'/'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())

    def get_adjustable_kernelv4(self,config,counter):
        k_size=config[0]
        channels=config[5]
        temperature=config[2]
        height=config[4]
        width=config[1]

        temperature=torch.clamp(temperature,0,100)
        width=torch.clamp(width,0.1,100)

        x_start=-k_size//2
        x_end=k_size//2
        y_start=-k_size//2
        y_end=k_size//2

        result=self.blur_base_v4[counter]
        cur_x=result[:,:,0:1]+x_start
        cur_y=result[:,:,1:2]+y_start
        cur_x_abs=torch.abs(cur_x)
        cur_y_abs=torch.abs(cur_y)
        cur_xy=torch.cat([cur_x_abs,cur_y_abs],2)
        cur_sigx,_=torch.max(cur_xy,2)
        forward_x=cur_sigx+(width/2)
        backward_x=-1*cur_sigx+(width/2)
        forward_y=self.sigmoid_func(forward_x*temperature)
        backward_y=self.sigmoid_func(backward_x*temperature)
        result=height*(forward_y+backward_y-1)

        result=torch.unsqueeze(result,0)
        result=torch.unsqueeze(result,0)
        result=result/torch.sum(result)
        result=result.repeat(channels,1,1,1)

        return result

    def get_blurring_kernel_v4(self,kernel_size,width,temp,stride,height,channels,counter):
        blur_kernel=self.get_adjustable_kernelv4([kernel_size,width,temp,stride,height,channels],counter)

        return blur_kernel

    def get_adjustable_kernel(self,config,counter):
        k_size=config[0]
        channels=config[5]
        temperature=config[2]
        temperature=torch.clamp(temperature,0,100)
        height=config[4]
        width=config[1]
        width=torch.clamp(width,0.1,100)

        x_start=-k_size//2
        x_end=k_size//2
        y_start=-k_size//2
        y_end=k_size//2

        result=self.blur_base[counter]
        cur_x=result[:,:,0]+x_start
        cur_y=result[:,:,1]+y_start
        cur_sigx=torch.sqrt(cur_x*cur_x+cur_y*cur_y)
        forward_x=cur_sigx+(width/2)
        backward_x=-1*cur_sigx+(width/2)
        forward_y=self.sigmoid_func(forward_x*temperature)
        backward_y=self.sigmoid_func(backward_x*temperature)
        result=height*(forward_y+backward_y-1)

        result=torch.unsqueeze(result,0)
        result=torch.unsqueeze(result,0)
        result=result/torch.sum(result)
        result=result.repeat(channels,1,1,1)

        return result

    def get_blurring_kernel(self,kernel_size,width,temp,stride,height,channels,counter):

        blur_kernel=self.get_adjustable_kernel([kernel_size,width,temp,stride,height,channels],counter)

        return blur_kernel

    def run_kernel(self,input,kernel,stride,type):
        in_s=input.shape
        ker_s=kernel.shape

        pad_crop_config=[]

        for i in range(2):
            decision_value=(in_s[2+i]-1)%stride
            if decision_value==0:
                pad_num=ker_s[2+i]//2
                if type=='expand':
                    pad_num=pad_num+stride
                remove1=False
                pad_crop_config.append([pad_num,remove1])
            else:
                if decision_value%2!=0:
                    remove1=True
                    pad_num=(ker_s[2+i]//2)-((decision_value-1)//2)
                else:
                    remove1=False
                    pad_num=(ker_s[2+i]//2)-((decision_value)//2)
                    if type=='expand':
                        pad_num=pad_num+stride
                pad_crop_config.append([pad_num,remove1])

        for i in range(len(pad_crop_config)):
            if i==0 and pad_crop_config[i][1]==True:
                input=input[:,:,0:in_s[2]-1,:]
            elif i==1 and pad_crop_config[i][1]==True:
                input=input[:,:,:,0:in_s[3]-1]

        filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_crop_config[0][0],pad_crop_config[1][0]],groups=3)
        return filtered

    def calculate_gaussian(self,x,hue_mean,hue_std):
        cur_mean=torch.clamp(hue_mean,0,1)
        cur_mean=torch.reshape(cur_mean,[1,cur_mean.shape[0],1,1])
        cur_mean=cur_mean.repeat(1,1,x.shape[2],x.shape[3])
        diffs=(x-cur_mean)*(x-cur_mean)

        cur_std=torch.clamp(hue_std,0,1)
        cur_std=torch.reshape(cur_std,[1,cur_std.shape[0],1,1])
        cur_std=cur_std.repeat(1,1,x.shape[2],x.shape[3])
        cur_std=cur_std+0.001
        cur_std_square=cur_std*cur_std

        results=(1/(cur_std_square*2*3.14))*torch.exp((-0.5)*(diffs/cur_std_square))
        results=torch.mean(results,dim=1,keepdims=True)
        return results

    def get_exp_value(self,T,Lambda,s,h_0=1,eot_size=0):
        n=1.0003

        change_val=self.EOT_N_exp*np.random.rand()-self.EOT_N_exp/2
        N_exp_temp=self.N_exp+change_val*eot_size
        N_exp_temp=torch.clamp(N_exp_temp,0.1,100)

        N=2.545*(10**N_exp_temp)
        p_n=0.035
        H_r0=7994
        K_R=1.0396

        change_val=self.EOT_lambda_divide*np.random.rand()-self.EOT_lambda_divide/2
        lambda_divide_temp=self.lambda_divide+change_val*eot_size
        lambda_divide_temp=torch.clamp(lambda_divide_temp,0.1,100)
        Lambda_cur=Lambda/(lambda_divide_temp*1000)

        beta_r=(8*(np.pi**3)*((n*n-1)**2))/(3*N*(Lambda_cur**4))*((6+3*p_n)/(6-7*p_n))*np.exp(-1*(h_0/H_r0))*K_R
        change_val=self.EOT_c_exp*np.random.rand()-self.EOT_c_exp/2
        c_exp_temp=self.c_exp+change_val*eot_size
        c_exp_temp=torch.clamp(c_exp_temp,0.1,100)
        if T=='self':
            c=(0.6544*self.turbidity-0.6510)*(1/(10**c_exp_temp))
        else:
            c=(0.6544*T-0.6510)*(1/(10**c_exp_temp))
        K_M=0.0092
        v=4
        H_m0=1200

        beta_m=0.434*c*np.pi*((2*np.pi/Lambda_cur)**(v-2))*np.exp(-1*h_0/H_m0)*K_M

        result=torch.exp(-1*(beta_r+beta_m)*s)

        result=torch.clamp(result,0,1)

        return result

    def run_blurring(self,input,counter,turbidity='self', sky_rbg='self', eot_size=0, useblur=True, day_index='default',blur_eot_factor=15):

        filtered=input

        if eot_size==0:
            cur_width=self.widths*self.strides[counter]
        else:
            cur_width=self.widths*self.strides[counter]+eot_size*blur_eot_factor*(np.random.rand()-0.5)
            cur_width=torch.clamp(cur_width,self.EOT_blur_width[0]*self.strides[counter],self.EOT_blur_width[1]*self.strides[counter])
        cur_temp=self.temperatures[counter]

        blur_kernel=self.get_blurring_kernel(self.kernel_sizes[counter],cur_width,cur_temp,1,self.height,self.channels,counter)

        if eot_size==0:
            cur_width2=self.widths2*self.strides[counter]
        else:
            cur_width2=self.widths2*self.strides[counter]+eot_size*blur_eot_factor*(np.random.rand()-0.5)
            cur_width2=torch.clamp(cur_width2,self.EOT_blur_width2[0]*self.strides[counter],self.EOT_blur_width2[1]*self.strides[counter])
        cur_temp2=self.temperatures2

        cur_ksize2=int(self.strides[counter]+2)
        if cur_ksize2<3:
            cur_ksize2=3
        elif cur_ksize2%3!=0:
            cur_ksize2=int(cur_ksize2+1)

        blur_kernel2=self.get_blurring_kernel_v4(self.kernel_sizes[counter],cur_width2,cur_temp2,self.strides[counter],self.height,self.channels,counter)

        change_val=self.EOT_rbg_lambda*np.random.rand()-self.EOT_rbg_lambda/2
        rbg_lambda_temp=self.rbg_lambda+change_val*eot_size
        rbg_lambda_temp=torch.clamp(rbg_lambda_temp,38,78)
        change_val=self.EOT_turbidity*np.random.rand()-self.EOT_turbidity/2
        if turbidity=='self':
            turbidity_temp=self.turbidity[day_index]+change_val*eot_size
        else:
            turbidity_temp=turbidity+change_val*eot_size
        turbidity_temp=torch.clamp(turbidity_temp,1,20)

        cur_rgbl=torch.reshape((rbg_lambda_temp*10),[1,3,1,1])
        cur_exp=self.get_exp_value(turbidity_temp,cur_rgbl,self.dist_list[counter],eot_size=eot_size)
        cur_exp=torch.Tensor.repeat(cur_exp,[1,1,filtered.shape[2],filtered.shape[3]])

        if sky_rbg=='self':
            change_val=self.EOT_sky_rbg*np.random.rand()-self.EOT_sky_rbg/2
            sky_rbg_temp=self.sky_rbg+change_val*eot_size
            sky_rbg_temp=torch.clamp(sky_rbg_temp,0,1)
            cur_sky_rbg=torch.reshape(sky_rbg_temp,[1,3,1,1])
        else:
            change_val=self.EOT_sky_rbg*np.random.rand()-self.EOT_sky_rbg/2
            sky_rbg_temp=sky_rbg+change_val*eot_size
            sky_rbg_temp=torch.clamp(sky_rbg_temp,0,1)
            cur_sky_rbg=torch.reshape(sky_rbg_temp,[1,3,1,1])
        cur_sky_rbg=torch.Tensor.repeat(cur_sky_rbg,[1,1,filtered.shape[2],filtered.shape[3]])

        filtered=filtered*cur_exp+cur_sky_rbg*(1-cur_exp)

        filtered_hsv=rgb2hsv_torch(filtered)
        filtered_h=filtered_hsv[:,0:1,:,:]
        filtered_s=filtered_hsv[:,1:2,:,:]
        filtered_v=filtered_hsv[:,2:3,:,:]
        if day_index=='default':
            filtered_s=filtered_s+self.sv_shift0[0]
            filtered_v=filtered_v+self.sv_shift0[1]
        else:
            filtered_s=filtered_s+self.sv_shift[day_index][0]
            filtered_v=filtered_v+self.sv_shift[day_index][1]
        filtered_hsv=torch.cat([filtered_h,filtered_s,filtered_v],1)
        filtered_hsv=torch.clamp(filtered_hsv,0,1)
        filtered=hsv2rgb_torch(filtered_hsv)
        filtered=torch.clamp(filtered,0,1)

        if useblur==True:
            cur_resize = T.Resize(size = [filtered.shape[2],filtered.shape[3]])
            filtered=self.run_kernel(filtered,blur_kernel,1,'expand')
            filtered=cur_resize(filtered)
            filtered=self.run_kernel(filtered,blur_kernel2,self.strides[counter],'same')
            filtered=torch.clamp(filtered,0,1)
        else:
            filtered=input
            filtered=torch.clamp(filtered,0,1)


        return filtered



class StyleFilters:
    def __init__(self, sharpen_configs, desharpen_configs, device, contrast_val,vibrance_val,shahigh_val,exposure_val,color_temp_r_b,model_name='stylefilters_default',load=False,manual_keep=None,trainable=True):


        self.trainable=trainable

        self.manual_keep=manual_keep
        load_index=0
        self.device=device

        self.model_name=model_name

        self.variable_list=[]

        self.sharpen_configs=sharpen_configs

        self.desharpen_configs,self.desharpen_diff_indexes,load_index=self.preprocess_configs(desharpen_configs,self.variable_list,load,load_index)

        self.sigmoid_func=torch.nn.Sigmoid()

        self.desharpen_base=[]
        for i in range(int(self.desharpen_configs[0])):
            cur=[]
            for j in range(int(self.desharpen_configs[0])):
                cur.append([i,j])
            self.desharpen_base.append(cur)
        self.desharpen_base=torch.tensor(np.asarray(self.desharpen_base)).to(self.device)

        self.contrast_val=torch.tensor(contrast_val).to(device)
        self.contrast_val,load_index,self.variable_list=self.make_variable_differentiable(self.contrast_val,load,load_index,self.variable_list)

        self.vibrance_val=torch.tensor(vibrance_val).to(device)
        self.vibrance_val,load_index,self.variable_list=self.make_variable_differentiable(self.vibrance_val,load,load_index,self.variable_list)

        self.shahigh_val=torch.tensor(np.asarray(shahigh_val)).to(device)
        self.shahigh_val,load_index,self.variable_list=self.make_variable_differentiable(self.shahigh_val,load,load_index,self.variable_list)

        self.exposure_val=torch.tensor(exposure_val).to(device)
        self.exposure_val,load_index,self.variable_list=self.make_variable_differentiable(self.exposure_val,load,load_index,self.variable_list)

        self.color_temp_r_b=torch.tensor(color_temp_r_b).to(device)
        self.color_temp_r_b,load_index,self.variable_list=self.make_variable_differentiable(self.color_temp_r_b,load,load_index,self.variable_list)
        self.EOT_ds_width=0.3
        self.EOT_ds_temp=1
        self.EOT_ob_width=0.8
        self.EOT_ob_temp=1
        self.EOT_contrast=0.08
        self.EOT_vibrance=0.15
        self.EOT_shahigh=[[0.15,0.15]]
        self.EOT_exposure=0.1
        self.EOT_color_temp_r_b=0.03
        self.EOT_gmm_prob=0.5

    def make_variable_differentiable(self,variable,load,load_index,variable_list):
        if load==True:
            value=np.load('./model/'+self.model_name+'/'+str(load_index)+'.npy')
            variable=value
            load_index+=1
        variable = torch.tensor(variable).float()
        variable = variable.to(self.device)
        if self.trainable==True:
            variable.requires_grad_(True)
        variable_list.append(variable)
        return variable,load_index,variable_list

    def save(self,epoch):
        save_dir='./model/'+self.model_name
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)
        for i in range(len(self.variable_list)):
            cur_name=save_dir+'/'+str(epoch)+'_'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())
            cur_name=save_dir+'/'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())

    def preprocess_configs(self,configs,all_variables,load,load_index):
        blur_configs=configs[0]
        blur_diff_indexes=configs[1]
        for i in range(len(blur_diff_indexes)):
            if load==True:
                value=np.load('./model/'+self.model_name+'/'+str(load_index)+'.npy')
                blur_configs[blur_diff_indexes[i]]=value
                load_index+=1
            blur_configs[blur_diff_indexes[i]] = torch.tensor(blur_configs[blur_diff_indexes[i]]).float()
            blur_configs[blur_diff_indexes[i]] = blur_configs[blur_diff_indexes[i]].to(self.device)
            if self.trainable==True:
                blur_configs[blur_diff_indexes[i]].requires_grad_(True)
            all_variables.append(blur_configs[blur_diff_indexes[i]])
        return blur_configs,blur_diff_indexes,load_index


    def get_adjustable_kernel(self,config,base):
        k_size=config[0]
        channels=config[5]
        temperature=config[2]
        temperature=torch.clamp(temperature,0,100)
        height=config[4]
        width=config[1]
        width=torch.clamp(width,0.1,100)

        x_start=-k_size//2
        x_end=k_size//2
        y_start=-k_size//2
        y_end=k_size//2

        result=base
        cur_x=result[:,:,0]+x_start
        cur_y=result[:,:,1]+y_start
        cur_sigx=torch.sqrt(cur_x*cur_x+cur_y*cur_y)
        forward_x=cur_sigx+(width/2)
        backward_x=-1*cur_sigx+(width/2)
        forward_y=self.sigmoid_func(forward_x*temperature)
        backward_y=self.sigmoid_func(backward_x*temperature)
        result=height*(forward_y+backward_y-1)

        result=torch.unsqueeze(result,0)
        result=torch.unsqueeze(result,0)
        result=result/torch.sum(result)
        result=result.repeat(channels,1,1,1)

        return result


    def run_filter_adjustable(self,input,kernel,stride,type='normal'):
        in_s=input.shape
        ker_s=kernel.shape

        pad_crop_config=[]

        for i in range(2):
            decision_value=(in_s[2+i]-1)%stride
            if decision_value==0:
                pad_num=ker_s[2+i]//2
                if type=='expand':
                    pad_num=pad_num+stride
                remove1=False
                pad_crop_config.append([pad_num,remove1])
            else:
                if decision_value%2!=0:
                    remove1=True
                    pad_num=(ker_s[2+i]//2)-((decision_value-1)//2)
                else:
                    remove1=False
                    pad_num=(ker_s[2+i]//2)-((decision_value)//2)
                    if type=='expand':
                        pad_num=pad_num+stride
                pad_crop_config.append([pad_num,remove1])

        for i in range(len(pad_crop_config)):
            if i==0 and pad_crop_config[i][1]==True:
                input=input[:,:,0:in_s[2]-1,:]
            elif i==1 and pad_crop_config[i][1]==True:
                input=input[:,:,:,0:in_s[3]-1]

        filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_crop_config[0][0],pad_crop_config[1][0]],groups=3)

        return filtered

    def get_sharpen_kernel(self,sharp_type,sharp_k_size):
        middle=sharp_k_size//2
        if sharp_type=='full':
            center_val=sharp_k_size*sharp_k_size

            kernel=-1*np.ones([3,1,sharp_k_size,sharp_k_size])
            for i in range(3):
                kernel[i,0,middle,middle]=center_val

        elif sharp_type=='cross':
            center_val=(sharp_k_size*2-1)

            kernel=np.zeros([3,1,sharp_k_size,sharp_k_size])
            for i in range(3):
                for j in range(sharp_k_size):
                    kernel[i,0,middle,j]=-1
                    kernel[i,0,j,middle]=-1
                    if j==middle:
                        kernel[i,0,j,j]=center_val

        kernel=torch.tensor(kernel).float().to(self.device)
        return kernel





    def get_blur_sharp(self,sharpen_configs,desharpen_configs):
        sharp_kernel=self.get_sharpen_kernel(sharpen_configs[0],sharpen_configs[1])
        desharpen_kernel=self.get_adjustable_kernel(desharpen_configs,self.desharpen_base)

        return sharp_kernel,desharpen_kernel

    def run_kernel(self,input,kernel,stride,type):
        in_s=input.shape
        ker_s=kernel.shape

        pad_crop_config=[]

        for i in range(2):
            decision_value=(in_s[2+i]-1)%stride
            if decision_value==0:
                pad_num=ker_s[2+i]//2
                if type=='expand':
                    pad_num=pad_num+stride
                remove1=False
                pad_crop_config.append([pad_num,remove1])
            else:
                if decision_value%2!=0:
                    remove1=True
                    pad_num=(ker_s[2+i]//2)-((decision_value-1)//2)
                else:
                    remove1=False
                    pad_num=(ker_s[2+i]//2)-((decision_value)//2)
                    if type=='expand':
                        pad_num=pad_num+stride
                pad_crop_config.append([pad_num,remove1])

        for i in range(len(pad_crop_config)):
            if i==0 and pad_crop_config[i][1]==True:
                input=input[:,:,0:in_s[2]-1,:]
            elif i==1 and pad_crop_config[i][1]==True:
                input=input[:,:,:,0:in_s[3]-1]

        filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_crop_config[0][0],pad_crop_config[1][0]],groups=3)
        return filtered

    def run_sharpen(self,input,eot_size=0):
        desharpen_configs_temp=self.desharpen_configs.copy()
        change_val=self.EOT_ds_width*np.random.rand()-self.EOT_ds_width/2
        desharpen_configs_temp[1]=self.desharpen_configs[1]+change_val*eot_size
        change_val=self.EOT_ds_temp*np.random.rand()-self.EOT_ds_temp/2
        desharpen_configs_temp[2]=self.desharpen_configs[2]+change_val*eot_size


        kernels=self.get_blur_sharp(self.sharpen_configs,
                            desharpen_configs_temp)
        filtered=self.run_kernel(input,kernels[0],1,'normal')
        filtered=self.run_filter_adjustable(filtered,kernels[1],self.desharpen_configs[3])
        filtered=torch.clamp(filtered,0,1)

        return filtered

    def run_manual_crop(self,input,counter):
        result=input[:,:,self.manual_keep[counter][0][0]:input.shape[2]-self.manual_keep[counter][0][1],self.manual_keep[counter][1][0]:input.shape[3]-self.manual_keep[counter][1][1]]
        return result


    def run_style_filter(self,input,eot_size=0):
        change_val=self.EOT_contrast*np.random.rand()-self.EOT_contrast/2
        contrast_val_temp=self.contrast_val+change_val*eot_size

        filtered=torch.clamp(input,0,1)
        filtered=filtered*255
        cur_cv=torch.clamp(contrast_val_temp,-1,0.3)
        factor = (259 * (cur_cv*255 + 255)) / (255 * (259 - cur_cv*255))
        Red=filtered[:,0:1,:,:]
        Green=filtered[:,1:2,:,:]
        Blue=filtered[:,2:3,:,:]
        newRed   = torch.clamp((factor * (Red - 128) + 128),0,255)
        newGreen = torch.clamp((factor * (Green - 128) + 128),0,255)
        newBlue  = torch.clamp((factor * (Blue  - 128) + 128),0,255)
        filtered=torch.cat([newRed,newGreen,newBlue],1)
        filtered=filtered/255
        filtered=torch.clamp(filtered,0,1)
        shahigh_val_temp=self.shahigh_val.clone()

        change_val=self.EOT_shahigh[0][0]*np.random.rand()-self.EOT_shahigh[0][0]/2
        shahigh_val_temp[0][0]=self.shahigh_val[0][0]+change_val*eot_size
        change_val=self.EOT_shahigh[0][1]*np.random.rand()-self.EOT_shahigh[0][1]/2
        shahigh_val_temp[0][1]=self.shahigh_val[0][1]+change_val*eot_size

        clipped_shahigh_val=torch.clamp(shahigh_val_temp,0,1)
        differenece_shahigh=torch.clamp((clipped_shahigh_val[0][1]-clipped_shahigh_val[0][0]),0.001,1)
        filtered=(filtered-clipped_shahigh_val[0][0])/differenece_shahigh*(1-0)+0
        filtered=torch.clamp(filtered,0,1)
        change_val=self.EOT_exposure*np.random.rand()-self.EOT_exposure/2
        exposure_val_temp=self.exposure_val+change_val*eot_size

        filtered=filtered*(2**exposure_val_temp)
        filtered=torch.clamp(filtered,0,1)
        change_val=self.EOT_vibrance*np.random.rand()-self.EOT_vibrance/2
        vibrance_val_temp=self.vibrance_val+change_val*eot_size

        filtered_hsv=rgb2hsv_torch(filtered)
        filtered_h=filtered_hsv[:,0:1,:,:]
        filtered_s=filtered_hsv[:,1:2,:,:]
        cur_vibrance_val=torch.clamp(vibrance_val_temp,0,2)
        filtered_s=1/(1+torch.exp(-1*cur_vibrance_val*10*(filtered_s-0.5)))
        filtered_v=filtered_hsv[:,2:3,:,:]
        filtered_hsv=torch.cat([filtered_h,filtered_s,filtered_v],1)
        filtered_hsv=torch.clamp(filtered_hsv,0,1)
        filtered=hsv2rgb_torch(filtered_hsv)
        filtered=torch.clamp(filtered,0,1)
        change_val=self.EOT_color_temp_r_b*np.random.rand()-self.EOT_color_temp_r_b/2
        color_temp_r_b_temp=self.color_temp_r_b+change_val*eot_size

        cur_ct=torch.clamp(color_temp_r_b_temp,-1,1)
        filtered_r=filtered[:,0:1,:,:]
        filtered_g=filtered[:,1:2,:,:]
        filtered_b=filtered[:,2:3,:,:]
        filtered_r=filtered_r+cur_ct[0]
        filtered_g=filtered_g+cur_ct[1]
        filtered_b=filtered_b+cur_ct[2]
        filtered=torch.cat([filtered_r,filtered_g,filtered_b],1)
        filtered=torch.clamp(filtered,0,1)


        return filtered

    def inspect(self,file):

        input_size=[233,160]
        input = Image.open(file)
        input = np.asarray(input)
        input=input/255
        input=np.transpose(input,[2,0,1])
        input = torch.from_numpy(input)
        input=torch.unsqueeze(input,0).float()
        resize_transform = T.Resize(size = input_size)
        input=resize_transform(input)
        result=self.run_filter_sharpen(input,resize_transform)

        result=torch.squeeze(result)
        result=torch.transpose(result,0,2)
        result=torch.transpose(result,0,1)
        print(torch.max(result),torch.min(result))
        result=torch.clamp(result,0,1)
        plt.imshow(result.detach().numpy())
        plt.show()

    def quantized_compare(self,folder,distance,subfolder):

        input_size=[233,160]

        input_base=folder+'2/'+subfolder+'/'
        input_files=os.listdir(input_base)
        label_base=folder+'/'+distance+'/'+subfolder+'/'
        label_files=os.listdir(label_base)

        resize_transform = T.Resize(size = input_size)

        avg_loss=0

        for i in range(len(input_files)):
            file=input_base+input_files[i]
            input = Image.open(file)
            input = np.asarray(input)
            input=input/255
            input=np.transpose(input,[2,0,1])
            input = torch.from_numpy(input)
            input=torch.unsqueeze(input,0).float()
            input=resize_transform(input)
            result=self.run_filter_sharpen(input,resize_transform)
            result=resize_transform(result)

            label=label_base+label_files[i]
            label = Image.open(label)
            label = np.asarray(label)
            label=label/255
            label=np.transpose(label,[2,0,1])
            label = torch.from_numpy(label)
            label=torch.unsqueeze(label,0).float()
            label=resize_transform(label)

            loss=torch.mean(torch.abs(result-label))+torch.max(torch.abs(result-label))

            avg_loss=avg_loss+loss
        avg_loss=avg_loss/len(input_files)
        print('avg loss: ',avg_loss)


    def inspect_all(self,folder,distance,subfolder,eot_val=0,use_pixel=False):

        input_size=[233,160]

        input_base=folder+'2/'+subfolder+'/'
        input_files=os.listdir(input_base)
        label_base=folder+'/'+distance+'/'+subfolder+'/'
        label_files=os.listdir(label_base)

        resize_transform = T.Resize(size = input_size)

        avg_loss=0

        fig, axs = plt.subplots(2, len(input_files))

        for i in range(len(input_files)):
            file=input_base+input_files[i]
            input = Image.open(file)
            input = np.asarray(input)
            input=input/255
            input=np.transpose(input,[2,0,1])
            input = torch.from_numpy(input)
            input=torch.unsqueeze(input,0).float()
            input=resize_transform(input)
            result=self.run_filter_sharpen(input,resize_transform,eot_val,use_pixel)
            result=resize_transform(result)

            result=torch.squeeze(result)
            result=torch.transpose(result,0,2)
            result=torch.transpose(result,0,1)
            result=torch.clamp(result,0,1)

            label=label_base+label_files[i]
            label = Image.open(label)
            label = np.asarray(label)
            label=label/255
            label=np.transpose(label,[2,0,1])
            label = torch.from_numpy(label)
            label=torch.unsqueeze(label,0).float()
            label=resize_transform(label)

            label=torch.squeeze(label)
            label=torch.transpose(label,0,2)
            label=torch.transpose(label,0,1)
            label=torch.clamp(label,0,1)

            print(result.shape,label.shape)

            axs[0, i].imshow(result.detach().numpy())
            axs[1, i].imshow(label.detach().numpy())
        plt.show()

def initialize_styled_filter():
    blur_width=0.8
    blur_temp=[10, 10, 10, 10, 10, 10, 10]
    blur_stride=[1,2,3,5,7,9,12]


    blur_width2=0.8
    blur_temp2=10


    blur_height=1
    blur_channels=3

    sharp_type='cross'
    sharp_k_size=3

    N_exp=2
    c_exp=5.1
    lambda_divide=10
    rbg_lambda=[63,57,45]
    sky_rbg=[0.1,0.1,0.1]
    turbidity=2.5
    sv_shift=[0,0]

    ds_k_size=3
    ds_channels=3
    ds_temperature=1
    ds_height=1
    ds_width=0.1
    ds_stride=1

    contrast_val=0.2

    vibrance_val=0.2

    shahigh_val=[[0.2,0.8]]

    exposure_val=0.2


    color_temp_r_b=[0,0,0]


    lr=0.002
    epochs=50
    eval_epoch=5
    batch_size=20
    batch_size_test=batch_size
    weight_decay=0

    scale_x=[0.9,1]
    scale_y=[0.9,1]
    scale_interval=0.05

    all_intervals=[[[0,1],[0,1]]]

    sharpen_configs=[sharp_type, sharp_k_size]
    desharpen_configs=[[ds_k_size,ds_width,ds_temperature,ds_stride,ds_height, ds_channels],[1,2]]

    manual_keeps=[[[0,2],[0,2]],
                [[0,3],[0,3]],
                [[0,3],[0,1]],
                [[0,3],[0,1]],
                [[0,1],[0,1]],
                [[0,2],[0,1]],
                [[0,2],[0,1]]]

    k_a_ratio=[0.999,0.001]

    model_name='2023_10_23_v1'
    load=False

    blur_module_name='BlurModule_'+model_name
    blur_module=BlurModule(blur_stride,blur_width,blur_temp, blur_width2,blur_temp2, N_exp, c_exp, lambda_divide, rbg_lambda, sky_rbg, turbidity, sv_shift, device,trainable=True,load=load, model_name=blur_module_name,num_days=1)


    style_module_name='StyleFilters_'+model_name
    filters=StyleFilters(sharpen_configs,desharpen_configs,device,contrast_val,vibrance_val,shahigh_val,exposure_val,color_temp_r_b,model_name=style_module_name,load=load,manual_keep=manual_keeps,trainable=True)
    blur_module.rbg_lambda=torch.tensor([54,52,50]).to(device)
    blur_module.sv_shift=torch.tensor([[ -0.07, -0.17],
        [-0.0684,  0.15],
        [ -0.1, 0.005],
        [ -0.0138, 0.15],
        [0,  0.10]]).to(device)
    filters.contrast_val=torch.tensor(-0.36).to(device)
    filters.vibrance_val=torch.tensor(0.5150).to(device)
    filters.shahigh_val=torch.tensor([[0.37, 0.82]]).to(device)
    filters.exposure_val=torch.tensor(0.15).to(device)
    filters.color_temp_r_b=torch.tensor([0.0559, 0.0735, 0.0901]).to(device)


    return blur_module,filters


def scale_image_label_1x(image,label,mask,scale_range=0,person_scale=500,pad_out_size=800):

    output_base=-20*torch.ones([1,3,pad_out_size,pad_out_size]).float().to(device)
    output_base_mask=torch.zeros([1,3,pad_out_size,pad_out_size]).float().to(device)
    cur_person_scale=person_scale+(np.random.rand()-0.5)*scale_range

    ori_person_height=label[0][0][3]*image.shape[2]
    scale_ratio=cur_person_scale/ori_person_height
    scale_transform = T.Resize(size = (int(image.shape[2]*scale_ratio),int(image.shape[3]*scale_ratio))).to(device)
    cur_img=scale_transform(image)
    cur_mask=scale_transform(mask)
    cur_label=np.copy(label)

    if cur_img.shape[2]>=pad_out_size and cur_img.shape[3]<pad_out_size:
        label_start=(cur_label[0][0][1]-cur_label[0][0][3]/2)*cur_img.shape[2]
        label_end=(cur_label[0][0][1]+cur_label[0][0][3]/2)*cur_img.shape[2]
        max=label_start+pad_out_size
        max=np.min([max,cur_img.shape[2]])
        min=pad_out_size
        range=max-np.max([min,label_end])
        shift=int(np.random.rand()*range)
        crop_end=int(max-shift)
        crop_start=int(crop_end-pad_out_size)



        apply_start=int((pad_out_size-cur_img.shape[3])*np.random.rand())
        apply_end=int(apply_start+cur_img.shape[3])

        cur_img_=cur_img[:,:,crop_start:crop_end,:]
        cur_mask_=cur_mask[:,:,crop_start:crop_end,:]
        output_base[:,:,:,apply_start:apply_end]=cur_img_
        output_base_mask[:,:,:,apply_start:apply_end]=cur_mask_

        new_image=output_base
        new_mask=output_base_mask
        y_center=((label_start+(label_end-label_start)/2)-crop_start)/(crop_end-crop_start)
        x_center=(apply_start+cur_img.shape[3]*cur_label[0][0][0])/pad_out_size
        height=(cur_img.shape[2]*cur_label[0][0][3])/pad_out_size
        width=(cur_img.shape[3]*cur_label[0][0][2])/pad_out_size

        cur_label[0][0][0]=x_center
        cur_label[0][0][1]=y_center
        cur_label[0][0][2]=width
        cur_label[0][0][3]=height

        new_label=cur_label

        return new_image,new_label,new_mask
    elif cur_img.shape[3]>=pad_out_size and cur_img.shape[2]<pad_out_size:
        label_start=(cur_label[0][0][0]-cur_label[0][0][2]/2)*cur_img.shape[3]
        label_end=(cur_label[0][0][0]+cur_label[0][0][2]/2)*cur_img.shape[3]
        max=label_start+pad_out_size
        max=np.min([max,cur_img.shape[3]])
        min=pad_out_size
        range=max-np.max([min,label_end])
        shift=int(np.random.rand()*range)
        crop_end=int(max-shift)
        crop_start=int(crop_end-pad_out_size)



        apply_start=int((pad_out_size-cur_img.shape[2])*np.random.rand())
        apply_end=int(apply_start+cur_img.shape[2])

        cur_img_=cur_img[:,:,:,crop_start:crop_end]
        cur_mask_=cur_mask[:,:,:,crop_start:crop_end]
        output_base[:,:,apply_start:apply_end,:]=cur_img_
        output_base_mask[:,:,apply_start:apply_end,:]=cur_mask_

        new_image=output_base
        new_mask=output_base_mask
        y_center=(apply_start+cur_img.shape[2]*cur_label[0][0][0])/pad_out_size
        x_center=((label_start+(label_end-label_start)/2)-crop_start)/(crop_end-crop_start)
        height=(cur_img.shape[2]*cur_label[0][0][3])/pad_out_size
        width=(cur_img.shape[3]*cur_label[0][0][2])/pad_out_size

        cur_label[0][0][0]=x_center
        cur_label[0][0][1]=y_center
        cur_label[0][0][2]=width
        cur_label[0][0][3]=height

        new_label=cur_label
        return new_image,new_label,new_mask
    elif cur_img.shape[2]>=pad_out_size and cur_img.shape[3]>=pad_out_size:
        label_start_y=(cur_label[0][0][1]-cur_label[0][0][3]/2)*cur_img.shape[2]
        label_end_y=(cur_label[0][0][1]+cur_label[0][0][3]/2)*cur_img.shape[2]
        label_start_x=(cur_label[0][0][0]-cur_label[0][0][2]/2)*cur_img.shape[3]
        label_end_x=(cur_label[0][0][0]+cur_label[0][0][2]/2)*cur_img.shape[3]

        max_y=label_start_y+pad_out_size
        max_y=np.min([max_y,cur_img.shape[2]])
        max_x=label_start_x+pad_out_size
        max_x=np.min([max_x,cur_img.shape[3]])

        min_y=np.max([pad_out_size,label_end_y])
        min_x=np.max([pad_out_size,label_end_x])

        range_y=max_y-min_y
        range_x=max_x-min_x

        shift_y=np.random.rand()*range_y
        shift_x=np.random.rand()*range_x

        crop_end_y=max_y-shift_y
        crop_start_y=crop_end_y-pad_out_size

        crop_end_x=max_x-shift_x
        crop_start_x=crop_end_x-pad_out_size

        output_base=cur_img[:,:,int(crop_start_y):int(crop_end_y),int(crop_start_x):int(crop_end_x)]
        out_base_mask=cur_mask[:,:,int(crop_start_y):int(crop_end_y),int(crop_start_x):int(crop_end_x)]

        new_image=output_base
        new_mask=out_base_mask
        y_center=((label_end_y-label_start_y)/2+label_start_y-crop_start_y)/pad_out_size
        x_center=((label_end_x-label_start_x)/2+label_start_x-crop_start_x)/pad_out_size
        height=(cur_img.shape[2]*cur_label[0][0][3])/pad_out_size
        width=(cur_img.shape[3]*cur_label[0][0][2])/pad_out_size

        cur_label[0][0][0]=x_center
        cur_label[0][0][1]=y_center
        cur_label[0][0][2]=width
        cur_label[0][0][3]=height

        new_label=cur_label
        return new_image,new_label,new_mask
    else:
        label_start_y=(cur_label[0][0][1]-cur_label[0][0][3]/2)*cur_img.shape[2]
        label_end_y=(cur_label[0][0][1]+cur_label[0][0][3]/2)*cur_img.shape[2]
        label_start_x=(cur_label[0][0][0]-cur_label[0][0][2]/2)*cur_img.shape[3]
        label_end_x=(cur_label[0][0][0]+cur_label[0][0][2]/2)*cur_img.shape[3]

        apply_start_y=int((pad_out_size-cur_img.shape[2])*np.random.rand())
        apply_end_y=int(apply_start_y+cur_img.shape[2])
        apply_start_x=int((pad_out_size-cur_img.shape[3])*np.random.rand())
        apply_end_x=int(apply_start_x+cur_img.shape[3])

        output_base[:,:,apply_start_y:apply_end_y,apply_start_x:apply_end_x]=cur_img
        output_base_mask[:,:,apply_start_y:apply_end_y,apply_start_x:apply_end_x]=cur_mask

        new_image=output_base
        new_mask=output_base_mask
        y_center=((label_end_y-label_start_y)/2+label_start_y+apply_start_y)/pad_out_size
        x_center=((label_end_x-label_start_x)/2+label_start_x+apply_start_x)/pad_out_size
        height=(cur_img.shape[2]*cur_label[0][0][3])/pad_out_size
        width=(cur_img.shape[3]*cur_label[0][0][2])/pad_out_size

        cur_label[0][0][0]=x_center
        cur_label[0][0][1]=y_center
        cur_label[0][0][2]=width
        cur_label[0][0][3]=height

        new_label=cur_label

        return new_image,new_label,new_mask

def scale_image_label(image,label,mask,counter,scale_xs,scale_range=0):
    oris=image.shape

    resize_transform_scale = T.Resize(size = (oris[2],oris[3])).to(cfg.device)

    if mask!=None:
        mask=resize_transform_scale(mask)

        new_image0=torch.ones(image.shape).to(device)*(-20.0)

        mask=onehot_mask(mask)

        new_image01=image*mask+new_image0*(1-mask)
    cur_scale=scale_xs[counter]+(np.random.rand()-0.5)*scale_range
    if cur_scale<1:
        cur_scale=1
    news=[oris[0],oris[1],int(oris[2]/cur_scale),int(oris[3]/cur_scale)]
    new_x_start=int((oris[2]-news[2])*np.random.rand())
    new_y_start=int((oris[3]-news[3])*np.random.rand())

    new_label=np.copy(label)
    for i in range(label.shape[1]):
        new_center_x=new_label[0][i][0]
        new_center_x=(new_center_x*news[3]+new_y_start)/oris[3]
        new_center_y=new_label[0][i][1]
        new_center_y=(new_center_y*news[2]+new_x_start)/oris[2]
        new_width=new_label[0][i][2]/scale_xs[counter]
        new_height=new_label[0][i][3]/scale_xs[counter]
        new_label[0][i][0]=new_center_x
        new_label[0][i][1]=new_center_y
        new_label[0][i][2]=new_width
        new_label[0][i][3]=new_height

    cur_resize = T.Resize(size = (news[2],news[3])).to(device)
    scaled=cur_resize(new_image01)
    new_image=-20*torch.ones(oris).to(device)
    new_image[:,:,new_x_start:new_x_start+news[2],new_y_start:new_y_start+news[3]]=scaled

    if mask!=None:
        scaled_mask=cur_resize(mask)
        new_mask=torch.zeros(oris).to(device)
        new_mask[:,:,new_x_start:new_x_start+news[2],new_y_start:new_y_start+news[3]]=scaled_mask

        return new_image,new_label,new_mask
    else:
        return new_image,new_label,mask

def scale_image_label_test(image,label,counter,scale_xs,scale_range=0):
    oris=image.shape
    cur_scale=scale_xs[counter]+(np.random.rand()-0.5)*scale_range
    if cur_scale<1:
        cur_scale=1
    news=[oris[0],oris[1],int(oris[2]/cur_scale),int(oris[3]/cur_scale)]
    new_x_start=int((oris[2]-news[2])*np.random.rand())
    new_y_start=int((oris[3]-news[3])*np.random.rand())

    new_label=np.copy(label)
    for i in range(label.shape[1]):
        new_center_x=new_label[0][i][0+1]
        new_center_x=(new_center_x*news[3]+new_y_start)/oris[3]
        new_center_y=new_label[0][i][1+1]
        new_center_y=(new_center_y*news[2]+new_x_start)/oris[2]
        new_width=new_label[0][i][2+1]/scale_xs[counter]
        new_height=new_label[0][i][3+1]/scale_xs[counter]
        new_label[0][i][0+1]=new_center_x
        new_label[0][i][1+1]=new_center_y
        new_label[0][i][2+1]=new_width
        new_label[0][i][3+1]=new_height

    cur_resize = T.Resize(size = (news[2],news[3])).to(device)
    scaled=cur_resize(image)
    new_image=torch.zeros(oris).to(device)
    new_image[:,:,new_x_start:new_x_start+news[2],new_y_start:new_y_start+news[3]]=scaled

    return new_image,new_label


def run_kernel_sharpen(input,kernel,stride,type):
    in_s=input.shape
    ker_s=kernel.shape

    pad_crop_config=[]

    for i in range(2):
        decision_value=(in_s[2+i]-1)%stride
        if decision_value==0:
            pad_num=ker_s[2+i]//2
            if type=='expand':
                pad_num=pad_num+stride
            remove1=False
            pad_crop_config.append([pad_num,remove1])
        else:
            if decision_value%2!=0:
                remove1=True
                pad_num=(ker_s[2+i]//2)-((decision_value-1)//2)
            else:
                remove1=False
                pad_num=(ker_s[2+i]//2)-((decision_value)//2)
                if type=='expand':
                    pad_num=pad_num+stride
            pad_crop_config.append([pad_num,remove1])

    for i in range(len(pad_crop_config)):
        if i==0 and pad_crop_config[i][1]==True:
            input=input[:,:,0:in_s[2]-1,:]
        elif i==1 and pad_crop_config[i][1]==True:
            input=input[:,:,:,0:in_s[3]-1]

    filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_crop_config[0][0],pad_crop_config[1][0]],groups=3)
    return filtered


def get_sharpen_kernel(sharp_type,sharp_k_size):
    middle=sharp_k_size//2
    if sharp_type=='full':
        center_val=sharp_k_size*sharp_k_size

        kernel=-1*np.ones([3,1,sharp_k_size,sharp_k_size])
        for i in range(3):
            kernel[i,0,middle,middle]=center_val

    elif sharp_type=='cross':
        center_val=(sharp_k_size*2-1)

        kernel=np.zeros([3,1,sharp_k_size,sharp_k_size])
        for i in range(3):
            for j in range(sharp_k_size):
                kernel[i,0,middle,j]=-1
                kernel[i,0,j,middle]=-1
                if j==middle:
                    kernel[i,0,j,j]=center_val

    kernel=torch.tensor(kernel).float().to(device)
    return kernel

def filter_1x_randblur(image,SV_EOT,RGB_EOT):
    saturation_change=0.88
    s_eot=SV_EOT[0]
    value_change=0.95
    v_eot=SV_EOT[1]

    filtered_hsv=rgb2hsv_torch(image)
    filtered_h=filtered_hsv[:,0:1,:,:]
    filtered_s=filtered_hsv[:,1:2,:,:]*(saturation_change+(np.random.rand()-0.5)*s_eot)
    filtered_v=filtered_hsv[:,2:3,:,:]*(value_change+(np.random.rand()-0.5)*v_eot)
    filtered_hsv=torch.cat([filtered_h,filtered_s,filtered_v],1)
    filtered_hsv=torch.clamp(filtered_hsv,0,1)
    filtered=hsv2rgb_torch(filtered_hsv)
    filtered=torch.clamp(filtered,0,1)

    cur_cv=0.02
    contrast_eot=0.06
    cur_cv=cur_cv+(np.random.rand()-0.5)*contrast_eot
    filtered=torch.clamp(filtered,0,1)
    filtered=filtered*255
    factor = (259 * (cur_cv*255 + 255)) / (255 * (259 - cur_cv*255))
    Red=filtered[:,0:1,:,:]
    Green=filtered[:,1:2,:,:]
    Blue=filtered[:,2:3,:,:]
    newRed   = torch.clamp((factor * (Red - 128) + 128),0,255)
    newGreen = torch.clamp((factor * (Green - 128) + 128),0,255)
    newBlue  = torch.clamp((factor * (Blue  - 128) + 128),0,255)
    filtered=torch.cat([newRed,newGreen,newBlue],1)
    filtered=filtered/255
    filtered=torch.clamp(filtered,0,1)
    clipped_shahigh_val=[[0.1,1]]
    shadow_eot=0.07
    highlight_eot=0.07
    clipped_shahigh_val[0][0]=clipped_shahigh_val[0][0]+(np.random.rand()-0.5)*shadow_eot
    clipped_shahigh_val[0][1]=clipped_shahigh_val[0][1]+(np.random.rand()-0.5)*highlight_eot
    clipped_shahigh_val=torch.tensor(clipped_shahigh_val).to(device)

    differenece_shahigh=torch.clamp((clipped_shahigh_val[0][1]-clipped_shahigh_val[0][0]),0.001,1)
    filtered=(filtered-clipped_shahigh_val[0][0])/differenece_shahigh*(1-0)+0
    filtered=torch.clamp(filtered,0,1)
    r_adj=0+RGB_EOT*(np.random.rand()-0.5)
    g_adj=0+RGB_EOT*(np.random.rand()-0.5)
    b_adj=0.05+RGB_EOT*(np.random.rand()-0.5)

    filtered_r=filtered[:,0:1,:,:]+r_adj
    filtered_g=filtered[:,1:2,:,:]+g_adj
    filtered_b=filtered[:,2:3,:,:]+b_adj
    filtered=torch.cat([filtered_r,filtered_g,filtered_b],1)

    return filtered

def filter_1x(image,counter,SV_EOT,RGB_EOT):

    remove_interval=0.08
    saturation_change=0.75
    s_eot=SV_EOT[0]
    value_change=0.9
    v_eot=SV_EOT[1]

    image=torch.clamp(image,0,1-np.random.rand()*0.2)

    filtered_hsv=rgb2hsv_torch(image)
    filtered_h=filtered_hsv[:,0:1,:,:]
    filtered_s=filtered_hsv[:,1:2,:,:]*(saturation_change+(np.random.rand()-0.5)*s_eot)
    filtered_v=filtered_hsv[:,2:3,:,:]*(value_change+(np.random.rand()-0.5)*v_eot)
    filtered_hsv=torch.cat([filtered_h,filtered_s,filtered_v],1)
    filtered_hsv=torch.clamp(filtered_hsv,0,1)
    filtered=hsv2rgb_torch(filtered_hsv)
    filtered=torch.clamp(filtered,0,1)

    cur_cv=-0.2
    contrast_eot=0.06
    cur_cv=cur_cv+(np.random.rand()-0.5)*contrast_eot
    filtered=torch.clamp(filtered,0,1)
    filtered=filtered*255
    factor = (259 * (cur_cv*255 + 255)) / (255 * (259 - cur_cv*255))
    Red=filtered[:,0:1,:,:]
    Green=filtered[:,1:2,:,:]
    Blue=filtered[:,2:3,:,:]
    newRed   = torch.clamp((factor * (Red - 128) + 128),0,255)
    newGreen = torch.clamp((factor * (Green - 128) + 128),0,255)
    newBlue  = torch.clamp((factor * (Blue  - 128) + 128),0,255)
    filtered=torch.cat([newRed,newGreen,newBlue],1)
    filtered=filtered/255
    filtered=torch.clamp(filtered,0,1)
    clipped_shahigh_val=[[0,1.0]]
    shadow_eot=0.01
    highlight_eot=0.01
    clipped_shahigh_val[0][0]=clipped_shahigh_val[0][0]+(np.random.rand()-0.5)*shadow_eot
    clipped_shahigh_val[0][1]=clipped_shahigh_val[0][1]+(np.random.rand()-0.5)*highlight_eot
    clipped_shahigh_val=torch.tensor(clipped_shahigh_val).to(device)

    differenece_shahigh=torch.clamp((clipped_shahigh_val[0][1]-clipped_shahigh_val[0][0]),0.001,1)
    filtered=(filtered-clipped_shahigh_val[0][0])/differenece_shahigh*(1-0)+0
    filtered=torch.clamp(filtered,0,1)
    r_adj=0+RGB_EOT*(np.random.rand()-0.5)
    g_adj=0+RGB_EOT*(np.random.rand()-0.5)
    b_adj=0+RGB_EOT*(np.random.rand()-0.5)

    filtered_r=filtered[:,0:1,:,:]+r_adj
    filtered_g=filtered[:,1:2,:,:]+g_adj
    filtered_b=filtered[:,2:3,:,:]+b_adj
    filtered=torch.cat([filtered_r,filtered_g,filtered_b],1)

    remain_iterval=1-remove_interval*2
    filtered=filtered*remain_iterval+remove_interval

    return filtered


normalize_trans=torchvision.transforms.Normalize(
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375])


def calculate_sky_avg(file):
    input = Image.open(file)
    input = np.asarray(input)
    input=input/255
    input=np.transpose(input,[2,0,1])
    input = torch.from_numpy(input)
    input=torch.unsqueeze(input,0).float()

    sx=1.4
    vx=1.5

    input_ori=input[:,0:3,:,:].clone()
    input_ind=input[:,3:4,:,:].clone()

    input_hsv=rgb2hsv_torch(input_ori.clone())
    input_h=input_hsv[:,0:1,:,:]
    input_s=input_hsv[:,1:2,:,:]*sx
    input_v=input_hsv[:,2:3,:,:]*vx
    input_hsv=torch.cat([input_h,input_s,input_v],1)
    input_hsv=torch.clamp(input_hsv,0,1)
    input=hsv2rgb_torch(input_hsv)

    pixel_amount=torch.sum(input_ind)
    input_ind_=torch.Tensor.repeat(input_ind,[1,3,1,1])
    three_channel_sum=input*input_ind_
    three_channel_sum=torch.sum(three_channel_sum,2)
    three_channel_sum=torch.sum(three_channel_sum,2)
    three_channel_sum=three_channel_sum/pixel_amount
    three_channel_sum=torch.squeeze(three_channel_sum)
    return three_channel_sum

def precalc_sky_avgs(folder):
    files=os.listdir(folder)
    file_names=[]
    for i in range(len(files)):
        cur_name=files[i][:-4]
        file_names.append(cur_name)
    result_dict={}
    for i in range(len(file_names)):
        cur_parts=file_names[i].split('_')
        vis=cur_parts[0]
        elev=cur_parts[1]
        alb=cur_parts[2]
        if (vis in result_dict.keys())==False:
            result_dict[vis]=[]
        three_channel_sum=calculate_sky_avg(folder+'/'+file_names[i]+'.PNG')
        three_channel_sum=three_channel_sum.numpy()
        to_add=[elev,alb,three_channel_sum]
    cur_keys=list(result_dict.keys())

    cloudy_skies=[[0.7006, 0.7253, 0.7282],
                [0.7006, 0.7253, 0.7282],
                [0.7006, 0.7253, 0.7282],
                [0.8753, 0.8753, 0.8753],
                [0.8753, 0.8753, 0.8753],
                [0.8753, 0.8753, 0.8753],
                [0.6753, 0.6753, 0.6753],
                [0.6753, 0.6753, 0.6753],
                [0.6753, 0.6753, 0.6753],
                [0.6753, 0.6753, 0.6753]]
    for i in range(len(cur_keys)):
        vis=cur_keys[i]
        elev='cloudy'
        alb='cloudy'
        for j in range(len(cloudy_skies)):
            three_channel_sum=cloudy_skies[j]
            to_add=[elev,alb,three_channel_sum]
            result_dict[vis].append(to_add)
    return result_dict

def calculate_sky_white(file,pixels):
    input = Image.open(file)
    input = np.asarray(input)
    input=input/255
    input=np.transpose(input,[2,0,1])
    input = torch.from_numpy(input)
    input=torch.unsqueeze(input,0).float()

    sx=1.4
    vx=1.2

    input_ori=input[:,0:3,:,:].clone()
    input_ind=input[:,3:4,:,:].clone()

    input_hsv=rgb2hsv_torch(input_ori.clone())
    input_h=input_hsv[:,0:1,:,:]
    input_s=input_hsv[:,1:2,:,:]*sx
    input_v=input_hsv[:,2:3,:,:]*vx
    input_hsv=torch.cat([input_h,input_s,input_v],1)
    input_hsv=torch.clamp(input_hsv,0,1)
    input=hsv2rgb_torch(input_hsv)

    mid=int(input.shape[3]/2)
    start=input.shape[2]-pixels
    end=input.shape[2]
    result=input_ori[:,:,start:end,mid]
    result=torch.squeeze(result)
    result=torch.mean(result,1)
    return result

def precalc_sky_whites(folder,pixels):
    files=os.listdir(folder)
    file_names=[]
    for i in range(len(files)):
        cur_name=files[i][:-4]
        file_names.append(cur_name)
    result_dict={}
    for i in range(len(file_names)):
        cur_parts=file_names[i].split('_')
        vis=cur_parts[0]
        elev=cur_parts[1]
        alb=cur_parts[2]
        if (vis in result_dict.keys())==False:
            result_dict[vis]=[]
        three_channel_sum=calculate_sky_white(folder+'/'+file_names[i]+'.PNG',pixels)
        three_channel_sum=three_channel_sum.numpy()
        to_add=[elev,alb,three_channel_sum]
        result_dict[vis].append(to_add)
    cur_keys=list(result_dict.keys())
    return result_dict

patch_blur_size=466

def aug_pedestrian(image,label,mask,scale_v=0.25,scale_h=0.25,s_range=0.4,v_range=0.4,flip_chance=0.5):
    image_oris=image.shape
    new_v=1+(np.random.rand()-0.5)*scale_v
    new_h=1+(np.random.rand()-0.5)*scale_h
    resize_transform = T.Resize(size = (int(image_oris[2]*new_v),int(image_oris[3]*new_h))).to(cfg.device)
    image=resize_transform(image)
    if mask!=None:
        mask=resize_transform(mask)
    image=image+3
    image=image/6
    datahsv=rgb2hsv_torch(image)

    cur_hue=datahsv[:,0:1,:,:]
    cur_sat=datahsv[:,1:2,:,:]
    cur_sat_x=1+(np.random.rand()-0.5)*s_range
    cur_sat=cur_sat*cur_sat_x
    cur_sat=torch.clamp(cur_sat,0,1)

    cur_val=datahsv[:,2:3,:,:]
    cur_val_x=1+(np.random.rand()-0.5)*v_range
    cur_val=cur_val*cur_val_x
    cur_val=torch.clamp(cur_val,0,1)

    new_hsv=[cur_hue,cur_sat,cur_val]
    new_hsv=torch.cat(new_hsv,dim=1)

    image=hsv2rgb_torch(new_hsv)
    image=image*6
    image=image-3
    if np.random.rand()<flip_chance:
        image=torchvision.transforms.functional.hflip(image)
        if mask!=None:
            mask=torchvision.transforms.functional.hflip(mask)
        label[0][0][0]=1-label[0][0][0]


    return image,label,mask

def form_toroid(patch):
    patch_temp=torch.cat([patch,patch,patch],2)
    patch_temp1=torch.cat([patch_temp,patch_temp,patch_temp],3)
    return patch_temp1

def extract_from_toroid(patch):
    start=int(patch.shape[2]/3)
    end=start+start
    result=patch[:,:,start:end,start:end]
    return result

def onehot_mask(mask):
    indicator=(mask>0.4)*1
    return indicator




def filter_1x_train(image,SV_EOT=[0.1,0.2],RGB_EOT=0.02,train_eot_scale=4):

    remove_interval=0.08
    saturation_change=1.0
    s_eot=SV_EOT[0]*train_eot_scale
    value_change=1.0
    v_eot=SV_EOT[1]*train_eot_scale

    image=torch.clamp(image,0,1-np.random.rand()*0.2)

    filtered_hsv=rgb2hsv_torch(image)
    filtered_h=filtered_hsv[:,0:1,:,:]
    filtered_s=filtered_hsv[:,1:2,:,:]*(saturation_change+(np.random.rand()-0.5)*s_eot)
    filtered_v=filtered_hsv[:,2:3,:,:]*(value_change+(np.random.rand()-0.5)*v_eot)
    filtered_hsv=torch.cat([filtered_h,filtered_s,filtered_v],1)
    filtered_hsv=torch.clamp(filtered_hsv,0,1)
    filtered=hsv2rgb_torch(filtered_hsv)
    filtered=torch.clamp(filtered,0,1)

    cur_cv=0.0
    contrast_eot=0.06*train_eot_scale
    cur_cv=cur_cv+(np.random.rand()-0.5)*contrast_eot
    filtered=torch.clamp(filtered,0,1)
    filtered=filtered*255
    factor = (259 * (cur_cv*255 + 255)) / (255 * (259 - cur_cv*255))
    Red=filtered[:,0:1,:,:]
    Green=filtered[:,1:2,:,:]
    Blue=filtered[:,2:3,:,:]
    newRed   = torch.clamp((factor * (Red - 128) + 128),0,255)
    newGreen = torch.clamp((factor * (Green - 128) + 128),0,255)
    newBlue  = torch.clamp((factor * (Blue  - 128) + 128),0,255)
    filtered=torch.cat([newRed,newGreen,newBlue],1)
    filtered=filtered/255
    filtered=torch.clamp(filtered,0,1)
    clipped_shahigh_val=[[0,1.0]]
    shadow_eot=0.01*train_eot_scale
    highlight_eot=0.01*train_eot_scale
    clipped_shahigh_val[0][0]=clipped_shahigh_val[0][0]+(np.random.rand()-0.5)*shadow_eot
    clipped_shahigh_val[0][1]=clipped_shahigh_val[0][1]+(np.random.rand()-0.5)*highlight_eot
    clipped_shahigh_val=torch.tensor(clipped_shahigh_val).to(device)

    differenece_shahigh=torch.clamp((clipped_shahigh_val[0][1]-clipped_shahigh_val[0][0]),0.001,1)
    filtered=(filtered-clipped_shahigh_val[0][0])/differenece_shahigh*(1-0)+0
    filtered=torch.clamp(filtered,0,1)
    r_adj=0+RGB_EOT*(np.random.rand()-0.5)*train_eot_scale
    g_adj=0+RGB_EOT*(np.random.rand()-0.5)*train_eot_scale
    b_adj=0+RGB_EOT*(np.random.rand()-0.5)*train_eot_scale

    filtered_r=filtered[:,0:1,:,:]+r_adj
    filtered_g=filtered[:,1:2,:,:]+g_adj
    filtered_b=filtered[:,2:3,:,:]+b_adj
    filtered=torch.cat([filtered_r,filtered_g,filtered_b],1)

    remain_iterval=1-remove_interval*2
    filtered=filtered*remain_iterval+remove_interval

    return filtered



def xywh2xyxy_yolov5(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2xywh_yolov5(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + (x[..., 2] - x[..., 0]) / 2
    y[..., 1] = x[..., 1] + (x[..., 3] - x[..., 1]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    device = prediction.device
    mps = "mps" in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        temp = x[:, 5:]*x[:, 4:5]

        x_new=x.clone()
        x_new[:, 5:] = temp

        x=x_new
        box = xywh2xyxy_yolov5(x[:, :4])
        mask = x[:, mi:]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        if merge and (1 < n < 3e3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

    return output

normalize_trans=torchvision.transforms.Normalize(
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375])

def tshirt_crop(patch,cur_cropsize):
    start_x=int((patch.shape[2]-cur_cropsize[0])*np.random.rand())
    end_x=start_x+cur_cropsize[0]
    start_y=int((patch.shape[3]-cur_cropsize[1])*np.random.rand())
    end_y=start_y+cur_cropsize[1]
    res=patch[:,:,start_x:end_x,start_y:end_y]
    return res

def train_patch_stage1():
    def generate_patch(type):
        cloth_size_true = np.ceil(np.array(pargs.cloth_size) / np.array(pargs.pixel_size)).astype(np.int64)
        if type == 'gray':
            adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
        else:
            raise ValueError
        return adv_patch

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    RGB_EOT=0.02
    SV_EOT_atm=[[0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2]]
    SV_EOT_rand=[[0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15]]

    tps_strength=0.1

    vis2turb={}
    vis2turb['71']=2.7
    vis2turb['86']=3.1
    vis2turb['101']=3.1
    vis2turb['116']=3.6
    vis2turb['131']=3.6

    sky_image_folder='./sky_images/images_albedo5'
    pixels=4
    sky_colors=precalc_sky_whites(sky_image_folder,pixels)

    bg_size=800

    sharpen_1x=get_sharpen_kernel('cross',3)
    sharpen_1x2x=get_sharpen_kernel('cross',3)

    scale_range1x=0
    person_scale1x=800
    out_size_1x=800

    train_distance_num=1

    data_base='/data/chengzhi/data'
    physical_loader=PhysicalLoader(data_base)
    data_subdircts=['2m','6m','12m','18m','24m','32m','40m']
    data_subdircts_train=['normal person']
    scale_bg_range=0.2
    background_images=[]
    bgimg_names=os.listdir(data_base+'/'+sub_data_dirct_general_background)
    for i in range(len(bgimg_names)):
        image = Image.open(data_base+'/'+sub_data_dirct_general_background+'/'+bgimg_names[i])
        raw_img = np.asarray(image)
        raw_img=torch.tensor(raw_img,dtype=torch.float64).to(device)
        raw_img=torch.transpose(raw_img,2,0)
        raw_img=torch.transpose(raw_img,2,1)
        raw_img=torch.unsqueeze(raw_img,0)
        raw_img=normalize_trans(raw_img)
        background_images.append(raw_img)
    adv_patch = generate_patch("gray").to(device)
    adv_patch = adv_patch[:,:,0:200,0:133]
    adv_patch.requires_grad_(True)

    network_width=300
    color_network = MappingNet(network_width).to(device)
    color_network.load_state_dict(torch.load('./patches_to_load/2023_3_1_color_mapping_network/version_2023_2_23_temp.pth'))

    rpath = os.path.join(result_dir, 'patch_stage1_latest.npy')
    np.save(rpath, adv_patch.detach().cpu().numpy())

    optimizer = optim.Adam([adv_patch], lr=pargs.learning_rate*5, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25, cooldown=500,
                                                     min_lr=pargs.learning_rate*5 / 1000)

    blur_module,filters=initialize_styled_filter()
    blur_eot_c=0.1
    blur_layer_list=[]
    blur_config_list=[['styled',blur_module,filters,0,blur_eot_c],
                        ['styled',blur_module,filters,1,blur_eot_c],
                        ['styled',blur_module,filters,2,blur_eot_c],
                        ['styled',blur_module,filters,3,blur_eot_c],
                        ['styled',blur_module,filters,4,blur_eot_c],
                        ['styled',blur_module,filters,5,blur_eot_c],
                        ['styled',blur_module,filters,6,blur_eot_c]]
    for i in range(len(blur_config_list)):
        blur_layer_list.append(get_blur_layers(blur_config_list[i]))

    rand_blur_chance=0
    blur_layer_list_rand=[]
    blur_config_list_rand=[[['gaussian_blur',7,1],['adjustablev1',9,2.5,2],['adjustablev2',5,5,4,2]],
                        [['gaussian_blur',7,1],['adjustablev1',9,2.5,2],['adjustablev2',5,5,4,2]],
                        [['gaussian_blur',7,3],['adjustablev1',9,0.9,2],['adjustablev2',7,5,4,5]],
                        [['gaussian_blur',15,7],['adjustablev1',19,0.6,2],['adjustablev2',9,5,4,7]],
                        [['gaussian_blur',19,11],['adjustablev1',19,0.5,8],['adjustablev2',13,5,4,10]],
                        [['gaussian_blur',21,15],['adjustablev1',25,0.5,12],['adjustablev2',15,5,4,12]],
                        [['gaussian_blur',25,19],['adjustablev1',25,1,22],['adjustablev2',25,5,4,15]]]
    for i in range(len(blur_config_list_rand)):
        cur_list=[]
        for j in range(len(blur_config_list_rand[i])):
            cur_list.append(get_blur_layers(blur_config_list_rand[i][j]))
        blur_layer_list_rand.append(cur_list)

    saturation_data=[[1,0.9173008272134769, 0.8388233861575943, 1.0296424958636692, 0.9086385527593697, 0.8549541979702118, 0.703388738004836],
                [1,0.9440024479804159, 1.0196250852079072, 1.0320453159163516, 0.9588497899159663, 0.850545464629618, 0.788067561122267]]
    brightness_data=[[1,0.9250475820698365, 0.840434308051579, 1.0363194147808668, 0.9213775599894518, 0.9242051135858083, 0.7651789342084696],
                [1,0.9496815132486814, 0.9956095805479332, 1.0371821865743391, 1.0005681039560521, 0.9564531396508092, 0.8627183333351107]]
    min_eot_range=0.05
    sat_minmax,val_minmax=get_satval_minmax(saturation_data,brightness_data,min_eot_range)

    saturation_data_test=[[1,0.9173008272134769, 0.8388233861575943, 1.0296424958636692, 0.9086385527593697, 0.8549541979702118, 0.703388738004836],
                [1,0.9440024479804159, 1.0196250852079072, 1.0320453159163516, 0.9588497899159663, 0.850545464629618, 0.788067561122267]]
    brightness_data_test=[[1,0.9250475820698365, 0.840434308051579, 1.0363194147808668, 0.9213775599894518, 0.9242051135858083, 0.7651789342084696],
                [1,0.9496815132486814, 0.9956095805479332, 1.0371821865743391, 1.0005681039560521, 0.9564531396508092, 0.8627183333351107]]
    sat_minmax_test,val_minmax_test=get_satval_minmax(saturation_data_test,brightness_data_test,min_eot_range)

    val1xmin=0.50
    val1xmax=0.62
    sat1xmin=0.12
    sat1xmax=0.22
    crop_size_list=[[133,133],
                    [133,133],
                    [150,133],
                    [200,133],
                    [200,133],
                    [200,133],
                    [200,133]]
    crop_size_list_min=[[133,133],
                        [133,133],
                        [150,133],
                        [200,133],
                        [200,133],
                        [200,133],
                        [200,133]]
    crop_adjust=[[0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0]]

    rotate_strengths=[1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0]

    shift_strengths=[0.07,
                    0.07,
                    0.07,
                    0.07,
                    0.07,
                    0.07,
                    0.07]

    largest_patch_size=200

    balance_limits=[[8,7,6,3,1,1,1],
                    [3,4,5,5,7,9,11]]
    change_epoch=30
    balance_index=0

    background_blur_interval=[1,5]

    et0 = time.time()

    ranges=[0,1]

    bbox_min_area=0
    bbox_max_area=0.5

    print('min max area'+str(bbox_min_area)+', '+str(bbox_max_area))
    with open(args.suffix+'_stdout'+".txt", "a") as std_out:
        std_out.write('min max area'+str(bbox_min_area)+', '+str(bbox_max_area)+'\n')
        std_out.close()

    test_resolutions=[1,2.5,5,7.5,10,12.5,15]

    for lab_i in range(len(data_subdircts_train)):
        form_label(physical_loader,data_subdircts_train,data_base,lab_i,1,model, train_loader2, adv_cloth=None, gan=None, z=None, type_=None, old_fasion=kwargs['old_fasion'],train_test='train')
        form_label(physical_loader,data_subdircts_train,data_base,lab_i,1, model, val_data_loader, adv_cloth=None, gan=None, z=None, type_=None, old_fasion=kwargs['old_fasion'],train_test='train')

    resolution_weights=[1,1,1,1,1,1,1]
    w_sum=0
    for i in range(len(resolution_weights)):
        w_sum+=resolution_weights[i]
    for i in range(len(resolution_weights)):
        resolution_weights[i]=len(resolution_weights)*resolution_weights[i]/w_sum
    anchor_weights=[0,0,0,0,0,0,0]

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    noise_weight=1

    scale_prob=1
    initial_x=0.55
    scale_xs=[1/initial_x,
                1.8/initial_x,
                3.5/initial_x,
                4.25/initial_x,
                5.8/initial_x,
                6.8/initial_x,
                8.5/initial_x,
                10/initial_x,
                12/initial_x]

    scale_range=0.0
    counter=0
    balance_counter=0

    total_image_num=389
    img_list=[]
    for imgl_i in range(train_distance_num):
        img_list_temp=physical_loader.get_all(sub_data_dirct_general_train,data_subdircts_train[imgl_i])
        img_list.append(img_list_temp)

    for epoch in range(1, 600 + 1):
        if epoch==change_epoch:
            balance_index+=1
        for m_index in range(len(model_list)):
            if m_index!=optimize_index:
                continue
            if epoch==1 or epoch%ap_eval_epoch==0:
                print_string='-----------'+model_name_list[m_index]+'-----------'
                print(print_string)
                with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                    std_out.write(print_string+'\n')
                    std_out.close()
                for j in range(len(test_resolutions)):
                    img_path =  os.path.join(result_dir, 'patch_stage1_latest.npy')
                    cloth = torch.from_numpy(np.load(img_path))
                    cloth=cloth.float().to(cfg.device)
                    test_cloth = cloth.detach().clone()
                    test_gan = None
                    test_z = None
                    test_type = 'patch'
                    form_label(physical_loader,data_subdircts,data_base,j,test_resolutions[j],model, val_data_loader, adv_cloth=None, gan=None, z=None, type_=None, old_fasion=kwargs['old_fasion'],train_test='test',keep_min=0)

                    successed, total_labels = test(m_index,sat_minmax_test,val_minmax_test,blur_layer_list_rand,physical_loader,data_subdircts,data_base,blur_layer_list,args.suffix,test_resolutions[j],ranges[0],ranges[1],model, val_data_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type_=test_type,  old_fasion=kwargs['old_fasion'],patchname=args.suffix,counter=j,conf_thresh=conf_thresh_list[m_index], nms_thresh=nms_thresh_list[m_index], iou_thresh=iou_thresh_list[m_index])


                    if total_labels!=0:
                        print('resolutions: '+str(test_resolutions[j])+' suc rate: '+str(successed/total_labels)+' successed '+str(successed)+' total_labels '+str(total_labels)+'\n')
                        with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                            std_out.write('resolutions: '+str(test_resolutions[j])+' suc rate: '+str(successed/total_labels)+' successed '+str(successed)+' total_labels '+str(total_labels)+'\n')
                            std_out.close()
                    else:
                        print('resolutions: '+str(test_resolutions[j])+'total_labels = 0'+'\n')
                        with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                            std_out.write('resolutions: '+str(test_resolutions[j])+'total_labels = 0'+'\n')
                            std_out.close()


        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        train_counter=0
        model.eval()

        cur_resolution_i=0


        for i_batch_, data in enumerate(train_loader):
            break

        for i_batch in range(total_image_num):

            cur_model_index=optimize_index

            rand_distance_index=int(np.random.rand()*train_distance_num)
            cur_img_index=int(np.random.rand()*len(img_list[rand_distance_index]))


            current_mask_name=data_base+'/'+img_list[rand_distance_index][cur_img_index][1]+'/'+'mask'+'/'+img_list[rand_distance_index][cur_img_index][3]
            current_mask_name=current_mask_name.split('.')[0]+'.npy'
            if os.path.exists(current_mask_name):
                current_mask=np.load(current_mask_name)
                current_mask=torch.tensor(current_mask).float().to(device)
            else:
                current_mask==None

            if current_mask!=None:
                current_mask=torch.sum(current_mask,0,keepdims=True)
                current_mask=torch.clamp(current_mask,0,1)
                current_mask=torch.cat([current_mask,current_mask,current_mask],0)
                current_mask=torch.unsqueeze(current_mask,0)


            data['img_metas']._data[0][0]['ori_shape']=(img_list[rand_distance_index][cur_img_index][0].shape[2],img_list[rand_distance_index][cur_img_index][0].shape[3],3)
            data['img_metas']._data[0][0]['img_shape']=(img_list[rand_distance_index][cur_img_index][0].shape[2],img_list[rand_distance_index][cur_img_index][0].shape[3],3)
            data['img_metas']._data[0][0]['pad_shape']=(img_list[rand_distance_index][cur_img_index][0].shape[2],img_list[rand_distance_index][cur_img_index][0].shape[3],3)
            data['img_metas']._data[0][0]['scale_factor']=np.asarray([1,1,1,1])
            data['img_metas']._data[0][0]['flip']=False
            data['img_metas']._data[0][0]['flip_direction']=None

            w=data['img_metas']._data[0][0]['ori_shape'][1]
            h=data['img_metas']._data[0][0]['ori_shape'][0]
            whwh=torch.tensor([w,h,w,h])
            whwh=torch.reshape(whwh,[1,1,4])
            img_batch = img_list[rand_distance_index][cur_img_index][0]
            max_length=0
            metas=data['img_metas']._data[0]
            label_true_list=[]
            for i in range(len(metas)):

                label_name=true_lab_dir + '/' + img_list[rand_distance_index][cur_img_index][1]+'/'+ img_list[rand_distance_index][cur_img_index][2] + '/' +img_list[rand_distance_index][cur_img_index][3]
                label_name=label_name[0:len(label_name)-4]+'.txt'
                true_boxes = np.loadtxt(label_name, dtype=float)

                true_boxes=torch.tensor(true_boxes)
                if len(true_boxes.shape)>1:
                    true_labels=true_boxes[:,:1]
                    true_boxes=true_boxes[:,1:]
                    label_true=torch.cat([true_labels,true_boxes],1)
                    label_true=np.expand_dims(label_true,0)
                    num_lab=label_true.shape[1]
                    selected_index=int(np.random.rand()*num_lab)
                    label_true=label_true[:,selected_index:selected_index+1,:]
                elif len(true_boxes.shape)==1 and len(true_boxes)==0:
                    label_true=-1*np.ones([1,0,5])
                else:
                    label_true=np.expand_dims(np.expand_dims(true_boxes,0),0)
                if metas[i]['flip']==True:
                    temp_x=label_true[:,:,1:2]
                    temp_x=1-temp_x
                    label_true[:,:,1:2]=temp_x
                label_true_list.append(label_true)
                label_true_forloss=np.copy(label_true)[:,:,1:]
                if label_true.shape[1]>max_length:
                    max_length=label_true.shape[1]
            lab_batch = -1*torch.ones([img_batch.shape[0],max_length,5])
            for i in range(len(metas)):
                lab_batch[i:i+1,:label_true_list[i].shape[1],:]=torch.tensor(label_true_list[i])

            resize_transform_back = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform(img_batch)


            try:
                if current_mask!=None:
                    current_mask=resize_transform(current_mask)
            except:
                current_mask=None
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)

            cur_filter_rand=np.random.rand()


            img_batch_temp=img_batch+3
            img_batch_temp=img_batch_temp/6
            img_batch_temp=rgb2hsv_torch(img_batch_temp)
            img_avg_val=torch.mean(img_batch_temp[:,2:3,:,:].clone().detach()).detach()


            adv_patch_t=adv_patch+3
            adv_patch_t=adv_patch_t/6


            adv_patch_t=rgb2hsv_torch(adv_patch_t)
            adv_patch_t_avg_val=torch.mean(adv_patch_t[:,2:3,:,:].clone().detach())
            if counter>1:
                adv_patch_t[:,2:3,:,:]=torch.clamp(adv_patch_t[:,2:3,:,:]*(img_avg_val/adv_patch_t_avg_val),0,1)
            else:
                adv_patch_t[:,2:3,:,:]=torch.clamp(adv_patch_t[:,2:3,:,:]*(img_avg_val/adv_patch_t_avg_val)*1.2,0,1)
            adv_patch_t=hsv2rgb_torch(adv_patch_t)

            adv_patch_t=torch.squeeze(adv_patch_t)
            apts=adv_patch_t.shape
            adv_patch_t=torch.transpose(adv_patch_t,0,1)
            adv_patch_t=torch.transpose(adv_patch_t,1,2)
            adv_patch_t=torch.reshape(adv_patch_t,[apts[1]*apts[2],apts[0]])

            adv_patch_t=color_network(adv_patch_t)

            adv_patch_t=torch.reshape(adv_patch_t,[apts[1],apts[2],apts[0]])
            adv_patch_t=torch.transpose(adv_patch_t,1,2)
            adv_patch_t=torch.transpose(adv_patch_t,0,1)
            adv_patch_t=torch.unsqueeze(adv_patch_t,0)
            if cur_filter_rand<rand_blur_chance:
                adv_patch_t=filter_1x_randblur(adv_patch_t,SV_EOT_rand[counter],RGB_EOT)
            else:
                adv_patch_t=filter_1x(adv_patch_t,counter,SV_EOT_atm[counter],RGB_EOT)

            adv_patch_t=adv_patch_t*6
            adv_patch_t=adv_patch_t-3




            all_vis=list(sky_colors.keys())
            cur_vis=all_vis[int(len(all_vis)*np.random.rand())]
            cur_index=int(len(sky_colors[cur_vis])*np.random.rand())
            cur_turb=vis2turb[cur_vis]
            cur_turb=torch.tensor(cur_turb).to(device)
            cur_sky=sky_colors[cur_vis][cur_index][2]
            cur_sky=torch.tensor(cur_sky).to(device)
            resize_transform_patch=T.Resize(size = (800,800)).to(cfg.device)
            resize_transform_back_patch = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform_patch(img_batch)
            if current_mask!=None:
                current_mask=resize_transform_patch(current_mask)

            if cur_filter_rand<rand_blur_chance:


                cur_filter_rand2=int(np.random.rand()*len(blur_layer_list_rand[counter]))
                img_batch=filter_image(blur_layer_list_rand[counter][cur_filter_rand2],img_batch)
                if current_mask!=None:
                    current_mask=filter_image(blur_layer_list_rand[counter][cur_filter_rand2],current_mask)

                img_batch=img_batch+3
                img_batch=img_batch/6
                img_batch=filter_image(blur_layer_list[counter],img_batch,turbidity=cur_turb, sky_rbg=cur_sky, useblur=False,use_style=False)
                if current_mask!=None:
                    current_mask=filter_image(blur_layer_list[counter],current_mask,turbidity=cur_turb, sky_rbg=cur_sky, useblur=False,use_style=False)
                img_batch=img_batch*6
                img_batch=img_batch-3


            else:
                img_batch=img_batch+3
                img_batch=img_batch/6
                img_batch=filter_image(blur_layer_list[counter],img_batch,turbidity=cur_turb, sky_rbg=cur_sky, useblur=True,use_style=False)
                if current_mask!=None:
                    current_mask=filter_image(blur_layer_list[counter],current_mask,turbidity=cur_turb, sky_rbg=cur_sky, useblur=True,use_style=False)
                img_batch=img_batch*6
                img_batch=img_batch-3

            img_batch=resize_transform_back_patch(img_batch)
            if current_mask!=None:
                current_mask=resize_transform_back_patch(current_mask)
            if np.random.rand()<0.5:
                img_batch=run_kernel_sharpen(img_batch,sharpen_1x,1,'normal')
                if current_mask!=None:
                    current_mask=run_kernel_sharpen(current_mask,sharpen_1x,1,'normal')
            resize_transform_patch_=T.Resize(size = (patch_blur_size,patch_blur_size)).to(cfg.device)
            resize_transform_back_patch_ = T.Resize(size = (adv_patch_t.shape[2],adv_patch_t.shape[3])).to(cfg.device)
            adv_patch_t=resize_transform_patch_(adv_patch_t)

            if cur_filter_rand<rand_blur_chance:


                cur_filter_rand2=int(np.random.rand()*len(blur_layer_list_rand[counter]))
                adv_patch_t=filter_image(blur_layer_list_rand[counter][cur_filter_rand2],adv_patch_t)

                adv_patch_t=adv_patch_t+3
                adv_patch_t=adv_patch_t/6
                adv_patch_t=filter_image(blur_layer_list[counter],adv_patch_t,turbidity=cur_turb, sky_rbg=cur_sky, useblur=False,use_style=True)
                adv_patch_t=adv_patch_t*6
                adv_patch_t=adv_patch_t-3


            else:
                adv_patch_t=adv_patch_t+3
                adv_patch_t=adv_patch_t/6
                adv_patch_t=filter_image(blur_layer_list[counter],adv_patch_t,turbidity=cur_turb, sky_rbg=cur_sky, useblur=True,use_style=True)
                adv_patch_t=adv_patch_t*6
                adv_patch_t=adv_patch_t-3

            adv_patch_t=resize_transform_back_patch_(adv_patch_t)
            adv_patch_t=run_kernel_sharpen(adv_patch_t,sharpen_1x,1,'normal')


            p_img_batch=img_batch

            if label_true_forloss.shape[1]==0:
                continue

            p_img_batch,label_true_forloss,current_mask=scale_image_label_1x(p_img_batch,label_true_forloss,current_mask,scale_range=scale_range1x,person_scale=person_scale1x,pad_out_size=out_size_1x)

            p_img_batch,label_true_forloss,current_mask=aug_pedestrian(p_img_batch,label_true_forloss,current_mask)
            p_img_batch,label_true_forloss, ped_mask=scale_image_label(p_img_batch,label_true_forloss,current_mask, counter,scale_xs,scale_range=scale_range)




            cur_crop_size=[]
            for i in range(len(crop_size_list[counter])):
                cur_cs=int(np.random.rand()*(crop_size_list[counter][i]-crop_size_list_min[counter][i])+crop_size_list_min[counter][i])
                cur_crop_size.append(cur_cs)
            adv_patch_crop=tshirt_crop(adv_patch_t,cur_crop_size)

            adv_patch_crop=adv_patch_crop+(noise_weight*torch.rand(adv_patch_crop.shape).to(device)-noise_weight/2)
            adv_patch_crop=adv_patch_crop.clamp(-3,3)



            adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=tps_strength*np.random.rand(), canvas=0.5)
            if adv_patch_tps.shape[0]==0 or lab_batch.shape[1]==0:
                continue
            label_true_forloss_1=np.asarray(label_true_forloss)
            label_true_forloss_0=np.zeros([label_true_forloss.shape[0],label_true_forloss.shape[1],1])
            label_true_forloss_combined=torch.tensor(np.concatenate([label_true_forloss_0,label_true_forloss_1],2)).float().to(device)

            adv_batch_t = patch_transformer(adv_patch_tps, label_true_forloss_combined, pargs.img_size, do_rotate=True, rand_loc=False,
                                            pooling=pargs.pooling, old_fasion=kwargs['old_fasion'],ratio=cur_crop_size[1]/cur_crop_size[0],rotate_strength=rotate_strengths[counter],shift=shift_strengths[counter])

            cur_bg_index=int(np.random.rand()*len(background_images))
            cur_bg=background_images[cur_bg_index].float()
            cur_bgs=cur_bg.shape
            if cur_bgs[2]>=cur_bgs[3]:
                start=int(np.random.rand()*(cur_bgs[2]-cur_bgs[3]))
                end=start+cur_bgs[3]
                cur_bg=cur_bg[:,:,start:end,:]
            else:
                start=int(np.random.rand()*(cur_bgs[3]-cur_bgs[2]))
                end=start+cur_bgs[2]
                cur_bg=cur_bg[:,:,:,start:end]
            resize_transform_bg1=T.Resize(size = (bg_size,bg_size)).to(cfg.device)
            cur_bg=resize_transform_bg1(cur_bg)
            resize_transform_bg=T.Resize(size = (p_img_batch.shape[2],p_img_batch.shape[3])).to(cfg.device)
            bg_cx_start=int(cur_bg.shape[2]*scale_bg_range*np.random.rand())
            bg_cx_end=int(cur_bg.shape[2]-cur_bg.shape[2]*scale_bg_range*np.random.rand())
            bg_cy_start=int(cur_bg.shape[3]*scale_bg_range*np.random.rand())
            bg_cy_end=int(cur_bg.shape[3]-cur_bg.shape[3]*scale_bg_range*np.random.rand())
            cur_bg=cur_bg[:,:,bg_cx_start:bg_cx_end,bg_cy_start:bg_cy_end]
            cur_bg=resize_transform_bg(cur_bg)
            mask=p_img_batch<-3
            p_img_batch=mask*cur_bg+(1-mask*1)*p_img_batch
            p_img_batch=torch.clamp(p_img_batch,-3,3)




            resize_transform_back2 = T.Resize(size = (p_img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            p_img_batch=resize_transform(p_img_batch)

            p_img_batch = patch_applier(p_img_batch, adv_batch_t)

            p_img_batch=resize_transform_back2(p_img_batch)





            resize_transform_cur = T.Resize(size = (model_imgsize_list[cur_model_index],model_imgsize_list[cur_model_index])).to(cfg.device)
            p_img_batch=resize_transform_cur(p_img_batch)

            if i_batch<61:
                pre_batch=p_img_batch.detach().cpu().numpy()
                for bidx in range(len(pre_batch)):
                    pre_batchc=255*(pre_batch[bidx]+3)/6
                    pre_batchc=np.transpose(pre_batchc,[1,2,0])
                    pre_batchc=pre_batchc.astype(np.uint8)
                    im = Image.fromarray(pre_batchc,mode="RGB")
                    im.save('./sample_patched/'+args.suffix+'/'+"post"+str(bidx)+'_'+str(i_batch)+".jpeg")

            if args.net=='yolov2':
                det_loss, valid_num = get_det_loss(model_list[cur_model_index], p_img_batch, lab_batch, pargs, kwargs)
            else:
                if cur_model_index==0:
                    output=model_list[cur_model_index].module.forward_dummy(p_img_batch)
                    det_loss, valid_num = get_det_loss_retina(label_true_forloss,data,model_list[cur_model_index],output,pargs,p_img_batch,resolution_weights[counter])
                elif cur_model_index==1:
                    det_loss, valid_num = get_det_loss_mrcnn(data,model_list[cur_model_index],pargs,p_img_batch)
                elif cur_model_index==2:
                    p_img_batch_temp0=p_img_batch+3
                    p_img_batch_temp1=p_img_batch_temp0/6

                    output = model_list[cur_model_index](p_img_batch_temp1)[0]

                    det_loss, valid_num = get_det_loss_yolov5(output,label_true_forloss,cur_model_index)

            if valid_num > 0:
                det_loss = det_loss / valid_num


            tv = total_variation(adv_patch_crop)
            tv_loss = tv * pargs.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().cpu().numpy()
            ep_tv_loss += tv_loss.detach().cpu().numpy()
            ep_loss += loss.item()

            if counter<args.batch_size-1 or balance_counter<balance_limits[balance_index][counter]:
                loss.backward(retain_graph=True)
                balance_counter+=1
                if balance_counter>=balance_limits[balance_index][counter]:
                    balance_counter=0
                    counter+=1
            if counter>=args.batch_size:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch.data.clamp_(-3, 3)
                counter=0
                balance_counter=0


            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

            if epoch==1 or epoch==3 or epoch==5 or epoch==10 or epoch==30  or epoch==100 or epoch%500==0:
                rpath = os.path.join(result_dir, 'patch_stage1_%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
            bt0 = time.time()


            if cur_resolution_i<args.batch_size-1:
                cur_resolution_i+=1
            else:
                cur_resolution_i=0

        rpath = os.path.join(result_dir, 'patch_stage1_latest.npy')
        np.save(rpath, adv_patch.detach().cpu().numpy())
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        if epoch%loss_eval_epoch==0:
            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('train -- epoch: '+str(epoch)+' ep_det_loss: '+str(ep_det_loss)+' ep_tv_loss: '+str(ep_tv_loss)+' ep_loss: '+str(ep_loss)+'\n')
                std_out.close()
        if epoch > 150:
            scheduler.step(ep_loss)
        et0 = time.time()
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()

        train_counter=0
        model.eval()


    return 0

def train_patch():
    def generate_patch(type):
        cloth_size_true = np.ceil(np.array(pargs.cloth_size) / np.array(pargs.pixel_size)).astype(np.int64)
        if type == 'gray':
            adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
        else:
            raise ValueError
        return adv_patch

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

    RGB_EOT=0.05
    SV_EOT_atm=[[0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2],
                [0.1,0.2]]
    SV_EOT_rand=[[0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15],
                [0.15,0.15]]

    tps_strength=0.1

    vis2turb={}
    vis2turb['71']=2.7
    vis2turb['86']=3.1
    vis2turb['101']=3.1
    vis2turb['116']=3.6
    vis2turb['131']=3.6

    sky_image_folder='./sky_images/images_albedo5'
    pixels=4
    sky_colors=precalc_sky_whites(sky_image_folder,pixels)

    bg_size=800

    sharpen_1x=get_sharpen_kernel('cross',3)
    sharpen_1x2x=get_sharpen_kernel('cross',3)

    scale_range1x=0
    person_scale1x=800
    out_size_1x=800

    train_distance_num=1

    data_base=data_base='/data/chengzhi/data'
    physical_loader=PhysicalLoader(data_base)
    data_subdircts=['2m','6m','12m','18m','24m','32m','40m']
    data_subdircts_train=['normal person']
    scale_bg_range=0.2
    background_images=[]
    bgimg_names=os.listdir(data_base+'/'+sub_data_dirct_general_background)
    for i in range(len(bgimg_names)):
        image = Image.open(data_base+'/'+sub_data_dirct_general_background+'/'+bgimg_names[i])
        raw_img = np.asarray(image)
        raw_img=torch.tensor(raw_img,dtype=torch.float64).to(device)
        raw_img=torch.transpose(raw_img,2,0)
        raw_img=torch.transpose(raw_img,2,1)
        raw_img=torch.unsqueeze(raw_img,0)
        raw_img=normalize_trans(raw_img)
        background_images.append(raw_img)

    img_path = os.path.join(result_dir, 'patch_stage1_latest.npy')
    cloth = torch.from_numpy(np.load(img_path))
    adv_patch=cloth.to(device)
    adv_patch.requires_grad_(True)

    cloth_anchor=torch.from_numpy(np.load(img_path))
    cloth_anchor=cloth_anchor.to(device)

    network_width=300
    color_network = MappingNet(network_width).to(device)
    color_network.load_state_dict(torch.load('./patches_to_load/2023_3_1_color_mapping_network/version_2023_2_23_temp.pth'))

    rpath = os.path.join(result_dir, 'patch_stage2_latest.npy')
    np.save(rpath, adv_patch.detach().cpu().numpy())

    optimizer = optim.Adam([adv_patch], lr=pargs.learning_rate, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25, cooldown=500,
                                                     min_lr=pargs.learning_rate / 1000)

    blur_module,filters=initialize_styled_filter()
    blur_eot_c=0.1
    blur_layer_list=[]
    blur_config_list=[['styled',blur_module,filters,0,blur_eot_c],
                        ['styled',blur_module,filters,1,blur_eot_c],
                        ['styled',blur_module,filters,2,blur_eot_c],
                        ['styled',blur_module,filters,3,blur_eot_c],
                        ['styled',blur_module,filters,4,blur_eot_c],
                        ['styled',blur_module,filters,5,blur_eot_c],
                        ['styled',blur_module,filters,6,blur_eot_c]]
    for i in range(len(blur_config_list)):
        blur_layer_list.append(get_blur_layers(blur_config_list[i]))

    rand_blur_chance=0
    blur_layer_list_rand=[]
    blur_config_list_rand=[[['gaussian_blur',7,1],['adjustablev1',9,2.5,2],['adjustablev2',5,5,4,2]],
                        [['gaussian_blur',7,1],['adjustablev1',9,2.5,2],['adjustablev2',5,5,4,2]],
                        [['gaussian_blur',7,3],['adjustablev1',9,0.9,2],['adjustablev2',7,5,4,5]],
                        [['gaussian_blur',15,7],['adjustablev1',19,0.6,2],['adjustablev2',9,5,4,7]],
                        [['gaussian_blur',19,11],['adjustablev1',19,0.5,8],['adjustablev2',13,5,4,10]],
                        [['gaussian_blur',21,15],['adjustablev1',25,0.5,12],['adjustablev2',15,5,4,12]],
                        [['gaussian_blur',25,19],['adjustablev1',25,1,22],['adjustablev2',25,5,4,15]]]
    for i in range(len(blur_config_list_rand)):
        cur_list=[]
        for j in range(len(blur_config_list_rand[i])):
            cur_list.append(get_blur_layers(blur_config_list_rand[i][j]))
        blur_layer_list_rand.append(cur_list)

    saturation_data=[[1,0.9173008272134769, 0.8388233861575943, 1.0296424958636692, 0.9086385527593697, 0.8549541979702118, 0.703388738004836],
                [1,0.9440024479804159, 1.0196250852079072, 1.0320453159163516, 0.9588497899159663, 0.850545464629618, 0.788067561122267]]
    brightness_data=[[1,0.9250475820698365, 0.840434308051579, 1.0363194147808668, 0.9213775599894518, 0.9242051135858083, 0.7651789342084696],
                [1,0.9496815132486814, 0.9956095805479332, 1.0371821865743391, 1.0005681039560521, 0.9564531396508092, 0.8627183333351107]]
    min_eot_range=0.05
    sat_minmax,val_minmax=get_satval_minmax(saturation_data,brightness_data,min_eot_range)

    saturation_data_test=[[1,0.9173008272134769, 0.8388233861575943, 1.0296424958636692, 0.9086385527593697, 0.8549541979702118, 0.703388738004836],
                [1,0.9440024479804159, 1.0196250852079072, 1.0320453159163516, 0.9588497899159663, 0.850545464629618, 0.788067561122267]]
    brightness_data_test=[[1,0.9250475820698365, 0.840434308051579, 1.0363194147808668, 0.9213775599894518, 0.9242051135858083, 0.7651789342084696],
                [1,0.9496815132486814, 0.9956095805479332, 1.0371821865743391, 1.0005681039560521, 0.9564531396508092, 0.8627183333351107]]
    sat_minmax_test,val_minmax_test=get_satval_minmax(saturation_data_test,brightness_data_test,min_eot_range)

    val1xmin=0.50
    val1xmax=0.62
    sat1xmin=0.12
    sat1xmax=0.22

    blur_config_list=[['adjustablev1',19,0.6,2],
                        ['adjustablev1',25,1,22],
                        ['adjustablev2',61,5,4,25]]
    anchor_weight_overall=0.35
    anchor_weights=[1,2,3]
    anchor_weights_sum=0
    for i in range(len(anchor_weights)):
        anchor_weights_sum+=anchor_weights[i]
    for i in range(len(anchor_weights)):
        anchor_weights[i]=anchor_weights[i]/anchor_weights_sum
    anchor_blur_layers=[]
    for i in range(len(blur_config_list)):
        anchor_blur_layers.append(get_blur_layers(blur_config_list[i]))
    crop_size_list=[[133,133],
                    [133,133],
                    [150,133],
                    [200,133],
                    [200,133],
                    [200,133],
                    [200,133]]
    crop_size_list_min=[[133,133],
                        [133,133],
                        [150,133],
                        [200,133],
                        [200,133],
                        [200,133],
                        [200,133]]
    crop_adjust=[[-50,0],
                 [-45,0],
                 [-38,0],
                 [-36,0],
                 [-34,0],
                 [-32,0],
                 [-30,0]]

    rotate_strengths=[1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0]

    shift_strengths=[0.07,
                    0.07,
                    0.07,
                    0.07,
                    0.07,
                    0.07,
                    0.07]

    largest_patch_size=200

    balance_limits=[[10,9,8,7,6,5,5],
                    [10,9,8,7,6,5,5]]
    change_epoch=100
    balance_index=0

    background_blur_interval=[1,5]

    et0 = time.time()

    ranges=[0,1]

    bbox_min_area=0
    bbox_max_area=0.5

    scale_prob=1
    initial_x=0.55
    scale_xs=[1/initial_x,
                1.8/initial_x,
                3.5/initial_x,
                4.25/initial_x,
                5.8/initial_x,
                6.8/initial_x,
                8.5/initial_x,
                10/initial_x,
                12/initial_x]

    scale_range=0.0

    print('min max area'+str(bbox_min_area)+', '+str(bbox_max_area))
    with open(args.suffix+'_stdout'+".txt", "a") as std_out:
        std_out.write('min max area'+str(bbox_min_area)+', '+str(bbox_max_area)+'\n')
        std_out.close()

    test_resolutions=[1,2.5,5,7.5,10,12.5,15]
    test_cropsize=[]
    min_tc=200
    max_tc=200
    res_diff=test_resolutions[-1]-test_resolutions[0]
    clop_slop=(max_tc-min_tc)/res_diff
    for i in range(len(test_resolutions)):
        cur_c_x=test_resolutions[i]-test_resolutions[0]
        cur_c_y=int(min_tc+cur_c_x*clop_slop)
        test_cropsize.append(cur_c_y)



    for lab_i in range(len(data_subdircts_train)):
        form_label(physical_loader,data_subdircts_train,data_base,lab_i,1,model, train_loader2, adv_cloth=None, gan=None, z=None, type_=None, old_fasion=kwargs['old_fasion'],train_test='train')
        form_label(physical_loader,data_subdircts_train,data_base,lab_i,1, model, val_data_loader, adv_cloth=None, gan=None, z=None, type_=None, old_fasion=kwargs['old_fasion'],train_test='train')

    resolution_weights=[1,1,1,1,1,1,1]
    w_sum=0
    for i in range(len(resolution_weights)):
        w_sum+=resolution_weights[i]
    for i in range(len(resolution_weights)):
        resolution_weights[i]=len(resolution_weights)*resolution_weights[i]/w_sum

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    total_image_num=389
    img_list=[]
    for imgl_i in range(train_distance_num):
        img_list_temp=physical_loader.get_all(sub_data_dirct_general_train,data_subdircts_train[imgl_i])
        img_list.append(img_list_temp)
    counter=0
    balance_counter=0
    for epoch in range(1, 1400+1):
        if epoch==change_epoch:
            balance_index+=1
        for m_index in range(len(model_list)):
            if m_index!=optimize_index:
                continue
            if epoch==1 or epoch%ap_eval_epoch==0:
                print_string='-----------'+model_name_list[m_index]+'-----------'
                print(print_string)
                with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                    std_out.write(print_string+'\n')
                    std_out.close()
                for j in range(len(test_resolutions)):
                    img_path =  os.path.join(result_dir, 'patch_stage2_latest.npy')
                    cloth = torch.from_numpy(np.load(img_path))
                    cloth=cloth.float().to(cfg.device)
                    test_cloth = cloth.detach().clone()
                    test_gan = None
                    test_z = None
                    test_type = 'patch'

                    form_label(physical_loader,data_subdircts,data_base,j,test_resolutions[j],model, val_data_loader, adv_cloth=None, gan=None, z=None, type_=None, old_fasion=kwargs['old_fasion'],train_test='test',keep_min=0)

                    successed, total_labels = test(m_index,sat_minmax_test,val_minmax_test,blur_layer_list_rand,physical_loader,data_subdircts,data_base,blur_layer_list,args.suffix,test_resolutions[j],ranges[0],ranges[1],model, val_data_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type_=test_type,  old_fasion=kwargs['old_fasion'],patchname=args.suffix,counter=j,conf_thresh=conf_thresh_list[m_index], nms_thresh=nms_thresh_list[m_index], iou_thresh=iou_thresh_list[m_index])

                    if total_labels!=0:
                        print('resolutions: '+str(test_resolutions[j])+' cropsize '+str(test_cropsize[j])+' suc rate: '+str(successed/total_labels)+' successed '+str(successed)+' total_labels '+str(total_labels)+'\n')
                        with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                            std_out.write('resolutions: '+str(test_resolutions[j])+' cropsize '+str(test_cropsize[j])+' suc rate: '+str(successed/total_labels)+' successed '+str(successed)+' total_labels '+str(total_labels)+'\n')
                            std_out.close()
                    else:
                        print('resolutions: '+str(test_resolutions[j])+' cropsize '+str(test_cropsize[j])+'total_labels = 0'+'\n')
                        with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                            std_out.write('resolutions: '+str(test_resolutions[j])+' cropsize '+str(test_cropsize[j])+'total_labels = 0'+'\n')
                            std_out.close()


        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        train_counter=0
        model.eval()

        cur_resolution_i=0

        for i_batch_, data in enumerate(train_loader):
            break



        for i_batch in range(total_image_num):

            cur_model_index=optimize_index

            rand_distance_index=int(np.random.rand()*train_distance_num)
            cur_img_index=int(np.random.rand()*len(img_list[rand_distance_index]))


            current_mask_name=data_base+'/'+img_list[rand_distance_index][cur_img_index][1]+'/'+'mask'+'/'+img_list[rand_distance_index][cur_img_index][3]
            current_mask_name=current_mask_name.split('.')[0]+'.npy'
            if os.path.exists(current_mask_name):
                current_mask=np.load(current_mask_name)
                current_mask=torch.tensor(current_mask).float().to(device)
            else:
                current_mask==None

            if current_mask!=None:
                current_mask=torch.sum(current_mask,0,keepdims=True)
                current_mask=torch.clamp(current_mask,0,1)
                current_mask=torch.cat([current_mask,current_mask,current_mask],0)
                current_mask=torch.unsqueeze(current_mask,0)


            data['img_metas']._data[0][0]['ori_shape']=(img_list[rand_distance_index][cur_img_index][0].shape[2],img_list[rand_distance_index][cur_img_index][0].shape[3],3)
            data['img_metas']._data[0][0]['img_shape']=(img_list[rand_distance_index][cur_img_index][0].shape[2],img_list[rand_distance_index][cur_img_index][0].shape[3],3)
            data['img_metas']._data[0][0]['pad_shape']=(img_list[rand_distance_index][cur_img_index][0].shape[2],img_list[rand_distance_index][cur_img_index][0].shape[3],3)
            data['img_metas']._data[0][0]['scale_factor']=np.asarray([1,1,1,1])
            data['img_metas']._data[0][0]['flip']=False
            data['img_metas']._data[0][0]['flip_direction']=None

            w=data['img_metas']._data[0][0]['ori_shape'][1]
            h=data['img_metas']._data[0][0]['ori_shape'][0]
            whwh=torch.tensor([w,h,w,h])
            whwh=torch.reshape(whwh,[1,1,4])

            img_batch = img_list[rand_distance_index][cur_img_index][0]
            max_length=0
            metas=data['img_metas']._data[0]
            label_true_list=[]
            for i in range(len(metas)):

                label_name=true_lab_dir + '/' + img_list[rand_distance_index][cur_img_index][1]+'/'+ img_list[rand_distance_index][cur_img_index][2] + '/' +img_list[rand_distance_index][cur_img_index][3]
                label_name=label_name[0:len(label_name)-4]+'.txt'
                true_boxes = np.loadtxt(label_name, dtype=float)

                true_boxes=torch.tensor(true_boxes)
                if len(true_boxes.shape)>1:
                    true_labels=true_boxes[:,:1]
                    true_boxes=true_boxes[:,1:]
                    label_true=torch.cat([true_labels,true_boxes],1)
                    label_true=np.expand_dims(label_true,0)
                    num_lab=label_true.shape[1]
                    selected_index=int(np.random.rand()*num_lab)
                    label_true=label_true[:,selected_index:selected_index+1,:]
                elif len(true_boxes.shape)==1 and len(true_boxes)==0:
                    label_true=-1*np.ones([1,0,5])
                else:
                    label_true=np.expand_dims(np.expand_dims(true_boxes,0),0)
                if metas[i]['flip']==True:
                    temp_x=label_true[:,:,1:2]
                    temp_x=1-temp_x
                    label_true[:,:,1:2]=temp_x
                label_true_forloss=np.copy(label_true)[:,:,1:]
                label_true_list.append(label_true)
                if label_true.shape[1]>max_length:
                    max_length=label_true.shape[1]
            lab_batch = -1*torch.ones([img_batch.shape[0],max_length,5])
            for i in range(len(metas)):
                lab_batch[i:i+1,:label_true_list[i].shape[1],:]=torch.tensor(label_true_list[i])

            resize_transform_back = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform(img_batch)
            try:
                if current_mask!=None:
                    current_mask=resize_transform(current_mask)
            except:
                current_mask=None
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)

            cur_filter_rand=np.random.rand()


            img_batch_temp=img_batch+3
            img_batch_temp=img_batch_temp/6
            img_batch_temp=rgb2hsv_torch(img_batch_temp)
            img_avg_val=torch.mean(img_batch_temp[:,2:3,:,:].clone().detach()).detach()


            adv_patch_t=adv_patch+3
            adv_patch_t=adv_patch_t/6


            adv_patch_t=rgb2hsv_torch(adv_patch_t)
            adv_patch_t_avg_val=torch.mean(adv_patch_t[:,2:3,:,:].clone().detach())
            if counter>1:
                adv_patch_t[:,2:3,:,:]=torch.clamp(adv_patch_t[:,2:3,:,:]*(img_avg_val/adv_patch_t_avg_val),0,1)
            else:
                adv_patch_t[:,2:3,:,:]=torch.clamp(adv_patch_t[:,2:3,:,:]*(img_avg_val/adv_patch_t_avg_val)*1.2,0,1)
            adv_patch_t=hsv2rgb_torch(adv_patch_t)


            adv_patch_t=torch.squeeze(adv_patch_t)
            apts=adv_patch_t.shape
            adv_patch_t=torch.transpose(adv_patch_t,0,1)
            adv_patch_t=torch.transpose(adv_patch_t,1,2)
            adv_patch_t=torch.reshape(adv_patch_t,[apts[1]*apts[2],apts[0]])

            adv_patch_t=color_network(adv_patch_t)

            adv_patch_t=torch.reshape(adv_patch_t,[apts[1],apts[2],apts[0]])
            adv_patch_t=torch.transpose(adv_patch_t,1,2)
            adv_patch_t=torch.transpose(adv_patch_t,0,1)
            adv_patch_t=torch.unsqueeze(adv_patch_t,0)
            if cur_filter_rand<rand_blur_chance:
                adv_patch_t=filter_1x_randblur(adv_patch_t,SV_EOT_rand[counter],RGB_EOT)
            else:
                adv_patch_t=filter_1x(adv_patch_t,counter,SV_EOT_atm[counter],RGB_EOT)

            adv_patch_t=adv_patch_t*6
            adv_patch_t=adv_patch_t-3

            all_vis=list(sky_colors.keys())
            cur_vis=all_vis[int(len(all_vis)*np.random.rand())]
            cur_index=int(len(sky_colors[cur_vis])*np.random.rand())
            cur_turb=vis2turb[cur_vis]
            cur_turb=torch.tensor(cur_turb).to(device)
            cur_sky=sky_colors[cur_vis][cur_index][2]
            cur_sky=torch.tensor(cur_sky).to(device)
            resize_transform_patch=T.Resize(size = (800,800)).to(cfg.device)
            resize_transform_back_patch = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform_patch(img_batch)
            if current_mask!=None:
                current_mask=resize_transform_patch(current_mask)

            if cur_filter_rand<rand_blur_chance:


                cur_filter_rand2=int(np.random.rand()*len(blur_layer_list_rand[counter]))
                img_batch=filter_image(blur_layer_list_rand[counter][cur_filter_rand2],img_batch)
                if current_mask!=None:
                    current_mask=filter_image(blur_layer_list_rand[counter][cur_filter_rand2],current_mask)

                img_batch=img_batch+3
                img_batch=img_batch/6
                img_batch=filter_image(blur_layer_list[counter],img_batch,turbidity=cur_turb, sky_rbg=cur_sky, useblur=False,use_style=False)
                if current_mask!=None:
                    current_mask=filter_image(blur_layer_list[counter],current_mask,turbidity=cur_turb, sky_rbg=cur_sky, useblur=False,use_style=False)
                img_batch=img_batch*6
                img_batch=img_batch-3


            else:
                img_batch=img_batch+3
                img_batch=img_batch/6
                img_batch=filter_image(blur_layer_list[counter],img_batch,turbidity=cur_turb, sky_rbg=cur_sky, useblur=True,use_style=False)
                if current_mask!=None:
                    current_mask=filter_image(blur_layer_list[counter],current_mask,turbidity=cur_turb, sky_rbg=cur_sky, useblur=True,use_style=False)
                img_batch=img_batch*6
                img_batch=img_batch-3

            img_batch=resize_transform_back_patch(img_batch)
            if current_mask!=None:
                current_mask=resize_transform_back_patch(current_mask)

            img_batch=img_batch+3
            img_batch=img_batch/6
            img_batch=filter_1x_train(img_batch)
            if current_mask!=None:
                current_mask=filter_1x_train(current_mask)
            img_batch=img_batch*6
            img_batch=img_batch-3

            if np.random.rand()<0.5:
                img_batch=run_kernel_sharpen(img_batch,sharpen_1x,1,'normal')
            resize_transform_patch_=T.Resize(size = (patch_blur_size,patch_blur_size)).to(cfg.device)
            resize_transform_back_patch_ = T.Resize(size = (adv_patch_t.shape[2],adv_patch_t.shape[3])).to(cfg.device)
            adv_patch_t=resize_transform_patch_(adv_patch_t)

            if cur_filter_rand<rand_blur_chance:


                cur_filter_rand2=int(np.random.rand()*len(blur_layer_list_rand[counter]))
                adv_patch_t=filter_image(blur_layer_list_rand[counter][cur_filter_rand2],adv_patch_t)

                adv_patch_t=adv_patch_t+3
                adv_patch_t=adv_patch_t/6
                adv_patch_t=filter_image(blur_layer_list[counter],adv_patch_t,turbidity=cur_turb, sky_rbg=cur_sky, useblur=False,use_style=True)
                adv_patch_t=adv_patch_t*6
                adv_patch_t=adv_patch_t-3


            else:
                adv_patch_t=adv_patch_t+3
                adv_patch_t=adv_patch_t/6
                adv_patch_t=filter_image(blur_layer_list[counter],adv_patch_t,turbidity=cur_turb, sky_rbg=cur_sky, useblur=True,use_style=True)
                adv_patch_t=adv_patch_t*6
                adv_patch_t=adv_patch_t-3

            adv_patch_t=resize_transform_back_patch_(adv_patch_t)
            adv_patch_t=run_kernel_sharpen(adv_patch_t,sharpen_1x,1,'normal')

            p_img_batch=img_batch
            p_img_batch=resize_transform_back(p_img_batch)
            if current_mask!=None:
                current_mask=resize_transform_back(current_mask)

            if label_true_forloss.shape[1]==0:
                continue

            p_img_batch,label_true_forloss,current_mask=scale_image_label_1x(p_img_batch,label_true_forloss,current_mask,scale_range=scale_range1x,person_scale=person_scale1x,pad_out_size=out_size_1x)
            p_img_batch,label_true_forloss,current_mask=aug_pedestrian(p_img_batch,label_true_forloss,current_mask)
            p_img_batch,label_true_forloss, ped_mask=scale_image_label(p_img_batch,label_true_forloss,current_mask, counter,scale_xs,scale_range=scale_range)

            cur_bg_index=int(np.random.rand()*len(background_images))
            cur_bg=background_images[cur_bg_index].float()
            cur_bgs=cur_bg.shape
            if cur_bgs[2]>=cur_bgs[3]:
                start=int(np.random.rand()*(cur_bgs[2]-cur_bgs[3]))
                end=start+cur_bgs[3]
                cur_bg=cur_bg[:,:,start:end,:]
            else:
                start=int(np.random.rand()*(cur_bgs[3]-cur_bgs[2]))
                end=start+cur_bgs[2]
                cur_bg=cur_bg[:,:,:,start:end]
            resize_transform_bg1=T.Resize(size = (bg_size,bg_size)).to(cfg.device)
            cur_bg=resize_transform_bg1(cur_bg)
            resize_transform_bg=T.Resize(size = (p_img_batch.shape[2],p_img_batch.shape[3])).to(cfg.device)
            bg_cx_start=int(cur_bg.shape[2]*scale_bg_range*np.random.rand())
            bg_cx_end=int(cur_bg.shape[2]-cur_bg.shape[2]*scale_bg_range*np.random.rand())
            bg_cy_start=int(cur_bg.shape[3]*scale_bg_range*np.random.rand())
            bg_cy_end=int(cur_bg.shape[3]-cur_bg.shape[3]*scale_bg_range*np.random.rand())
            cur_bg=cur_bg[:,:,bg_cx_start:bg_cx_end,bg_cy_start:bg_cy_end]
            cur_bg=resize_transform_bg(cur_bg)
            mask=p_img_batch<-3
            p_img_batch=mask*cur_bg+(1-mask*1)*p_img_batch
            p_img_batch=torch.clamp(p_img_batch,-3,3)






            cur_crop_size=[]
            for i in range(len(crop_size_list[counter])):
                cur_cs=int(np.random.rand()*(crop_size_list[counter][i]-crop_size_list_min[counter][i])+crop_size_list_min[counter][i])
                cur_crop_size.append(cur_cs)
            adv_patch_crop=tshirt_crop(adv_patch_t,cur_crop_size)

            adv_patch_crop_copy=adv_patch_crop
            cos_losses=[]
            for anchor_i in range(len(anchor_weights)):


                adv_patch_crop_v2=adv_patch
                adv_patch_crop_v2_copy=cloth_anchor
                adv_patch_crop_v2_blurred=filter_image(anchor_blur_layers[anchor_i],adv_patch_crop_v2)
                adv_patch_crop_v2_blurred=torch.clamp(adv_patch_crop_v2_blurred,-3,3)
                adv_patch_crop_v2_copy_blurred=filter_image(anchor_blur_layers[anchor_i],adv_patch_crop_v2_copy)
                adv_patch_crop_v2_copy_blurred=torch.clamp(adv_patch_crop_v2_copy_blurred,-3,3)
                oris=adv_patch_crop_v2_blurred.shape
                copys=adv_patch_crop_v2_copy_blurred.shape
                ori_blurred_reshaped=torch.reshape(adv_patch_crop_v2_blurred,[oris[0],oris[1]*oris[2]*oris[3]])
                copy_blurred_reshaped=torch.reshape(adv_patch_crop_v2_copy_blurred,[copys[0],copys[1]*copys[2]*copys[3]])
                cos_loss=cos(ori_blurred_reshaped,copy_blurred_reshaped)*anchor_weights[anchor_i]

                cos_losses.append(cos_loss)



            adv_patch_tps, _ = tps.tps_trans(adv_patch_crop_copy, max_range=tps_strength*np.random.rand(), canvas=0.5)
            if adv_patch_tps.shape[0]==0 or lab_batch.shape[1]==0:
                continue


            label_true_forloss_1=np.asarray(label_true_forloss)
            label_true_forloss_0=np.zeros([label_true_forloss.shape[0],label_true_forloss.shape[1],1])
            label_true_forloss_combined=torch.tensor(np.concatenate([label_true_forloss_0,label_true_forloss_1],2)).float().to(device)

            adv_batch_t = patch_transformer(adv_patch_tps, label_true_forloss_combined, pargs.img_size, do_rotate=True, rand_loc=False,
                                            pooling=pargs.pooling, old_fasion=kwargs['old_fasion'],ratio=cur_crop_size[1]/cur_crop_size[0],rotate_strength=rotate_strengths[counter],shift=shift_strengths[counter])

            resize_transform_back2 = T.Resize(size = (p_img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            p_img_batch=resize_transform(p_img_batch)

            p_img_batch = patch_applier(p_img_batch, adv_batch_t)

            p_img_batch=resize_transform_back2(p_img_batch)



            resize_transform_cur = T.Resize(size = (model_imgsize_list[cur_model_index],model_imgsize_list[cur_model_index])).to(cfg.device)
            p_img_batch=resize_transform_cur(p_img_batch)

            if i_batch<10:
                pre_batch=p_img_batch.detach().cpu().numpy()
                for bidx in range(len(pre_batch)):
                    pre_batchc=255*(pre_batch[bidx]+3)/6
                    pre_batchc=np.transpose(pre_batchc,[1,2,0])
                    pre_batchc=pre_batchc.astype(np.uint8)
                    im = Image.fromarray(pre_batchc,mode="RGB")
                    im.save('./sample_patched/'+args.suffix+'/'+"post"+str(bidx)+'_'+str(i_batch)+".jpeg")

            if args.net=='yolov2':
                det_loss, valid_num = get_det_loss(model_list[cur_model_index], p_img_batch, lab_batch, pargs, kwargs)
            else:
                if cur_model_index==0:
                    output=model_list[cur_model_index].module.forward_dummy(p_img_batch)
                    det_loss, valid_num = get_det_loss_retina(label_true_forloss,data,model_list[cur_model_index],output,pargs,p_img_batch,resolution_weights[counter])
                elif cur_model_index==1:
                    det_loss, valid_num = get_det_loss_mrcnn(data,model_list[cur_model_index],pargs,p_img_batch)
                elif cur_model_index==2:
                    p_img_batch_temp0=p_img_batch+3
                    p_img_batch_temp1=p_img_batch_temp0/6
                    output = model_list[cur_model_index](p_img_batch_temp1)[0]
                    det_loss, valid_num = get_det_loss_yolov5(output,label_true_forloss,cur_model_index)

            if valid_num > 0:
                det_loss = det_loss / valid_num


            tv = total_variation(adv_patch_crop)
            tv_loss = tv * pargs.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            for cos_i in range(len(cos_losses)):
                loss=loss-cos_losses[cos_i]*anchor_weight_overall
            ep_det_loss += det_loss.detach().cpu().numpy()
            ep_tv_loss += tv_loss.detach().cpu().numpy()
            ep_loss += loss.item()

            if counter<args.batch_size-1 or balance_counter<balance_limits[balance_index][counter]:
                loss.backward(retain_graph=True)
                balance_counter+=1
                if balance_counter>=balance_limits[balance_index][counter]:
                    balance_counter=0
                    counter+=1
            if counter>=args.batch_size:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch.data.clamp_(-3, 3)
                counter=0
                balance_counter=0


            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

            if epoch==1 or epoch==3 or epoch==5 or epoch==10 or epoch==30  or epoch==100 or epoch%500==0:
                rpath = os.path.join(result_dir, 'patch_stage2_%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
            bt0 = time.time()


            if cur_resolution_i<args.batch_size-1:
                cur_resolution_i+=1
            else:
                cur_resolution_i=0

        rpath = os.path.join(result_dir, 'patch_stage2_latest.npy')
        np.save(rpath, adv_patch.detach().cpu().numpy())
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        if epoch%loss_eval_epoch==0:
            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('train -- epoch: '+str(epoch)+' ep_det_loss: '+str(ep_det_loss)+' ep_tv_loss: '+str(ep_tv_loss)+' ep_loss: '+str(ep_loss)+'\n')
                std_out.close()
        if epoch > 150:
            scheduler.step(ep_loss)
        et0 = time.time()
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()

        train_counter=0
        model.eval()


def train_EGA():
    gen = GAN_dis(DIM=pargs.DIM, z_dim=pargs.z_dim, img_shape=pargs.patch_size)
    gen.to(device)
    gen.train()

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

    optimizerG = optim.Adam(gen.G.parameters(), lr=pargs.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=pargs.learning_rate, betas=(0.5, 0.9))

    et0 = time.time()
    counter=0
    for epoch in range(1, pargs.n_epochs + 1):
        if epoch>1 and (epoch==2 or epoch%ap_eval_epoch==0):
            gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
            cpt = os.path.join(result_dir, args.net + '_' + args.method.lower() + '.pkl')
            d = torch.load(cpt, map_location='cpu')
            gan.load_state_dict(d)
            gan.to(cfg.device)
            gan.eval()
            for p in gan.parameters():
                p.requires_grad = False
            test_cloth = None
            test_gan = gan

            test_z = None
            test_type = 'gan'
            cloth = gan.generate(torch.randn(1, 128, *([9] * 2), device=cfg.device))

            prec, rec, ap, confs = test(model, train_loader2, adv_cloth=test_cloth, gan=test_gan, z=test_z, type_=test_type,   old_fasion=kwargs['old_fasion'])
            precv, recv, apv, confsv = test(model, val_data_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type_=test_type,   old_fasion=kwargs['old_fasion'])

            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('----------------epoch: '+str(epoch)+' train AP: '+str(ap)+' val AP: '+str(apv)+'\n')
                std_out.close()

        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        D_loss = 0
        bt0 = time.time()
        model.eval()
        for i_batch, data in enumerate(train_loader):
            print('epoch',epoch,'i_batch',i_batch)

            w=data['img_metas']._data[0][0]['ori_shape'][1]
            h=data['img_metas']._data[0][0]['ori_shape'][0]
            whwh=torch.tensor([w,h,w,h])
            whwh=torch.reshape(whwh,[1,1,4])

            img_batch = data['img']._data[0]
            max_length=0
            metas=data['img_metas']._data[0]
            label_true_list=[]
            for i in range(len(metas)):
                label_name=metas[i]['ori_filename']
                label_name=label_name[0:len(label_name)-4]+'.txt'
                true_boxes = np.loadtxt(true_lab_dir+'/'+label_name, dtype=float)

                true_boxes=torch.tensor(true_boxes)

                if len(true_boxes.shape)>1:
                    true_labels=true_boxes[:,:1]
                    true_boxes=true_boxes[:,1:]
                    label_true=torch.cat([true_labels,true_boxes],1)
                    label_true=np.expand_dims(label_true,0)
                elif len(true_boxes.shape)==1 and len(true_boxes)==0:
                    label_true=-1*np.ones([1,0,5])
                else:
                    label_true=np.expand_dims(np.expand_dims(true_boxes,0),0)
                if metas[i]['flip']==True:
                    temp_x=label_true[:,:,1:2]
                    temp_x=1-temp_x
                    label_true[:,:,1:2]=temp_x
                label_true_list.append(label_true)
                if label_true.shape[1]>max_length:
                    max_length=label_true.shape[1]
            lab_batch = -1*torch.ones([img_batch.shape[0],max_length,5])
            for i in range(len(metas)):
                lab_batch[i:i+1,:label_true_list[i].shape[1],:]=torch.tensor(label_true_list[i])

            resize_transform_back = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform(img_batch)

            z = torch.randn(img_batch.shape[0], pargs.z_dim, pargs.z_size, pargs.z_size, device=device)

            adv_patch = gen.generate(z)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=tps_strength, canvas=0.5, target_shape=adv_patch.shape[-2:])
            if adv_patch_tps.shape[0]==0 or lab_batch.shape[1]==0:
                continue
            adv_batch_t = patch_transformer(adv_patch_tps.to(cfg.device), lab_batch.to(cfg.device), pargs.img_size, do_rotate=True, rand_loc=False,
                                            pooling=pargs.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch.to(cfg.device), adv_batch_t.to(cfg.device))
            p_img_batch=resize_transform_back(p_img_batch)

            if args.net=='yolov2':
                det_loss, valid_num = get_det_loss(model, p_img_batch, lab_batch, pargs, kwargs)
            else:
                output=model.module.forward_dummy(p_img_batch)
                det_loss, valid_num = get_det_loss_retina(data,model,output,pargs,p_img_batch)

            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], pargs.gp)
            tv_loss = tv * pargs.tv_loss
            disc_loss = disc * pargs.disc if epoch >= pargs.dim_start_epoch else disc * 0.0

            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()

            if counter<args.batch_size:
                loss.backward(retain_graph=True)
                counter+=1
            else:
                loss.backward()
                optimizerG.step()
                optimizerD.step()
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                adv_patch.data.clamp_(-3, 3)
                counter=0

            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

            if epoch==1 or epoch==3 or epoch==5 or epoch==10 or epoch==30  or epoch==100 or epoch%500==0:
                rpath = os.path.join(result_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                torch.save(gen.state_dict(), os.path.join(result_dir, args.suffix + '.pkl'))
            torch.save(gen.state_dict(), os.path.join(result_dir, args.suffix + '.pkl'))

            bt0 = time.time()
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        if epoch%loss_eval_epoch==0:
            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('train -- epoch: '+str(epoch)+' ep_det_loss: '+str(ep_det_loss)+' ep_tv_loss: '+str(ep_tv_loss)+' ep_loss: '+str(ep_loss)+'\n')
                std_out.close()

        et0 = time.time()
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        D_loss = 0
        bt0 = time.time()
        model.eval()
        for i_batch, data in enumerate(val_data_loader):
            print('epoch',epoch,'i_batch',i_batch)

            w=data['img_metas'][0]._data[0][0]['ori_shape'][1]
            h=data['img_metas'][0]._data[0][0]['ori_shape'][0]
            whwh=torch.tensor([w,h,w,h])
            whwh=torch.reshape(whwh,[1,1,4])

            img_batch = data['img'][0]
            max_length=0
            metas=data['img_metas'][0]._data[0]
            label_true_list=[]
            if_continue=False
            for i in range(len(metas)):
                label_name=metas[i]['ori_filename']
                label_name=label_name[0:len(label_name)-4]+'.txt'
                true_boxes = np.loadtxt(true_lab_dir+'/'+label_name, dtype=float)

                true_boxes=torch.tensor(true_boxes)

                if len(true_boxes.shape)>1:
                    true_labels=true_boxes[:,:1]
                    true_boxes=true_boxes[:,1:]
                    label_true=torch.cat([true_labels,true_boxes],1)
                    label_true=np.expand_dims(label_true,0)
                elif len(true_boxes.shape)==1 and len(true_boxes)==0:
                    if_continue=True
                    label_true=-1*np.ones([1,max_length,5])
                else:
                    label_true=np.expand_dims(np.expand_dims(true_boxes,0),0)
                if metas[i]['flip']==True:
                    temp_x=label_true[:,:,1:2]
                    temp_x=1-temp_x
                    label_true[:,:,1:2]=temp_x
                label_true_list.append(label_true)
                if label_true.shape[1]>max_length:
                    max_length=label_true.shape[1]
            lab_batch = -1*torch.ones([img_batch.shape[0],max_length,5])
            for i in range(len(metas)):
                lab_batch[i:i+1,:label_true_list[i].shape[1],:]=torch.tensor(label_true_list[i])
            if if_continue==True:
                continue

            resize_transform_back = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform(img_batch)

            z = torch.randn(img_batch.shape[0], pargs.z_dim, pargs.z_size, pargs.z_size, device=device)

            adv_patch = gen.generate(z)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=tps_strength, canvas=0.5, target_shape=adv_patch.shape[-2:])
            if adv_patch_tps.shape[0]==0 or lab_batch.shape[1]==0:
                continue
            adv_batch_t = patch_transformer(adv_patch_tps.to(cfg.device), lab_batch.to(cfg.device), pargs.img_size, do_rotate=True, rand_loc=False,
                                            pooling=pargs.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch.to(cfg.device), adv_batch_t.to(cfg.device))
            p_img_batch=resize_transform_back(p_img_batch)

            if args.net=='yolov2':
                det_loss, valid_num = get_det_loss(model, p_img_batch, lab_batch, pargs, kwargs)
            else:
                output=model.module.forward_dummy(p_img_batch)
                det_loss, valid_num = get_det_loss_retina(data,model,output,pargs,p_img_batch)

            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], pargs.gp)
            tv_loss = tv * pargs.tv_loss
            disc_loss = disc * pargs.disc if epoch >= pargs.dim_start_epoch else disc * 0.0

            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()
        ep_det_loss = ep_det_loss / len(val_data_loader)
        ep_tv_loss = ep_tv_loss / len(val_data_loader)
        ep_loss = ep_loss / len(val_data_loader)
        if epoch%loss_eval_epoch==0:
            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('val -- epoch: '+str(epoch)+' ep_det_loss: '+str(ep_det_loss)+' ep_tv_loss: '+str(ep_tv_loss)+' ep_loss: '+str(ep_loss)+'\n')
                std_out.close()

    return gen


def train_z(gen=None):
    z_path = './results/result_' + args.net + '_' + args.method+'zlatest.npy'
    if gen is None:
        gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
        suffix_load = pargs.gen_suffix
        d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
        gen.load_state_dict(d)
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    z0 = torch.randn(*pargs.z_shape, device=device)
    z = z0.detach().clone()
    z.requires_grad_(True)

    optimizer = optim.Adam([z], lr=pargs.learning_rate_z, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=pargs.learning_rate_z / 100)

    resolutions=[]
    min_res=1
    max_res=4
    resolutions.append(min_res)
    interval=(max_res-min_res)/(args.batch_size-1)
    for i in range(args.batch_size-1):
        cur=min_res+i*interval
        resolutions.append(cur)

    et0 = time.time()
    counter=0
    for epoch in range(1, pargs.z_epochs + 1):
        if epoch>1 and (epoch==2 or epoch%ap_eval_epoch==0):
            print('---point2-----')
            gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)

            cpt = os.path.join(result_dir, args.net + '_' + args.method.lower() + '.pkl')
            d = torch.load(cpt, map_location='cpu')
            gan.load_state_dict(d)
            gan.to(cfg.device)
            gan.eval()
            for p in gan.parameters():
                p.requires_grad = False
            test_cloth = None
            test_gan = gan
            z = np.load(z_path)
            z = torch.from_numpy(z).to(cfg.device)
            test_z = z
            test_type = 'z'
            z_crop, _, _ = random_crop(z, [9] * 2, pos=None, crop_type='recursive')
            cloth = gan.generate(z_crop)

            prec, rec, ap, confs = test(model, train_loader2, adv_cloth=test_cloth, gan=test_gan, z=test_z, type_=test_type,   old_fasion=kwargs['old_fasion'])
            precv, recv, apv, confsv = test(model, val_data_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type_=test_type,   old_fasion=kwargs['old_fasion'])

            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('----------------epoch: '+str(epoch)+' train AP: '+str(ap)+' val AP: '+str(apv)+'\n')
                std_out.close()

        z.requires_grad_(True)
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        model.eval()

        cur_resolution_i=0

        for i_batch, data in enumerate(train_loader):
            print('epoch',epoch,'i_batch',i_batch)

            w=data['img_metas']._data[0][0]['ori_shape'][1]
            h=data['img_metas']._data[0][0]['ori_shape'][0]
            whwh=torch.tensor([w,h,w,h])
            whwh=torch.reshape(whwh,[1,1,4])


            img_batch = data['img']._data[0]
            max_length=0
            metas=data['img_metas']._data[0]
            label_true_list=[]
            for i in range(len(metas)):
                label_name=metas[i]['ori_filename']
                label_name=label_name[0:len(label_name)-4]+'.txt'
                true_boxes = np.loadtxt(true_lab_dir+'/'+label_name, dtype=float)

                true_boxes=torch.tensor(true_boxes)

                if len(true_boxes.shape)>1:
                    true_labels=true_boxes[:,:1]
                    true_boxes=true_boxes[:,1:]
                    label_true=torch.cat([true_labels,true_boxes],1)
                    label_true=np.expand_dims(label_true,0)
                elif len(true_boxes.shape)==1 and len(true_boxes)==0:
                    label_true=-1*np.ones([1,0,5])
                else:
                    label_true=np.expand_dims(np.expand_dims(true_boxes,0),0)
                if metas[i]['flip']==True:
                    temp_x=label_true[:,:,1:2]
                    temp_x=1-temp_x
                    label_true[:,:,1:2]=temp_x
                label_true_list.append(label_true)
                if label_true.shape[1]>max_length:
                    max_length=label_true.shape[1]
            lab_batch = -1*torch.ones([img_batch.shape[0],max_length,5])
            for i in range(len(metas)):
                lab_batch[i:i+1,:label_true_list[i].shape[1],:]=torch.tensor(label_true_list[i])

            resize_transform_back = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform(img_batch)
            z_crop, _, _ = random_crop(z, [9] * 2, pos=None, crop_type='recursive')

            adv_patch = gen.generate(z_crop)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=tps_strength, canvas=0.5, target_shape=adv_patch.shape[-2:])
            if adv_patch_tps.shape[0]==0 or lab_batch.shape[1]==0:
                continue
            adv_batch_t = patch_transformer(adv_patch_tps.to(cfg.device), lab_batch.to(cfg.device), pargs.img_size, do_rotate=True, rand_loc=False,
                                            pooling=pargs.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch.to(cfg.device), adv_batch_t)
            p_img_batch=resize_transform_back(p_img_batch)
            output=model.module.forward_dummy(p_img_batch)
            det_loss, valid_num = get_det_loss_retina(data,model,output,pargs,p_img_batch)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            tv_loss = tv * pargs.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()
            if counter<args.batch_size:
                loss.backward(retain_graph=True)
                counter+=1
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch.data.clamp_(-3, 3)
                counter=0
            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

            if epoch==1 or epoch==3 or epoch==5 or epoch==10 or epoch==30  or epoch==100 or epoch%500==0:
                rpath = os.path.join(result_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                rpath = os.path.join(result_dir, 'z%d' % epoch)
                np.save(rpath, z.detach().cpu().numpy())
            np.save(z_path, z.detach().cpu().numpy())
            bt0 = time.time()

            if cur_resolution_i<args.batch_size-1:
                cur_resolution_i+=1
            else:
                cur_resolution_i=0

        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        if epoch%loss_eval_epoch==0:
            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('train -- epoch: '+str(epoch)+' ep_det_loss: '+str(ep_det_loss)+' ep_tv_loss: '+str(ep_tv_loss)+' ep_loss: '+str(ep_loss)+'\n')
                std_out.write('\n')
                std_out.close()
        if epoch > 300:
            scheduler.step(ep_loss)
        et0 = time.time()


        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        model.eval()
        for i_batch, data in enumerate(val_data_loader):
            print('epoch',epoch,'i_batch',i_batch)

            w=data['img_metas'][0]._data[0][0]['ori_shape'][1]
            h=data['img_metas'][0]._data[0][0]['ori_shape'][0]
            whwh=torch.tensor([w,h,w,h])
            whwh=torch.reshape(whwh,[1,1,4])


            img_batch = data['img'][0]
            max_length=0
            metas=data['img_metas'][0]._data[0]
            label_true_list=[]
            if_continue=False
            for i in range(len(metas)):
                label_name=metas[i]['ori_filename']
                label_name=label_name[0:len(label_name)-4]+'.txt'
                true_boxes = np.loadtxt(true_lab_dir+'/'+label_name, dtype=float)

                true_boxes=torch.tensor(true_boxes)

                if len(true_boxes.shape)>1:
                    true_labels=true_boxes[:,:1]
                    true_boxes=true_boxes[:,1:]
                    label_true=torch.cat([true_labels,true_boxes],1)
                    label_true=np.expand_dims(label_true,0)
                elif len(true_boxes.shape)==1 and len(true_boxes)==0:
                    if_continue=True
                    label_true=-1*np.ones([1,max_length,5])
                else:
                    label_true=np.expand_dims(np.expand_dims(true_boxes,0),0)
                if metas[i]['flip']==True:
                    temp_x=label_true[:,:,1:2]
                    temp_x=1-temp_x
                    label_true[:,:,1:2]=temp_x
                label_true_list.append(label_true)
                if label_true.shape[1]>max_length:
                    max_length=label_true.shape[1]
            if if_continue==True:
                continue
            lab_batch = -1*torch.ones([img_batch.shape[0],max_length,5])
            for i in range(len(metas)):
                lab_batch[i:i+1,:label_true_list[i].shape[1],:]=torch.tensor(label_true_list[i])

            resize_transform_back = T.Resize(size = (img_batch.shape[2],img_batch.shape[3])).to(cfg.device)
            img_batch=resize_transform(img_batch)


            z_crop, _, _ = random_crop(z, pargs.crop_size_z, pos=pargs.pos, crop_type=pargs.crop_type_z)

            adv_patch = gen.generate(z_crop)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=tps_strength, canvas=0.5, target_shape=adv_patch.shape[-2:])
            if adv_patch_tps.shape[0]==0 or lab_batch.shape[1]==0:
                continue
            adv_batch_t = patch_transformer(adv_patch_tps.to(cfg.device), lab_batch.to(cfg.device), pargs.img_size, do_rotate=True, rand_loc=False,
                                            pooling=pargs.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch.to(cfg.device), adv_batch_t)
            p_img_batch=resize_transform_back(p_img_batch)
            output=model.module.forward_dummy(p_img_batch)
            det_loss, valid_num = get_det_loss_retina(data,model,output,pargs,p_img_batch)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            tv_loss = tv * pargs.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()
        ep_det_loss = ep_det_loss / len(val_data_loader)
        ep_tv_loss = ep_tv_loss / len(val_data_loader)
        ep_loss = ep_loss / len(val_data_loader)
        if epoch%loss_eval_epoch==0:
            with open(args.suffix+'_stdout'+".txt", "a") as std_out:
                std_out.write('val -- epoch: '+str(epoch)+' ep_det_loss: '+str(ep_det_loss)+' ep_tv_loss: '+str(ep_tv_loss)+' ep_loss: '+str(ep_loss)+'\n')
                std_out.close()



    return 0


if args.method == 'RCA':
    train_patch_stage1()
    train_patch()
elif args.method == 'TCA':
    train_patch_stage1()
    train_patch()
elif args.method == 'EGA':
    train_EGA()
elif args.method == 'TCEGA':
    gen = train_EGA()
    print('Start optimize z')
    train_z(gen)
