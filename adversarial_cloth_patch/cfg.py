from easydict import EasyDict

v2kwargs={
    'name': 'yolov2',
    'cfgfile': "./data/models/yolov2.cfg",
    'weightfile': "./data/models/yolov2.weights",
    'max_lab': 15,
    'batch_size': 8,
    'old_fasion': True,
}

v2kwargs_pvt_tiny_retina={
    'name': 'pvt_tiny_retina',
    'cfgfile': "./data/models/yolov2.cfg",
    'weightfile': "./data/models/yolov2.weights",
    'max_lab': 20,
    'batch_size': 8,
    'old_fasion': True,
}

v2kwargs_res50_retina={
    'name': 'res50_retina',
    'cfgfile': "./data/models/yolov2.cfg",
    'weightfile': "./data/models/yolov2.weights",
    'max_lab': 20,
    'batch_size': 8,
    'old_fasion': True,
}

v2kwargs_swin_tiny_maskrcnn={
    'name': 'swin_tiny_maskrcnn',
    'cfgfile': "./data/models/swin_tiny_maskrcnn.cfg",
    'weightfile': "./data/models/swin_tiny_maskrcnn.weights",
    'max_lab': 20,
    'batch_size': 8,
    'old_fasion': True,
}

v2kwargs_resnet50_maskrcnn={
    'name': 'resnet50_maskrcnn',
    'cfgfile': "./data/models/swin_tiny_maskrcnn.cfg",
    'weightfile': "./data/models/swin_tiny_maskrcnn.weights",
    'max_lab': 20,
    'batch_size': 8,
    'old_fasion': True,
}

v2kwargs_faster_rcnn={
    'name': 'faster_rcnn',
    'cfgfile': "./data/models/faster_rcnn.cfg",
    'weightfile': "./data/models/faster_rcnn.weights",
    'max_lab': 20,
    'batch_size': 8,
    'old_fasion': True,
}

v2kwargs_detr={
    'name': 'detr',
    'cfgfile': "./data/models/faster_rcnn.cfg",
    'weightfile': "./data/models/faster_rcnn.weights",
    'max_lab': 20,
    'batch_size': 8,
    'old_fasion': True,
}

v2kwargs_deformable_detr={
    'name': 'deformable_detr',
    'cfgfile': "./data/models/faster_rcnn.cfg",
    'weightfile': "./data/models/faster_rcnn.weights",
    'max_lab': 20,
    'batch_size': 8,
    'old_fasion': True,
}

args_RCA = {
    'cloth_size': [150,150],#[900, 900],
    'crop_size': 'equal',#[150, 150],
    'crop_type': None,
    'pooling': 'gauss',
    'pixel_size': [1, 1],
    'pos': None,
    'tps_range': 0.1,
    'tps_canvas': 0.5,
    'n_epochs': 2000,
    'learning_rate': 0.03,
    'tv_loss': 0,
    'img_size': 416,
    'eps': 1e-5,
    'gp': 0,
    'iou_thresh': 0.01,
    'conf_thresh': 0.01,
}

args_TCA = {
    'cloth_size': [300, 300],
    'crop_size': [150, 150],
    'crop_type': 'recursive',
    'pooling': 'gauss',
    'pixel_size': [1, 1],
    'pos': None,
    'tps_range': 0.1,
    'tps_canvas': 0.5,
    'n_epochs': 2000,
    'learning_rate': 0.03,
    'tv_loss': 0,
    'img_size': 416,
    'eps': 1e-5,
    'gp': 0,
    'iou_thresh': 0.01,
    'conf_thresh': 0.01,
}

args_EGA = {
    'crop_size': 'equal',
    'crop_type': None,
    'pixel_size': [1, 1],
    'pos': None,  # ['center', None]
    'tps_range': 0.1,
    'tps_canvas': 0.5,
    'n_epochs': 2000,
    'learning_rate': 0.001,
    'DIM': 128,
    'z_dim': 128,
    'z_size': 9,
    'patch_size': [324] * 2,
    'pooling': 'median',
    'dim_start_epoch': 0,
    'det_epoch': 0,
    'disc': 0.5,
    'img_size': 416,
    'eps': 1e-5,
    'tv_loss': 2.5,
    'gp': 0,
    'iou_thresh': 0.3,
    'conf_thresh': 0.01,

}

args_TCEGA = {
    'crop_size': 'equal',
    'crop_type': None,
    'z_shape': [1, 128, 4, 4],
    'crop_size_z': [9, 9],
    'crop_type_z': 'recursive',
    'pixel_size': [1, 1],
    'pos': None,  # ['center', None]
    'tps_range': 0.1,
    'tps_canvas': 0.5,
    'n_epochs': 2000,
    'z_epochs': 2000,
    'learning_rate': 0.001,
    'learning_rate_z': 0.03,
    'DIM': 128,
    'z_dim': 128,
    'z_size': 9,
    'patch_size': [324] * 2,
    'pooling': 'median',
    'dim_start_epoch': 0,
    'det_epoch': 0,
    'disc': 0.5,
    'img_size': 416,
    'eps': 1e-5,
    'tv_loss': 0,
    'gp': 0,
    'iou_thresh': 0.3,
    'conf_thresh': 0.01,
}

targs_RCA = {
    'pos': None,
    'crop_size': [150] * 2,
    'crop_type': None,
    'pixel_size': [1] * 2,
    'pooling': 'gauss',
    'img_size': 416,
    'batch_size': 8,
}

targs_TCA = {
    'pos': None,
    'crop_size': [150] * 2,
    'crop_type': 'recursive',
    'pixel_size': [1] * 2,
    'pooling': 'gauss',

    'img_size': 416,
    'batch_size': 8,
}

targs_EGA = {
    'z_size': [9] * 2,
    'pos': 'center',
    'crop_size': 'equal',
    'crop_type': None,
    'pixel_size': [1] * 2,
    'pooling': 'median',
    'img_size': 416,
    'batch_size': 8,
}

targs_TCEGA = {
    'z_pos': None,
    'z_crop_size': [9] * 2,
    'z_crop_type': 'recursive',
    'pos': 'center',
    'crop_size': 'equal',
    'crop_type': None,
    'pixel_size': [1] * 2,
    'pooling': 'median',
    'img_size': 416,
    'batch_size': 8,
}

kwargs_dict = {
    'yolov2': v2kwargs,
    'pvt_tiny_retina': v2kwargs_pvt_tiny_retina,
    'swin_tiny_maskrcnn':v2kwargs_swin_tiny_maskrcnn,
    'faster_rcnn':v2kwargs_faster_rcnn,
    'res50_retina':v2kwargs_res50_retina,
    'resnet50_maskrcnn':v2kwargs_resnet50_maskrcnn,
    'detr':v2kwargs_detr,
    'deformable_detr':v2kwargs_deformable_detr,
}

args_dict = {
    'RCA': args_RCA,
    'TCA': args_TCA,
    'EGA': args_EGA,
    'TCEGA': args_TCEGA
    }

targs_dict = {
    'RCA': targs_RCA,
    'TCA': targs_TCA,
    'EGA': targs_EGA,
    'TCEGA': targs_TCEGA
    }


def get_cfgs(net_name, method_name, mode='training'):
    if mode == 'training':
        args = args_dict[method_name]
        args = EasyDict(args)
    elif mode == 'test':
        args = targs_dict[method_name]
        args = EasyDict(args)
    else:
        raise ValueError
    kwargs = kwargs_dict[net_name]
    return args, kwargs