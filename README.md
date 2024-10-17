# Full-Distance Attack

This is the official github repository for the NIPS-2024 paper Full-Distance Evasion of Pedestrian Detectors in the Physical World

# Introduction

Many studies have proposed attack methods to generate adversarial patterns for evading pedestrian detection, alarming the Computer Vision (CV) community about the need for more attention to the robustness of detectors. However, adversarial patterns optimized by these methods commonly have limited performance at medium to long distances in the physical world. To overcome this limitation, we identify two main challenges. First, in existing methods, there is commonly an appearance gap between simulated distant adversarial patterns and their physical world counterparts, leading to incorrect optimization. Second, there exists a conflict between adversarial losses at different distances, which causes difficulties in optimization. To overcome these challenges, we introduce a Full Distance Attack (FDA) method. Our physical world experiments demonstrate the effectiveness of our FDA patterns across various detection models like YOLOv5, Deformable-DETR, and Mask RCNN.

# Method

In this study, we find the cause for existing attack methods to fail at medium to long distances in the physical world is the naive distant image simulation technique used when optimizing the adversarial patterns. More specifically, to simulate the appearance of a distant adversarial pattern during optimization, the existing attack algorithms usually naively downscale and apply the adversarial patterns according to the size of the pedestrians (as illustrated in the figure below). Such a naive technique creates a widening appearance gap between the simulated patterns and their real-world counterparts as distance increases. This leads to the optimization of incorrect adversarial patterns. 
![alt text](https://github.com/zhicheng2T0/Full-Distance-Attack/blob/main/image1.PNG)

To solve this problem, we propose a Distant Image Converter (DIC) to convert images of short-distance objects into an appearance similar to their physical world counterparts at long distances. In DIC, We find it necessary to simulate three factors in the physical world that contribute to the appearance gap. These factors include the effect of atmospheric perspective which changes object colors due to increasing scattering of light as distance increases, the effect of camera hardware which blurs the field of light projected from the target object to form a digital image, and the effect of the default effect filters commonly installed in digital cameras which change the color and texture details of the captured images for better visual appearances. The DIC is illustrted in the figure below.
![alt text](https://github.com/zhicheng2T0/Full-Distance-Attack/blob/main/image2.PNG)

By applying the DIC during optimization, we found that different low frequency patterns were required at short and long distances, causing a conflict, hindering full distance attack (FDA) pattern optimization. To overcome the difficulty, we propose a Multi-Frequency Optimization (MFO) technique. By combining DIC and MFO, we form the FDA method which generates effective adversarial patterns for evading pedestrian detectors at varying distances.

# Key Results

When treating the YOLOV5 model as the target model, the FDA pattern obtained an average ASR of 74%, outperforming the baseline method (Adv-T.) by 52%.
![alt text](https://github.com/zhicheng2T0/Full-Distance-Attack/blob/main/image3.PNG)

In the setting of clothing attack, the FDA pattern has also obtained an average ASR of 76% (in the front and back view), outperforming the baseline method (TCA) by 39%.
![alt text](https://github.com/zhicheng2T0/Full-Distance-Attack/blob/main/image4.PNG)

# Citation

If you find our work to be useful for your research, please consider citing:

      @InProceedings{Hu_2023_CVPR,
          author    = {Cheng, Zhi and Hu, Zhanhao and Liu, Yuqiu and Li, Jianmin and Su, Hang and Hu, Xiaolin},
          title     = {Full-Distance Evasion of Pedestrian Detectors in the Physical World},
          booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
          month     = {November},
          year      = {2024}
      }

# Setup the Environment


We recommend using CUDA11.8, python 3.9.18

Install mmdetection: 

      pip install mmdet

Install mmcv-full-1.7.0:

      pip install -U openmim 
      mim install mmcv-full==1.7.0
      pip install tensorboardX 

Install other packages imported with pip

Download the files provided in 

      https://drive.google.com/drive/folders/1ZsRN7ke8z2-q9aigZsAWQK4irEausvlH?usp=drive_link

cd to adversarial_cloth_patch

create a checkpoint folder and move the checkpoints downloaded into it.

create a dataset folder outside the adversarial_cloth_patch folder and move the 2023_5_11_diverse_person_background and 2024_3_5_internet_ped folder into it

move the version_2023_2_23_temp.pth in checkpoints to /patches_to_load/2023_3_1_color_mapping_network/ in adversarial_cloth_patch

Download the inria dataset with: 

      curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar

decompress it with: tar xf inria.tar
move it to adversarial_cloth_patch/data with: mv INRIAPerson ./data/INRIAPerson
Then, convert its format with

      cd adversarial_cloth_patch/data/INRIAPerson/Test/
      create Annotations folder in Test
      copy inria_to_voc.py into Test

run: 

      python inria_to_voc.py

Then,

      cd adversarial_cloth_patch/data/INRIAPerson/Train/
      create Annotations folder in Train
      copy inria_to_voc.py into Train

run:

      python inria_to_voc.py

In server_move_train.py, replace /data/chengzhi/adversarial_cloth with the current adversarial_cloth_patch directory
create /adversarial_cloth_patch/data/INRIAPerson/VOC2007_COCO/images/ folder
cd to adversarial_cloth_patch/data/INRIAPerson/Train, upload server_move_train.py, run: 

      python server_move_train.py

In server_move_test.py, replace /data/chengzhi/adversarial_cloth with the current adversarial_cloth_patch directory
create /adversarial_cloth_patch/data/INRIAPerson/VOC2007_COCO/test_images/ folder
cd to adversarial_cloth_patch/data/INRIAPerson/Test, upload server_move_train.py, run: 

      python server_move_test.py

create /INRIAPerson/VOC2007/JPEGImages folder
cd to adversarial_cloth_patch
replace /data/chengzhi/adversarial_cloth in inria_voc2coco.py and inria_voc2coco_test.py with the current adversarial_cloth_patch directory
run: 

      python inria_voc2coco.py and python inria_voc2coco_test.py

In create_json.py, replace /data/chengzhi/adversarial_cloth with the current adversarial_cloth_patch directory and run: 

      python create_json.py

Replace /data/chengzhi/adversarial_cloth in adversarial_cloth_patch/configs/_base_/datasets/inria_3090.py with the current adversarial_cloth_patch directory.

# Optimize Color Mapping Function

Within our work, since we found there are gaps between the digital world colors and the printed colors, we have included color mapping functions to bridge this gap. Here we provide the polynomial mapper code from the AdvCaT paper (refer to Appendix 1.4 of Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling paper for more details). The codes are provided in the color_mapping folder.

To preprocess the data: 

      python 2023_7_13_color_mapping_data.py

To optimize a color mapping model: 

      python 2023_7_13_color_mapping.py

We strongly encourage you to collect your own "digital color - printed color pairs" dataset to train your own mapping function, since the deviation is commonly observed in most printers. 

If you find color mapping performance to be limited, please try tuning the "degree" parameter.

# Optimizing the DIC

All sample code related to training and evaluating the DIC are provided in the DIC_codes folder. Before training the DIC, please remember to replace the color mapping function with your own. 

To train the DIC: 

      CUDA_VISIBLE_DEVICES=x python optimize_DIC.py

Considering different cameras can be very different, we strongly encourage you to collect your own distant image dataset so that you can train your own DIC to obtain better attack performance.

# Optimizing FDA patterns

Please remember to replace the DIC and the color mapping function with your own version before optimizing the FDA pattern. 

Relevant codes are provided in adversarial_cloth_patch folder.

Before optimizing FDA patterns, we need to form the pedestrian masks for the pedestrian dataset using MRCNN.

      CUDA_VISIBLE_DEVICES=1 python form_mask.py --config=configs/pvt/retinanet_pvt-t_fpn_1x_inria.py --checkpoint=checkpoints/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth --config2=configs/mask_rcnn/mask_rcnn_r50_fpn_1x_inria.py --checkpoint2=checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --method=TCA --net=pvt_tiny_retina --suffix run_form_mask

After running the code above, move the folder for the masks generated into the 2024_3_5_internet_ped folder.

Also, to perform informative digital world tests, please collect your own testing images at 4m, 8m, 14m, 20m, 26m, 34m, 40m and provide the base address of the testing images to the parameter "sub_data_dirct_general". For each distance, we recommend to include around 300 images including three subjects at three locations. If you are only interested in physical world results, please remember to comment out the testing loops.

For all FDA pattern optimization codes, set the data_base variable to be the base address of 2023_5_11_diverse_person_background and 2024_3_5_internet_ped folder.

AdvTshirt - YOLOV5 - FDA pattern optimization

      CUDA_VISIBLE_DEVICES=x python yolov5_advtshirt_fda.py --config=configs/pvt/retinanet_pvt-t_fpn_1x_inria.py --checkpoint=checkpoints/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth --config2=configs/mask_rcnn/mask_rcnn_r50_fpn_1x_inria.py --checkpoint2=checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --config3=configs/deformable_detr/deformable_detr_r50_16x2_50e_inria.py --checkpoint3=checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth --method=TCA --net=pvt_tiny_retina --suffix run_yolov5_advtshirt_fda

AdvTshirt - Ensemble - FDA pattern optimization

      To optimize FDA pattern effective for different models, we leveraged the ensemble attack method. That is, we optimized the FDA pattern to be effective for two white box models (the ResNet based MRCNN and the SWIN based MRCNN). To achieve this, the optimization is divided into three overall stages. In the first two overall stages, the FDA pattern was optimized against the ResNet based MRCNN and in the last overall stage, it was optimized against the SWIN based MRCNN.
For overall stage 2 and 3, please load the patch from the previous stage.

To run overall stage 1: 

      CUDA_VISIBLE_DEVICES=x python ensemble_advtshirt_stage1.py --config=configs/pvt/retinanet_pvt-t_fpn_1x_inria.py --checkpoint=checkpoints/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth --config2=configs/mask_rcnn/mask_rcnn_r50_fpn_1x_inria.py --checkpoint2=checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --method=TCA --net=pvt_tiny_retina --suffix run_advtshirt_ensemble_stage1

To run overall stage 2: 

      CUDA_VISIBLE_DEVICES=x python ensemble_advtshirt_stage2.py --config=configs/pvt/retinanet_pvt-t_fpn_1x_inria.py --checkpoint=checkpoints/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth --config2=configs/mask_rcnn/mask_rcnn_r50_fpn_1x_inria.py --checkpoint2=checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --method=TCA --net=pvt_tiny_retina --suffix run_advtshirt_ensemble_stage2

To run overall stage 3: 

      CUDA_VISIBLE_DEVICES=x python ensemble_advtshirt_stage3.py --config=configs/pvt/retinanet_pvt-t_fpn_1x_inria.py --checkpoint=checkpoints/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth --config2=configs/mask_rcnn/mask_rcnn_r50_fpn_1x_inria.py --checkpoint2=checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --config3=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_inria.py --checkpoint3=checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth --method=TCA --net=pvt_tiny_retina --suffix run_ensemble_advtshirt_stage3

TCA- YOLOV5 - FDA pattern optimization

      CUDA_VISIBLE_DEVICES=x python yolov5_tca_fda.py --config=configs/pvt/retinanet_pvt-t_fpn_1x_inria.py --checkpoint=checkpoints/retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth --config2=configs/mask_rcnn/mask_rcnn_r50_fpn_1x_inria.py --checkpoint2=checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --config3=configs/deformable_detr/deformable_detr_r50_16x2_50e_inria.py --checkpoint3=checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth --method=TCA --net=pvt_tiny_retina --suffix run_yolov5_tca_fda

# Our suggestions on how to tune the FDA algorithmn when average ASR is low

If you are trying to attack a new target model and you find the average ASR to be low, we would recommend you to try different weights for the IOU loss and the confidence loss. We found the optimum weight to be different for different models.

There also exist some models that have a large amount of local minimums (e.g. DETR and deformable DETR). If you wish to avoid those local minimums, we would recommend you try to set different adversarial patterns as starting points and continue optimzing those patterns against your target model.

We have also found it important to tune the balance_limits parameter for different models.
