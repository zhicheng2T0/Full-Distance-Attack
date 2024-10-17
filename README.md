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

