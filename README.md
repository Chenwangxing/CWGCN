# CWGCN: Cascaded Wavelet Graph Convolution Network for Pedestrian Trajectory Prediction

The code and weights have been released, enjoy it！ You can easily run the model！ To use the pretrained models at checkpoint/CWGCN and evaluate the models performance run:  test.py


## Introduction
Existing GNN-based methods typically model pedestrian social interactions at a global scale, overlooking multi-scale interactions, as shown in Fig. 1(a). The wavelet transform is utilized as a multi-scale modeling method in image processing. While WTGCN integrates the wavelet transform with graph convolution, it does not explore cascaded wavelet transforms. Pedestrian social interactions can be viewed as spatially multi-resolution signals and cascaded wavelet transforms can be implemented to analyze these relationships at different resolutions, thus enabling the capture of pedestrian social relationships at different spatial scales, as shown in Fig. 1(b). We combine cascaded wavelet transform and graph convolution to achieve modeling of multi-scale social interactions.
![Figure 1](https://github.com/user-attachments/assets/638fbdde-74c5-4225-bd43-6e13e538a516)


Specifically, we first employ graph convolution to preliminarily describe the social relationships and then apply cascaded wavelet transform for multi-level decomposition. Finally, we utilize convolutional neural networks to model multi-scale features, as shown in Fig. 2(a). Accurately capturing pedestrian temporal interaction features can enhance the perception of future movement trends among pedestrians, thereby achieving more precise trajectory prediction. Similarly, previous methods have often neglected the modeling of temporal interactions at multiple scales. Therefore, we extend cascaded wavelet transforms to model temporal interactions, as shown in Fig. 2(b).
![Figure 2](https://github.com/user-attachments/assets/dded1b10-85ed-4990-8ac7-52b931b13783)


## Method
We propose a cascaded wavelet graph convolution network (CWGCN) that combines graph convolution with cascaded wavelet transforms, as shown in Fig. 3. Building on graph convolution, we design a cascaded wavelet transform module to further model the social and temporal interactions of pedestrians at multiple scales. Then, we develop a spatial-temporal guided fusion module to appropriately weigh and fuse the social and temporal interaction features. Finally, we employ temporal convolutional networks to directly predict multiple trajectories of pedestrians, removing the limitation of sampling.
![Figure 3](https://github.com/user-attachments/assets/51f2375e-0c6b-48aa-bb9e-ed6682f78af6)


## Code Structure
checkpoint folder: contains the trained models

dataset folder: contains ETH, UCY and SDD datasets

model.py: the code of CHGNN

train.py: for training the code

test.py: for testing the code

utils.py: general utils used by the code

metrics.py: metrics tools used by the code

The specific code of the training part will be released after the paper is officially published!


## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py


## Acknowledgement
Some codes are borrowed from SGCN, IMGCN, and WTConv. We gratefully acknowledge the authors for posting their code.


## Cite this article:
Chen W, Sang H, Zhao Z. CWGCN: Cascaded Wavelet Graph Convolution Network for pedestrian trajectory prediction[J]. Computers and Electrical Engineering, 2025, 127: 110609.
