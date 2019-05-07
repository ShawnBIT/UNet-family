# Introduction
Since 2015, **UNet** has made major breakthroughs in the medical image segmentation , opening the era of deep learning. Later researchers have made a lot of improvements on the basis of UNet in order to improve the performance of semantic segmentation.

In this project, we have compiled the semantic segmentation models related to UNet(**UNet family**) in recent years.
My implementation is mainly based on **pytorch**, and other implementations are collected from author of original paper or excellent repositories. For the record, this project is still **under construction**. If you have any advice or question, please raise an issue or contact me from email.
 
Furthermore, why does the UNet neural network perform well in medical image segmentation?
You can figure it out from [my answer](https://www.zhihu.com/question/269914775/answer/586501606) in Zhihu.

# UNet-family
## 2015
  * U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI) [[paper](https://arxiv.org/pdf/1505.04597.pdf)]  [[my-pytorch](https://github.com/ShawnBIT/UNet-family/blob/master/networks/UNet.py)][[keras](https://github.com/zhixuhao/unet)] 
## 2016 
  * V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation [[paper](http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf)] [[caffe](https://github.com/faustomilletari/VNet)][[pytorch](https://github.com/mattmacy/vnet.pytorch)]
  * 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation [[paper](https://arxiv.org/pdf/1606.06650.pdf)][[pytorch](https://github.com/wolny/pytorch-3dunet)]
## 2017 
  * H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes (IEEE Transactions on Medical Imaging)[[paper](https://arxiv.org/pdf/1709.07330.pdf)][[keras](https://github.com/xmengli999/H-DenseUNet)]
## 2018 
  * UNet++: A Nested U-Net Architecture for Medical Image Segmentation (MICCAI) [[paper](https://arxiv.org/pdf/1807.10165.pdf)][[my-pytorch](https://github.com/ShawnBIT/UNet-family/blob/master/networks/UNet_Nested.py)][[keras](https://github.com/MrGiovanni/UNetPlusPlus)]
  * MDU-Net: Multi-scale Densely Connected U-Net for biomedical image segmentation [[paper](https://arxiv.org/pdf/1812.00352.pdf)]
  * DUNet: A deformable network for retinal vessel segmentation [[paper](https://arxiv.org/pdf/1811.01206.pdf)]
  * RA-UNet: A hybrid deep attention-aware network to extract liver and tumor in CT scans [[paper](https://arxiv.org/pdf/1811.01328.pdf)]
  * Dense Multi-path U-Net for Ischemic Stroke Lesion Segmentation in Multiple Image Modalities [[paper](https://arxiv.org/pdf/1810.07003.pdf)]
  * Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment [[paper](https://arxiv.org/pdf/1812.01936.pdf)]
  * Prostate Segmentation using 2D Bridged U-net [[paper](https://arxiv.org/pdf/1807.04459.pdf)]
  * nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation [[paper](https://arxiv.org/pdf/1809.10486.pdf)][[pytorch](https://github.com/MIC-DKFZ/nnUNet)]
  * SUNet: a deep learning architecture for acute stroke lesion segmentation and
outcome prediction in multimodal MRI [[paper](https://arxiv.org/pdf/1810.13304.pdf)]
  * IVD-Net: Intervertebral disc localization and segmentation in MRI with a multi-modal UNet [[paper](https://arxiv.org/pdf/1811.08305.pdf)]
  * LADDERNET: Multi-Path Networks Based on U-Net for Medical Image Segmentation [[paper](https://arxiv.org/pdf/1810.07810.pdf)][[pytorch](https://github.com/juntang-zhuang/LadderNet)]
  * Glioma Segmentation with Cascaded Unet [[paper](https://arxiv.org/pdf/1810.04008.pdf)]
  * Attention U-Net: Learning Where to Look for the Pancreas [[paper](https://arxiv.org/pdf/1804.03999.pdf)]
  * Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation [[paper](https://arxiv.org/pdf/1802.06955.pdf)]
  * Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks [[paper]](https://arxiv.org/pdf/1803.02579.pdf)
  * A Probabilistic U-Net for Segmentation of Ambiguous Images (NIPS) [[paper](https://arxiv.org/pdf/1806.05034.pdf)] [[tensorflow](https://github.com/SimonKohl/probabilistic_unet)]
  * AnatomyNet: Deep Learning for Fast and Fully Automated Whole-volume Segmentation of Head and Neck Anatomy [[paper](https://arxiv.org/pdf/1808.05238.pdf)]
  * 3D RoI-aware U-Net for Accurate and Efficient Colorectal Cancer Segmentation [[paper](https://arxiv.org/pdf/1806.10342.pdf)][[pytorch](https://github.com/huangyjhust/3D-RU-Net)]
## 2019 
  * MultiResUNet : Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation [[paper](https://arxiv.org/pdf/1902.04049v1.pdf)][[keras](https://github.com/nibtehaz/MultiResUNet)]
  * U-NetPlus: A Modified Encoder-Decoder U-Net Architecture for Semantic and Instance Segmentation of Surgical Instrument [[paper](https://arxiv.org/pdf/1902.08994.pdf)]
  * Probability Map Guided Bi-directional Recurrent UNet for Pancreas Segmentation [[paper](https://arxiv.org/pdf/1903.00923.pdf)]
  * CE-Net: Context Encoder Network for 2D Medical Image Segmentation [[paper](https://arxiv.org/pdf/1903.02740.pdf)][[pytorch](https://github.com/Guzaiwang/CE-Net)]
  * Graph U-Net [[paper](https://openreview.net/pdf?id=HJePRoAct7)]
  * A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation (ISBI) [[paper](https://arxiv.org/pdf/1810.07842.pdf)]
  * ST-UNet: A Spatio-Temporal U-Network for Graph-structured Time Series Modeling [[paper](https://arxiv.org/pdf/1903.05631.pdf)]
  * Connection Sensitive Attention U-NET for Accurate Retinal Vessel Segmentation [[paper](https://arxiv.org/pdf/1903.05558.pdf)]
  * CIA-Net: Robust Nuclei Instance Segmentation with Contour-aware Information Aggregation [[paper](https://arxiv.org/pdf/1903.05358.pdf)]
  * W-Net: Reinforced U-Net for Density Map Estimation [[paper](https://arxiv.org/pdf/1903.11249.pdf)]
  


# Reference
  * https://github.com/ozan-oktay/Attention-Gated-Networks (Our model style mainly refers to this repository.）
