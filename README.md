# FDEP-Net

------

Enhancing Medical Image Registration with Fusion-Decomposition Enhanced Pyramid Network

## Introduction

------

Deformable registration is a crucial part for a wide range of medical image analysis tasks. Recently, various advanced registration models with pyramid strategy were proposed to enhance model's capability to handle large deformation. However, most of these networks focus on predicting the subfield of deformation field by warped moving feature and fixed feature and directly warp the moving feature from encoder for the next stage. They usually overlook the relation between these subfield and the final deformation field which may leads to inconsistency in the generation of deformation field. In this paper, we introduce a Fusion-Decomposition Enhanced Pyramid Network (FDEP-Net) for unsupervised non-rigid registration to maintain a consistency process to predict the deformation field and improve the robustness of pyramid strategy. Unlike most recent pyramid networks, our method enables the moving and fixed image features to perceive each other's spatial displacement before the warping operation. So we put forward the feature fusion-decomposition module to enhance the effectiveness of pyramid strategy. In addition, we introduce the enhance pyramid module to improve the consistency and robustness of predicting subfield from each stage of pyramid operation. By conducting extensive experiments on two public brain magnetic resonance imaging (MRI) datasets, we verify that the proposed FDEP-Net outperforms SOTA iterative-based methods and requires a relatively small memory-consuming for a pure convolution network.

## Environment

------

IDE: Pycharm  (version 2023.2.1)

OS: Ubuntu 22.04

GPU: RTX 3090

 All the dependencies are shown in `requirement.txt` and you can install them in your conda environment:

```
pip install -r requirements.txt
```

## Dataset

------

In this study, the proposed FDEP-Net was evaluated by the two public available datasets which can be downloaded on their official website, including OASIS(https://sites.wustl.edu/oasisbrains/datasets/) and IXI(https://brain-development.org/ixi-dataset/ ).

## Training

------

User can directly run the training script to reproduce our reported result by:

```
python training.py
```

## Information

------

The paper corresponding to this project has been submitted to *The Visual Computer*.