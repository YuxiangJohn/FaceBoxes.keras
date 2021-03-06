## FaceBoxes implementation in Keras
---
Status: just started, fixing bugs, wanting helps

### Overview

This is a Keras port of the FaceBoxes architecture introduced by Shifeng Zhang et al. in the paper [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/pdf/1708.05234.pdf).

<p align="left">
<img src="https://github.com/sfzhang15/FaceBoxes/blob/master/faceboxes_framework.jpg" alt="FaceBoxes Framework" width="777px">
</p>

### Content

The project refers to the SSD framework:  [SSD_keras](https://github.com/pierluigiferrari/ssd_keras)

The first implementation is without density.

The faceboxes code are written in ./models/FaceBoxes.py. The training process and the setting of parameters are in ./FaceBoxes_train.py.
