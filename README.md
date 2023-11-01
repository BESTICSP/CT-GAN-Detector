# CT-GAN-Detector
## GAN-based Medical Image Small Region Forgery Detection via a Two-Stage Cascade Framework

Using generative adversarial network (GAN) for data enhancement of medical images is significantly helpful for many computer-aided diagnosis (CAD) tasks. A new attack called CT-GAN has emerged. It can inject or remove lung cancer lesions to CT scans. Because the tampering region may even account for less than 1% of the original image, even state-of-the-art methods are challenging to detect the traces of such tampering. In order to solve above-mentioned problem. we proposes a cascade framework to detect GAN-based medical image small region forgery like CT-GAN. In the local detection stage, we train the detector network with small sub-images so that interference information in authentic regions will not affect the detector. We use depthwise separable convolution and residual to prevent the detector from over-fitting and enhance the ability to find forged regions through the attention mechanism. The detection results of all sub-images in the same image will be combined into a heatmap. In the global classification stage, using gray level co-occurrence matrix (GLCM) can better extract features of the heatmap. Because the shape and size of the tampered area are uncertain, we train PCA and SVM methods for classification. Our method can classify whether a CT image has been tampered and locate the tampered position. Sufficient experiments show that our method can achieve excellent performance.

## Experimental environment

CUDA and cuDNN versions are as follows:

|       | version  |
|  ---  | ---   |
| CUDA  | 10.0  |
| cuDNN | 7.4.1 |

Use the following code to install the required packages:

```pip install -r requirements.txt```

## How to run

### Data

Please use CT-GAN(https://github.com/ymirsky/CT-GAN) to generate forged samples in ```.dcm``` format. For each fake sample, we select the tampered point and two slices before and after it, five CT slices, and the corresponding five slices before tampering. Then randomly selected about half of the real CT slice images and replaced them with slices at random locations.

The file is named in the following format:

```inject_%04d_%03dT.dcm```
```inject_%04d_%03dF.dcm```
```remove_%04d_%03dT.dcm```
```remove_%04d_%03dT.dcm```

for example:```inject_0155_225F.dcm```

Take the four digit integer as the number of CT scan and the three digit integer as the number of slice image. "T" means the real image (negative class) and "F" means the forged image (positive class).

use .csv file to record the coordinates and corresponding file name of each tamper. for example:

| filename | x | y | z |
|---|---|---|---|
|inject_0000_158|176|183|158|
|inject_0001_195|113|119|195|

Replace the corresponding path in generate_tfrecord_my.py with your data path and .csv file path.

### Path

Create the following folders under this project: 

```
checkpoint
ml_model
cv_tfrecord
test_tfrecord
train_tfrecord
origin_tfrecord
npy_data
npy_slice
step_accuracy
```

Replace the corresponding path in generate_tfrecord_my.py with the above folder path.
(They may require a lot of storage space, if you want to save this data in other path, please replace the path in generate_tfrecord_my.py, train.py, sliding_window_test_my.py, svm_pca_classifier.py)

### RUN

Run the following files in sequence: 
```
generate_tfrecord_my.py
train.py
sliding_window_test_my.py
svm_pca_classifier.py
```

### Reviewer Comments
PLOS ONE
Reviewer #1: This paper has successfully proposed a novel two-stage cascade framework based on a local detection network and a global classification method that can detect GAN-based small region forgery attacks in medical images....Sufficient experiments show that this method can achieve excellent
performance than the state-of-the-art detection methods... The authors have done good work. It will add new knowledge and new areas of research to the subject area compared with other published material.
Reviewer #2:Experiments show that the proposed method can obtain better results.

### E-mail

Programmer: Xuanxi Huang, Supervisor: Jianyi Zhang

Email: zjy@besti.edu.cn

Zhang, Jianyi, Xuanxi Huang, Yaqi Liu, Yuyang Han, and Zixiao Xiang. "GAN-based Medical Image Small Region Forgery Detection via a Two-Stage Cascade Framework."PLOS One, 2023.

北京电子科技学院CSP实验室
