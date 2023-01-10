# YOLOv3-tiny with Different Depths Implemented in TensorFlow 2.0
This repository builds and trains YOLOv3-tiny models with different depths in TensorFlow 2.0 and converts them into TensorFlow Lite models. These TF Lite models are benchmarked by the repository "https://gitlab.lrz.de/chair-of-cyber-physical-systems-in-production-engineering/nn-rt-bench.git" in order to explore the impact of the depth of YOLOv3-tiny on the real-time features (inference time, accuracy and memory usage) of inference. Relevant results and  analysis can be found in Paper "__Benchmarking Real-time Features of Deep Neural Network Inferences__".

This repository is an extension of the repository "https://github.com/zzh8829/yolov3-tf2". The latter provides a clean implementation of the original YOLOv3/YOLOv3-tiny. In addition, it also provides the methods for training and converting obtained TensorFlow models to TF Lite models. Familiarity with this repository is a prerequisite for understanding the subsequent content.
<br/>

## Installation
###  Conda Environment
```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov3-tf2-cpu
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```

###  Nvidia Driver (For GPU)
```bash
# Ubuntu 18.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
<br/>

## Inference with pre-trained weights
### Convert pre-trained Darknet weights
```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```

### Detection
```bash
# yolov3
python detect.py --weights ./checkpoints/yolov3.tf --image ./data/street.jpg
# yolov3-tiny
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./data/street.jpg
```
<br/>

## YOLOv3-tiny with Different Depths
The repository "https://github.com/zzh8829/yolov3-tf2" built the original YOLOv3-tiny. We changed the depth of YOLOv3-tiny by duplicating and adding the first five _Convolutional_ modules (__Conv. Layer 1__ ∼ __Conv. Layer 5__) in Darknet-tiny (backbone of YOLOv3-tiny). The duplicated and added _Convolutional_ modules are placed next to the original _Convolutional_ module. In addition, we also limited the growth of the depth to 0 ∼ 3 layers. Based on the above rules, we can obtain 56 models through different combinations between __Conv. Layer 1__ ∼ __Conv. Layer 5__. You can find these modified models and the original model under the folder "./yolov3_tf2".
We use the "XXXXX" five-digit code to distinguish modified YOLOv3-tiny models. From right to left, each coded digit corresponds to the module __Conv. Layer 5__ ∼ __Conv. Layer 1__. The number "X" on a certain digit means that we duplicated and added its corresponding _Convolutional_ module X times. For example, "10101" means that we duplicated and added __Conv. Layer 1__, __Conv. Layer 3__ and __Conv. Layer 5__ respectively, and each added _Convolutional_ module is placed next to the original _Convolutional_ module.  Further, "02010" means that we duplicated and added __Conv. Layer 2__ twice, __Conv. Layer 4__ once, and "00000" means that we did not modify the structure of YOLOv3-tiny, but will train it by ourselves without using the pre-trained weights.
You can get a more detailed explanation in Section 4.3.1 of Paper "__Benchmarking Real-time Features of Deep Neural Network Inferences__".
<br/>

## Train with COCO2017 Dataset
The repository "https://github.com/zzh8829/yolov3-tf2" provides the method for training from scratch or transfer training using VOC2012 Dataset. Based on it, this repository provides the method for training with COCO2017 Dataset.

### Download COCO2017 Dataset
Place COCO2017 Dataset into the folder "./data/coco2voc2tfrecord".

### Convert COCO2017 to ASCAL VOC Format 
Perform the following operations on the COCO2017 train and validation datasets respectively.
```bash
cd ./tools/coco2voc
# select the 80 categories actually contained by COCO2017
python select_categories.py
# remove the images that don’t contain any objects of the 80 categories
python remove.py
# convert .json to .xml
python create_xml.py
# create image list
python create_list.py
```
Organise all the obtained files/folders in the form of VOC for train and validation datasets respectively.

### Transform Dataset to TFRecord Format
```bash
# train dataset
python tools/voc2012.py \
  --data_dir './data/coco2voc2tfrecord/train2017' \
  --split train \
  --output_file ./data/coco2017_train.tfrecord
# validation dataset
python tools/voc2012.py \
  --data_dir './data/coco2voc2tfrecord/val2017' \
  --split val \
  --output_file ./data/coco2017_val.tfrecord
```

### Train
We created a corresponding training program for each model, which can be found in the folder "./train". 

```bash
cd ./scripts
bash training_from_scratch.sh
```
During training, the weights after each Epoch will be preserved in folder "./scripts/checkpoints"

### Test
```bash
python detect_coco2017_val.py
```
All detection results for each image will be recorded in the folder "./results".
<br/>

## TensorFlow Lite
### TensorFlow Lite Converter
You can convert these post-trained models to TF Lite format.
```bash
cd ./tools
python export_tflite.py
```

### Test
```bash
python ./tools/test_tflite.py
python ./tools/test_tflite_coco2017_val.py
```