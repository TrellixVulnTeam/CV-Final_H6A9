## Faster-Rcnn：Two-Stage目标检测模型在Pytorch当中的实现
---

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [评估步骤 How2eval](#评估步骤)
7. [参考资料 Reference](#Reference)



## 性能情况
| 训练数据集 | 实验                     | finetune | 权值文件名称                           | 测试数据集 | mAP 0.5 |
| :--------: | ------------------------ | -------- | -------------------------------------- | ---------- | :-----: |
|  VOC07+12  | ImageNet预训练模型       | $\surd$  | best_epoch_weights_ImageNet.pth        | VOC-Test07 |  68.7   |
|  VOC07+12  | COCO预训练Mask-R-CNN模型 | $\surd$  | best_epoch_weights_COCO.pth            | VOC-Test07 |  12.87  |
|  VOC07+12  | 随机初始化               |          | best_epoch_weights_random.pth          | VOC-Test07 |  35.75  |
|  VOC07+12  | 随机初始化               | $\surd$  | best_epoch_weights_random_finetune.pth | VOC-Test07 |  4.10   |

## 所需环境
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.6.0
torchvision==0.10.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0

## 文件下载
我们小组训练的模型可以在下面的链接下载

链接： https://pan.baidu.com/s/1LGlLeP3j9Ve31adkpTYiUg?pwd=bnqs 提取码: bnqs 

## 训练步骤
### 训练VOC07+12数据集
1. 数据集的准备   
  **本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 数据集的处理   
  修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   

  1）运行train_voc_pre.py 训练初始化ImageNet预训练模型

  2）运行train_coco_pre.py训练初始化COCO预训练模型

  3）运行train_not_pre.py训练随机初始化模型

  4）运行train_notprefinetune.py训练随机初始化模型(finetune)

4. 训练结果预测   
  训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。我们首先需要去frcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。   
  **model_path指向训练好的权值文件，在logs文件夹里。   
  classes_path指向检测类别所对应的txt。**   
  完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载frcnn_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在frcnn.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测，更改frcnn.py里的参数可以绘制rpn层输出的proposal bounding boxes.

## 评估步骤 
### 评估VOC07+12的测试集
1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在frcnn.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  

https://github.com/eriklindernoren/PyTorch-YOLOv3  

https://github.com/BobLiu20/YOLOv3_PyTorch  

https://github.com/bubbliiiing/faster-rcnn-pytorch
