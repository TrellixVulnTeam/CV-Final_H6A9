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
| 训练数据集 | 权值文件名称 | 测试数据集 | mAP 0.5 |
| :-----: | :-----: | :------: | :-----: |
| VOC07+12 | best_epoch_weights_resnet.pth | VOC-Test07 | 78.67 |
| VOC07+12 | best_epoch_weights_vgg.pth | VOC-Test07 | 79.41 |

## 所需环境
torch == 1.2.0

## 文件下载
我们小组训练的模型可以在下面的链接下载

链接：https://pan.baidu.com/s/1E__c41UAsqllSkNNmT_TfQ?pwd=a2s6

原仓库：

训练所需的voc_weights_resnet.pth或者voc_weights_vgg.pth以及主干的网络权重可以在百度云下载。  
voc_weights_resnet.pth是resnet为主干特征提取网络用到的；  
voc_weights_vgg.pth是vgg为主干特征提取网络用到的；   
链接: https://pan.baidu.com/s/1S6wG8sEXBeoSec95NZxmlQ      
提取码: 8mgp    

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A    
提取码: uack   

## 训练步骤
### 训练VOC07+12数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 数据集的处理   
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   

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
