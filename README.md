# TianChi-SteamPredict
## 阿里云天池比赛-工业蒸汽量预测
### 赛题背景
火力发电的基本原理是：燃料在燃烧时加热水生成蒸汽，蒸汽压力推动汽轮机旋转，然后汽轮机带动发电机旋转，产生电能。在这一系列的能量转化中，影响发电效率的核心是锅炉的燃烧效率，即燃料燃烧加热水产生高温高压蒸汽。锅炉的燃烧效率的影响因素很多，包括锅炉的可调参数，如燃烧给量，一二次风，引风，返料风，给水水量；以及锅炉的工况，比如锅炉床温、床压，炉膛温度、压力，过热器的温度等。
### 赛题描述
经脱敏后的锅炉传感器采集的数据（采集频率是分钟级别），根据锅炉的工况，预测产生的蒸汽量。
### 数据说明
数据分成训练数据（train.txt）和测试数据（test.txt），其中字段”V0”-“V37”，这38个字段是作为特征变量，”target”作为目标变量。选手利用训练数据训练出模型，预测测试数据的目标变量，排名结果依据预测结果的MSE（mean square error）。

## 1.文件说明
### 主程序 
+ [天池-蒸汽量预测.ipynb](https://github.com/TR-Wang/TianChi-SteamPredict/blob/main/%E5%A4%A9%E6%B1%A0-%E8%92%B8%E6%B1%BD%E9%87%8F%E9%A2%84%E6%B5%8B.ipynb) 
+ [pytorchtools.py](https://github.com/Bjarten/early-stopping-pytorch) （Early Stopping 程序）
### 训练好的网络与预测结果
[TrainedModels_&_Predictions(MSE=0.117)](https://github.com/TR-Wang/TianChi-SteamPredict/tree/main/TrainedModels_%26_Predictions(MSE%3D0.117)) 文件夹内含有四个训练好的网络，对 test.txt 中数据分别进行预测后取平均得到 Predictions.txt，线上测试结果 MSE=0.117。

## 2.模型说明
### 特征选择
+ 通过对比训练集与测试集的 38 个特征的分布，排除 "V5", "V9", "V11", "V17", "V22", "V28"。
+ 进行 min-max 归一化，并选择相关性高的 18 个特征。
### 神经网络结构
+ 神经网络含有 3 个隐藏层，分别有 100、50、10 个节点，均使用 ReLU 激活。输出层含有 1 个节点，输出层不激活。
+ 使用 Adam 加速优化，并加入 L2-正则化，正则化系数 0.005。
+ 初始学习速率为 0.001，每训练 10 次学习速率×0.9。
+ 使用 Early Stopping，若开发集上的损失在 20 次迭代中没有下降则停止训练（早停后返回 20 次迭代之前的模型）。
### 训练及预测
+ 训练时使用五折交叉训练，训练集平均损失在 1.1 左右，开发集平均损失在 1.06 左右。
+ 选择损失函数较小的若干个模型进行预测，对预测结果取平均。

## 3.所用程序包
```Python
numpy, pandas, matplotlib, torch, sklearn, math, datetime, pytorchtools
