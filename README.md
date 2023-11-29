

#### 零、环境依赖

项目推理所有环境均为申请的算能 BM164x 云开发空间环境，主要版本如下：

- python 3.7.5
- torch 1.13.1+cpu


#### 一、解决方案简介

解决方案主要包括两块，第一是超分辨率模型研究，第二推理性能优化；

###### 1. 超分辨率模型

超分辨率模型选择为 Real-ESRGAN，但选择轻量级网络作为主干结构，使用提供的训练数据集进行微调。

###### 2. 推理性能优化

主要涉及以下几块优化点：

- 计算图优化
- 模型压缩，训练后量化
- 流水线并行
- 图像 tile 分块处理 
- batch 批预测
- 多线程图片保存

#### 二、运行步骤

###### 0. 准备数据

将提供的原始数据文件放在 dataset 文件夹下，文件结构如下：

```shell
|-- dataset
    |-- test
        |-- 0001.png
        |-- ...
    |-- val
        |-- 0001.png
        |-- ...
```


###### 1. 模型转换为 bmodel

在 `FT_SUBMIT/models` 目录下，运行以下命令。该步骤执行成功后，可以在该目录下找到 `out.bmodel` 文件。

```shell
sh convert.sh
```


###### 2. 超分模型推理

在 `FT_SUBMIT` 目录下，运行以下命令，执行时间大约在 30 分钟左右。

```shell
python3 inference/detect.py

# 或者后台执行
# nohup python3 inference/detect.py > output.log 2>&1
```

###### 3. 结果校验

在 `FT_SUBMIT` 目录下，运行以下命令，主要检验以下 3 个条件，执行成功后提示 `Congratulations, the result is valid !!`。

```shell
python3 inference/check.py
```

1. 单张图片推理时间 `runtime` 不为 0；

  - 对输入图片进行分块，当 tile size 较小且 batch size 较大时，单个 tile 的运行时间可能很小，使用 round (4) 时容易为 0， 导致该幅图片的整体的 runtime 为 0；
  - 此处运行时间精度 round bug 很容易导致 score 偏高；

2. 推理图片数量正确，不缺不漏；

3. 图片尺寸符合要求的 x4；

