# PRNet-Paddle
A reproduction of PRNet by PaddlePaddle

**快速使用参见[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/3615494?shared=1)**

## 数据集准备
```
/home/aistudio
|-- datasets
|   |-- train
|   |   |-- RainTrainH
|   |   |-- RainTrainL
|   |   |-- Rain12600
|   |-- test
|   |   |-- Rain100H
|   |   |-- Rain100L
|   |   |-- Rain1400
|   |   |-- test12
```


## 训练
**`.sh`文件包含了大量训练和测试脚本，部分可能需要修改路径** 或 单个使用如下命令

```
python train_PReNet.py --preprocess True --save_path logs/Rain100H/PReNet --data_path ../datasets/train/RainTrainH
```

**注意到：--preprocess会处理指定数据集并在相应数据集目录生成`.h5`文件，如果第一次运行后，后面可以不用指定**

**本仓库端到端训练了部分网络和数据集，全部文件保存在logs目录下，例如训练日志（包括文本和visualdl），以PReNet为例**
```
visualdl --logdir ./logs/Rain100H/PReNet
```

<center><img src="https://user-images.githubusercontent.com/49911294/162574926-3d176a1e-6df8-4d73-ad5d-38853e90a567.png" width="400"/><img src="https://user-images.githubusercontent.com/49911294/162574924-37fa1946-b483-440b-9291-bf485e8dd392.png" width="400"/></center>

## 测试
**`.sh`文件包含了大量训练和测试脚本，部分可能需要修改路径** 或 单个使用如下命令
```
python test_PReNet.py --logdir logs/Rain100H/PReNet --save_path results/Rain100H/PReNet --data_path ../datasets/test/Rain100H/rainy
```
**生成的结果保存在results相应文件夹，例如PReNet/results/Rain100H/PReNet**

## 评估

评估脚本在`statistic`文件下，**原始评估脚本为`Matlab`代码，本仓库将其转换为`Python`代码，参见`evaluate.py`**
```
cd statistic
python evaluate.py --gt_path ../../datasets/test/Rain100H/ --pred_path ../results/Rain100H/PReNet/
```
```
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:34<00:00,  1.06it/s]
SSIM: 0.899107845008588 PSNR: 29.49246224614776
```

**依次为雨图，原图，推理图**

<img src="https://user-images.githubusercontent.com/49911294/162574848-5eb5caa7-7895-4745-b5ed-ff4b67e30e2f.png" width="300"/>
<img src="https://user-images.githubusercontent.com/49911294/162574855-a868ec05-3366-49a8-b8a1-e8bf7b672d1b.png" width="300"/>
<img src="https://user-images.githubusercontent.com/49911294/162574839-ef2ead60-9715-462c-8ffc-fe58c80dd6b4.png" width="300"/>

**实现结果：SSIM: 0.899107845008588 PSNR: 29.49246224614776**
