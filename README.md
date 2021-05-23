# 2021-ESUN-AI-Competition

This is a Chinese handwriting recognition project. The dataset is provided by Esun Bank.

## Data Prepared

You can get the dataset by signing in the [website](https://tbrain.trendmicro.com.tw/Competitions/Details/14), which is provided by Trend Micro.

You should put the data and code as the structure below.

```
├── data
│   └── clean
│       ├── 0_戶.jpg
│       ├── 1_經.jpg
│       └── ...
├── models  // optional, if you don't want to train the model by yourself
│   ├── efficientnet-b3_epoch100.pth
│   └── ...
├─ Data Analysis.ipynb
├─ Data Preprocessing.ipynb
├─ Evaluation.ipynb
└─ FineTuning ResNet18.ipynb
```
Run **Data Preprocessing.ipynb** to split training data into training data and validation data.

## Data Analysis

Run **Data Analysis.ipynb** to display some feature from the data.

## FineTuning and Evaluation
  
EfficientNet-B3

Run **FineTuning EfficientNet-B3.ipynb** to finetune EfficientNet-B3 with pretrained weight (pretrain via ImageNet).

## Evaluation

You must to have model in directory "models".

Run **Evaluation.ipynb** to evaluate some cases of the validation data.

## References

- [Transfer Learning For Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
