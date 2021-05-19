# 2021-ESUN-AI-Competition

This is a Chinese handwriting recognition project. The dataset is provided by Esun Bank.

## Data Prepared

You can get the dataset by signing in the [website](https://tbrain.trendmicro.com.tw/Competitions/Details/14), which is provided by Trend Micro.

You should put the data and code as the structure below.

```
├── data
│   │── all
│   │   ├── 0_戶.jpg
│   │   ├── 1_經.jpg
│   │   └── ...
│   └── clean
│       ├── 0_戶.jpg
│       ├── 1_經.jpg
│       └── ...
├── model  // optional, if you want to train the model by yourself
│   ├── efficientnet-b3_epoch100.pth
│   └── ...
├─ Data Analysis.ipynb
├─ Data Preprocessing.ipynb
├─ FineTuning ResNet18.ipynb
└─ training data dict.txt
```
Run **Data Preprocessing.ipynb** to split training data into training data and validation data.

## Data Analysis

Run **Data Analysis.ipynb** to display some feature from the data.

## FineTuning and Evaluation

- ResNet18

  Run **FineTuning ResNet18.ipynb** to finetune ResNet18 with pretrained weight (pretrain via ImageNet).
  
- EfficientNet-B7

  Run **FineTuning EfficientNet-B7.ipynb** to finetune EfficientNet-B7 with pretrained weight (pretrain via ImageNet).

## Evaluation

Run **Evaluation.ipynb** to evaluate some cases of the validation data.

## References

- [Transfer Learning For Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
