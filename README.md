# 2021-ESUN-AI-Competition

This is a Chinese handwriting recognition project. The dataset is provided by Esun Bank.

## Installation

Assume you have a container with GPU and Cuda version >= 11.1
   
Install the requirements by `pip install -r requirements.txt`

## Code Structure

You should put the code as the structure below.

```
├── data  // optional, you can put your data wherever you want
│   ├── clean v2
│   │   ├── 0_戶.jpg
│   │   ├── 1_經.jpg
│   │   └── ...
│   └── training data dic.txt
├── lib
│   ├── dataset.py
│   ├── model.py
│   └── options.py
│
├── models  // optional, you can put your models wherever you want
│   ├── efficientnet-b0.pth
│   └── ...
├─ main.py
├─ README.md
└─ requirements.txt
```

## FineTuning EfficientNet

To list the arguments, run the following command:
```
python main.py -h
```

### Training on custom data

To train the model on custom data, you can modify the following command and run it.

For more training options, please check the arguments.

```bash
python main.py \
 --dataroot </your/image/dir> \
 --n_epoch <number of epoch> \
 --model_name <efficientnet-bx> \
 --load_weights_path </your/model/path>        
```

For example:

```bash
python main.py \
 --dataroot ./data/clean v2 \
 --n_epoch 10 \
 --model_name efficientnet-b0 \
 --load_weights_path ./models/efficientnet-b0.pth        
```

## References

- [Transfer Learning For Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
