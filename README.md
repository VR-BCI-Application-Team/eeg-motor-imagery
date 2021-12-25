# EEG Motor Imagery
🍳 An EGG neural network for Motor Imagery task

**Notes:** ⚠️ *This document is under construction, please raise issues if you need anything.*

# The dataset
We've preprocessed the BCI IV 2a and trunked it into `event`.

# How to train
Install requirements
```shell
pip install -r requirements.txt
```

Training with argparse
```shell
python train.py --gpus 1 --batch_size 16 --lr 1e-3 --max_epochs 10
```

# Reference
[1] Roots, K.; Muhammad, Y.; Muhammad, N. Fusion Convolutional Neural Network for Cross-Subject 
EEG Motor Imagery Classification. Computers 2020, 9, 72. https://doi.org/10.3390/computers9030072
