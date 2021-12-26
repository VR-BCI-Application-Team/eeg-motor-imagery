import time
import socketio
import argparse

import torch
import numpy as np

from model import Classifier


# connect
sio = socketio.Client()
@sio.event
def connect():
    print("I'm connected!")
# sio.connect('http://localhost:8000')
sio.connect('http://13.212.87.101:8000/')


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='./model/pretrain_weight.ckpt')
    parser.add_argument('--lo_thres', type=float, default=0.4)
    parser.add_argument('--hi_thres', type=float, default=0.6)
    return parser.parse_args()


def inference(model, data, lo_thres=0.5, hi_thres=0.7):
    # Left = 0; Right = 1; Other = -1
    model.eval()
    input = torch.from_numpy(data)[None, None, :].float()
    pred = model(input).squeeze().detach().numpy()

    if pred > hi_thres:
        return 1
    elif pred < lo_thres:
        return 0
    else: 
        return -1


def emit(name='signal', value: int = -1):
    sio.emit(name, {'direction': value})


if __name__ == "__main__":
    opt = args()
    # TODO: load input data and model
    data = np.load(opt.data_path)  # demo

    model = Classifier()
    model.load_from_checkpoint(opt.model_path)

    # predict
    pred = inference(model, data, opt.lo_thres, opt.hi_thres)
    print('AFTER_THRES', pred)

    for timestamp in pred:
        emit(value = int(timestamp))
