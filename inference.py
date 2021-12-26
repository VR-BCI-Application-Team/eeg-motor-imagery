import os
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
    input = torch.from_numpy(data)
    if len(input.shape) == 2:
        input = input[None, None, :]
    elif len(input.shape) == 3:
        input = input.unsqueeze(1)

    pred = model(input.float()).squeeze().detach().numpy()

    if pred.ndim <= 0:
        pred = [pred]

    res = []
    for item in pred:
        if item > hi_thres:
            res.append(1)
        elif item < lo_thres:
            res.append(0)
        else: 
            res.append(-1)
    
    return res


def emit(name='signal', value: int = -1):
    print('VALUE', value)
    sio.emit(name, {'direction': value})


if __name__ == "__main__":
    opt = args()
    # TODO: load input and preprocess sequence data
    if os.path.isfile(opt.data_path):
        data = np.load(opt.data_path)  # demo
    elif os.path.isdir(opt.data_path):
        files = os.listdir(opt.data_path)
        data = []
        for file in files:
            data.append(np.load(os.path.join(opt.data_path, file)))
        data = np.array(data)
    else:
        raise Exception('`data_path` are not specific')

    model = Classifier()
    model.load_from_checkpoint(opt.model_path)

    # # predict
    pred = inference(model, data, opt.lo_thres, opt.hi_thres)

    print(len(pred))
    
    for timestamp in pred:
        emit(value = int(timestamp))
