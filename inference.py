import socketio
import numpy as np

# connect
sio = socketio.Client()
@sio.event
def connect():
    print("I'm connected!")
# sio.connect('http://localhost:8000')
sio.connect('http://13.212.87.101:8000/')


def inference():
    # TODO: Left = 0; Right = 1; Other = -1
    pred = 0
    pass


def emit(name='signal', value: int = -1):
    sio.emit(name, {'direction': value})

if __name__ == "__main__":
    # TODO: load data


    # predict
    pred = inference()
    # pred = np.random.randint(-1, 2, size=100)

    # TODO: send to server
    for timestamp in pred:
        print(timestamp)
        emit(value = int(timestamp))
