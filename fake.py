import librosa
import numpy as np
from network import *

# print("[+] opening mp3 file")
# x, sr = librosa.load('voice.mp3')
# data = np.append(x, [0 for i in range(22050 - (len(x) % sr))])

# data = data[:-4*22050] 

# data = data.reshape(int(len(data) / 22050), 22050)
# datalen = len(data)
# print(f"[+] Parsed {datalen} objects to train")


data = np.random.randn(10, 22050)



network = [
    Dense(22050, 1000),
    Lower(0.01),
    Dense(1000, 200),
    Lower(0.01),
    Dense(200, 1000),
    Lower(0.01),
    Dense(1000, 22050),
    Lower(0.01)
]

epoch = 50
lr = 0.1


for e in range(epoch):
    error = 0
    for veri in data:
        output = np.array([veri]).T

        for layer in network:
            output = layer.ileri(output)


        error += square( np.array([veri]).T, output)

        grad = square_prime( np.array([veri]).T, output)[0]

        for layer in reversed(network):
            grad = layer.geri(grad, lr)
        
    error /= 10
    print(f"[+] {e + 1} / {epoch} error: {error}")

