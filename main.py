import numpy as np
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from model import Net
from utils import *
import os

LR = [0.01, 0.1, 1.0]

def train(model, device, train_loader, optimizer, epoch, lr, p):
    
    model.train()
    
    fs = 44100
    duration = 0.01
    f = 200.0

    frames = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        norms = []
        for layer in model.ordered_layers:
            norm_grad = layer.weight.grad.norm()
            norms.append(norm_grad)

            tone = f + ((norm_grad.numpy()) * 100.0)
            tone = tone.astype(np.float32)
            samples = generate_tone(fs, tone, duration)

            frames.append(samples)

        silence = np.zeros(samples.shape[0] * 2,
                           dtype=np.float32)
        frames.append(silence)

        optimizer.step()

        # Just 200 steps per epoach
        if batch_idx == 200:
            break

    wf = wave.open("training_audio/sgd_" + str(lr) + ".wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Saved audio file for SGD LR:", lr)


def main():
    device = torch.device("cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=256, shuffle=True)

    model = Net().to(device)

    fs = 44100
    duration = 0.01
    f = 200.0
    p, stream = open_stream(fs)

    if os.path.isdir("training_audio") == False:
        os.mkdir("./training_audio")
    
    for lr in LR:
        print("Training NN with LR:", lr)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

        for epoch in range(1, 2):
            train(model, device, train_loader, optimizer, epoch, lr, p)

    stream.stop_stream()
    stream.close()
    p.terminate()
    

if __name__ == "__main__":
    main()