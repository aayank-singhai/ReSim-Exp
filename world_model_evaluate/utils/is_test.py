import os

from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3

UCF101_PATH = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/public_datasets/UCF-101"


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=10):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


print("Calculating Inception Score...")
import cv2
root = UCF101_PATH
max_sample = 20480
device = "cuda:0"
print("collecting videos...")
paths = []
for folder in os.listdir(root):
    for video in os.listdir(os.path.join(root, folder)):
        paths.append(os.path.join(root, folder, video))

paths.sort()
length = len(paths)
if max_sample < 0:
    max_sample = length
elif max_sample > length:
    print("Warning: max_sample ({}) is larger than the number of clips ({}). Changing to ({})... ".format(max_sample, length, length))
    max_sample = length

indices = np.arange(length)
np.random.shuffle(indices)

frames = []
for idx in tqdm(indices[:max_sample]):
    video = paths[idx]
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame)
        else:
            break
    cap.release()

print(inception_score(frames))