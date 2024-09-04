import cv2
import numpy as np
import json
import os
import imageio
import random
from tqdm import tqdm
import argparse

def latent_to_vid_in_dir(latent_dir):
    # use os.walk to get all latent files
    # latent_files = []
    for root, dirs, files in os.walk(latent_dir):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                latent_to_vid(file_path)


def latent_to_vid(latent_path):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=int, default=1)
    args = parser.parse_args()
    latent_to_vid_in_dir(args.latent_dir)