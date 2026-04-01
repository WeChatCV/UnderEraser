import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from random import shuffle
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, Camera, VideoReader_contextmanager,
                    custom_meshgrid, get_random_mask, get_relative_pose,
                    get_video_reader_batch, padding_image, process_pose_file,
                    process_pose_params, ray_condition, resize_frame,
                    resize_image_with_target_area)

class EditedVideoDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_n_frames=16,
        video_sample_size=512,
        video_sample_stride=None,
        video_repeat=0,
        image_sample_size=512,
        enable_inpaint=True,
        return_file_name=False,
        enable_bucket=False,
    ):
        # ... (Initialization logic remains the same) ...
        import os, json, csv
        self.data_root = data_root
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_stride = video_sample_stride
        self.video_repeat = video_repeat
        self.image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.larger_side_of_image_and_video = video_sample_size
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name
        self.enable_bucket = enable_bucket 


        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as f:
                dataset = list(csv.DictReader(f))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        else:
            raise ValueError("Unsupported annotation file format.")


        self.dataset = []
        for data in dataset:
            self.dataset.append(data) 
        for _ in range(video_repeat):
            for data in dataset:
                self.dataset.append(data)


        self.video_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        edited_path   = data_info["edited"]
        masked_path   = data_info["masked"]
        unedited_path = data_info["unedited"]

        if self.data_root is not None:
            edited_path   = os.path.join(self.data_root, edited_path)
            masked_path   = os.path.join(self.data_root, masked_path)
            unedited_path = os.path.join(self.data_root, unedited_path)


        def get_frames_idx(video_len):
            if self.video_sample_stride is None:
                return np.linspace(0, video_len-1, self.video_sample_n_frames, dtype=int)
            else:
                idxs = np.arange(0, video_len, self.video_sample_stride)
                if len(idxs) >= self.video_sample_n_frames:
                    return idxs[:self.video_sample_n_frames]
                else:
                    return np.linspace(0, video_len-1, self.video_sample_n_frames, dtype=int)


        with VideoReader_contextmanager(edited_path, num_threads=2) as vr:
            frames_idx = get_frames_idx(len(vr))
            edited_frames = vr.get_batch(frames_idx).asnumpy()
            edited_frames = np.array([resize_frame(f, self.larger_side_of_image_and_video) for f in edited_frames])
            

            edited_frames = torch.from_numpy(edited_frames).permute(0,3,1,2)/255.
            

            pixel_values = self.video_transforms(edited_frames.clone()) 


        with VideoReader_contextmanager(unedited_path, num_threads=2) as vr:
            un_frames = vr.get_batch(frames_idx).asnumpy()
            un_frames = np.array([resize_frame(f, self.larger_side_of_image_and_video) for f in un_frames])
            

            unedit_tensor = torch.from_numpy(un_frames).permute(0,3,1,2)/255.
            

            mask_pixel_values = self.video_transforms(unedit_tensor.clone()) 


        with VideoReader_contextmanager(masked_path, num_threads=2) as vr:
            mask_frames = vr.get_batch(frames_idx).asnumpy()
            mask_frames = np.array([resize_frame(f, self.larger_side_of_image_and_video) for f in mask_frames])
            

            mask_gray = np.mean(mask_frames, axis=-1)
            mask = (mask_gray > 127).astype(np.float32) 
            mask = torch.from_numpy(mask).unsqueeze(1)  # [T, 1, H, W]


        masked_input_tensor = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
        

        clip_pixel_values = masked_input_tensor * 0.5 + 0.5
        clip_pixel_values = clip_pixel_values * 255

        clip_pixel_values = clip_pixel_values.permute(0, 2, 3, 1).contiguous()


        sample = {
            "pixel_values": pixel_values,
            "clip_pixel_values": clip_pixel_values,
            "mask_pixel_values": mask_pixel_values,
            "mask": mask,
            "idx": idx,
            "text": '' 
        }

        if self.return_file_name:
            sample["file_name"] = os.path.basename(edited_path)

        return sample




from torch.utils.data import DataLoader, Sampler, BatchSampler
import random

class ImageVideoSampler(BatchSampler):

    def __init__(self, sampler: Sampler, dataset: Dataset, batch_size: int, drop_last: bool = False):
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of Sampler')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer')
        
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket = {'image': [], 'video': []} 

    def __iter__(self):
        for idx in self.sampler:
            data_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[data_type].append(idx)


            if len(self.bucket[data_type]) == self.batch_size:
                batch = self.bucket[data_type][:]
                yield batch
                del self.bucket[data_type][:]

    def __len__(self):
        return len(self.dataset) // self.batch_size



