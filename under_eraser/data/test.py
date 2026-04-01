# ---------------------------
# DataLoader 示例
# ---------------------------
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from dataset_videoedit import (EditedVideoDataset,
                            ImageVideoSampler)
# 创建 dataset
dataset = EditedVideoDataset(
    ann_path="/mnt/shangcephfs/mm-base-vision-ascend/dingmingliu/ROSE-Dataset/video_dataset_index.json",
    data_root="/path/to/videos",
    video_sample_n_frames=16,
    video_sample_size=512,
    enable_inpaint=True
)

# 创建 sampler
sampler = RandomSampler(dataset)
batch_sampler = ImageVideoSampler(sampler, dataset, batch_size=4, drop_last=True)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=4,  # 根据机器调整
    pin_memory=True
)

# ---------------------------
# 遍历示例
# ---------------------------
for batch in dataloader:
    # batch 是字典形式，每个 key 对应一个列表/张量 batch
    print(batch["pixel_values"].shape)
    print(batch["clip_pixel_values"].shape)
    print(batch["mask_pixel_values"].shape)
    print(batch["mask"].shape)
    break