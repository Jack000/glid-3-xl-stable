import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms

def load_data(
    *,
    good_data_dir,
    bad_data_dir,
    batch_size,
    image_size,
    val_split=False,
    val_num=0
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param good_data_dir: images that will be labelled "1"
    :param bad_data_dir: images that will be labelled "0"
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    """
    if not good_data_dir or not bad_data_dir:
        raise ValueError("unspecified data directory")

    good_files = _list_image_files_recursively(good_data_dir)
    bad_files = _list_image_files_recursively(bad_data_dir)

    print('found ', len(good_files),' good files')
    print('found ', len(bad_files),' bad files')

    if len(good_files) < len(bad_files):
        good_files *= math.floor(len(bad_files)/len(good_files))
        print('duplicated good files to ', len(good_files))
    elif len(bad_files) < len(good_files):
        bad_files *= math.floor(len(good_files)/len(bad_files))
        print('duplicated bad files to ', len(bad_files))

    all_data = [(d,0) for d in bad_files] + [(d,1) for d in good_files]
    random.Random(99).shuffle(all_data)

    if val_num > 0:
        if val_split:
            all_data = all_data[:val_num]
        else:
            all_data = all_data[val_num:]

    dataset = ImageDataset(
        image_size,
        all_data,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        entry = entry.split(".")
        ext = entry[-1].strip()
        filename = entry[0]
        if ext and ext.lower() in ["jpg", "jpeg", "png", "gif", "webp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        file_paths,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.resolution = resolution
        self.data = file_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pdata = self.data[idx]
        image_path = pdata[0]
        image_class = pdata[1]

        with bf.BlobFile(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        pil_image = pil_image.convert("RGB")

        arr = center_crop_arr(pil_image, self.resolution)

        arr = arr.astype(np.float32) / 127.5 - 1

        return np.transpose(arr, [2, 0, 1]), {'y':np.array(image_class, dtype=np.int64)}

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
