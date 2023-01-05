import random
import boto3
import json
import pickle
import os
import io
import re
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset
from botocore.config import Config
from PIL import Image
from nemo.collections.multimodal.data.clip.wds_utils import WebDataset
from nemo.collections.multimodal.data.clip.clip_augment import image_transform
from nemo.utils import logging

try:
    from apex.transformer import parallel_state
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

Image.MAX_IMAGE_PIXELS = 933120000
_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()

def identical_transform(x):
    return x

def pil_loader(key, data):
    r"""
    Function to load an image.
    If the image is corrupt, it returns a black image.
    Args:
        key: Image key.
        data: Image data stream.
    """

    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _IMG_EXTENSIONS:
        return None

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        img = img.convert("RGB")

    return img


class WebDatasetUrls(Dataset):
    def __init__(self, data_cfg, is_train=True):
        r"""
        Webdataloader class
        Args:
            data_cfg: Dataset Config
            is_train (bool): Is the dataset used in training mode?
        """
        super().__init__()

        self.data_cfg = data_cfg
        self.webdata_cfg = data_cfg.webdataset
        if is_train:
            dataset_info = data_cfg.train.dataset_info
            self.batch_size = self.data_cfg.train.batch_size
            self.augmentations = self.data_cfg.train.augmentations
        else:
            dataset_info = data_cfg.val.dataset_info
            self.batch_size = self.val.batch_size
            self.augmentations = self.data_cfg.val.augmentations

        # Concatenate all dataset infos
        # Create an url list of tar files
        self.urls = []
        self.chunk_size = None
        self.total_key_count = 0

        for ds_info_path in dataset_info:
            with open(ds_info_path, 'rb') as fp:
                ds_info = pickle.load(fp)
                self.urls.extend(ds_info['tar_files'])
                self.total_key_count += ds_info['total_key_count']
                if self.chunk_size is None:
                    self.chunk_size = ds_info['chunk_size']
                else:
                    assert self.chunk_size == ds_info['chunk_size'], \
                        "Chunk size needs to be consistent across different shards."

        assert self.total_key_count > 0, "No WebDataset data is found."

    def __getitem__(self, index: int) -> str:
        """
        Args:
            index (int): Index
        Returns:
            string: a tar file url
        """
        return self.urls[index]

    def __len__(self) -> int:
        return len(self.urls)


class WDSDataset(IterableDataset):
    def __init__(self, data_cfg, url_dataset, is_train=True):
        r"""
        Webdataloader class
        Args:
            data_cfg (DictConfig): Dataset config
            url_dataset (Dataset): an iterable dataset which will be consumed by WebDataset
            is_train (bool): Is the dataset used in training mode?
        """
        super().__init__()

        self.data_cfg = data_cfg
        self.url_dataset = url_dataset
        self.chunk_size = url_dataset.chunk_size
        self.dataset = None
        self.is_train = is_train

        webdata_cfg = data_cfg.get("webdataset", {})
        if webdata_cfg.get("object_store", False):
            # Initializing PBSS
            logging.info(f"Init PBSS using credentials file at {webdata_cfg.pbss_credentials_file}")
            self.use_object_store = True
            assert webdata_cfg.pbss_credentials_file is not None
            with open(webdata_cfg.pbss_credentials_file) as fin:
                self.credentials = json.load(fin)
            config = Config(connect_timeout=30,
                            signature_version="s3",
                            retries={"max_attempts": 999999})
            self.s3 = boto3.client("s3", **self.credentials, config=config)
            self.bucket = webdata_cfg.bucket
            self.local_root_path = None
        else:
            self.use_object_store = False
            self.s3 = None
            self.bucket = None
            self.local_root_path = webdata_cfg.local_root_path
            logging.info(f'Read Webdataset locally. Data stores at {self.local_root_path}')


        self.img_transform = image_transform(
            224,
            is_train=is_train,
            mean=None,
            std=None
        )
        self.text_transform = identical_transform
        self.build_dataset()

    def build_dataset(self):
        """See base class."""

        chunk_size = self.url_dataset.chunk_size

        # This function maps data that are tuples to dictionary.
        def tuple_to_dict(inp):
            for input in inp:
                out_dict = dict()
                out_dict['images'] = input[0]
                out_dict['captions'] = input[1]
                yield out_dict

        self.dataset = (
            WebDataset(
                self.url_dataset,
                load_from_object_store=self.data_cfg.get,
                s3_client=self.s3,
                s3_bucket_name=self.bucket,
                local_root_path=self.local_root_path,
            )
            .shuffle(chunk_size)  # Shuffling the buffer
            .decode(pil_loader)   # Decoding the data
            .to_tuple("jpg txt")  # Splitting into tuple
            .map_tuple(
                self.img_transform,
                self.text_transform
            )  # Augmentation
            .compose(tuple_to_dict)  # Converting tuple to data dict
        )

        # TODO (yuya): what's this?
        self.dataset.total_images = len(self.url_dataset) * self.chunk_size

        logging.info(f"Total number of training shards: {len(self.url_dataset)}")
        logging.info(f"Total training key count: {self.dataset.total_images}")


    def __iter__(self):
        return self.dataset.__iter__()

    def __len__(self):
        return len(self.url_dataset) * self.chunk_size