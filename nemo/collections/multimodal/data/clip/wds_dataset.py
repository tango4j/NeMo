import random
import boto3
import json
import pickle
import os
import io
import re
import itertools
import webdataset as wds
from webdataset import warn_and_continue
from webdataset.filters import _shuffle

import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset
from botocore.config import Config
from PIL import Image
from nemo.collections.multimodal.data.clip.data_samplers import SharedEpoch
from nemo.collections.multimodal.data.clip.wds_utils import WebDataset
from nemo.utils import logging

try:
    from apex.transformer import parallel_state
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

Image.MAX_IMAGE_PIXELS = 933120000
_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()

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


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        # This seed to be deterministic AND the same across all nodes/workers in each epoch
        if parallel_state.is_unitialized():
            seed = self.seed + epoch
        else:
            seed = self.seed + epoch + (100 * parallel_state.get_data_parallel_rank())
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


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
            data_path = data_cfg.train.data_path
        else:
            data_path = data_cfg.validation.data_path

        self.urls = []
        self.chunk_size = None
        self.total_key_count = 0

        if data_path[0].endswith(".pkl"):
            # Concatenate all dataset infos
            # Create an url list of tar files

            for ds_info_path in data_path:
                with open(ds_info_path, 'rb') as fp:
                    ds_info = pickle.load(fp)
                    self.urls.extend(ds_info['tar_files'])
                    if self.chunk_size is None:
                        self.chunk_size = ds_info['chunk_size']
                    else:
                        assert self.chunk_size == ds_info['chunk_size'], \
                            "Chunk size needs to be consistent across different shards."
        else:
            self.urls = map(wds.shardlists.expand_urls, data_path)
            self.urls = list(itertools.chain.from_iterable(a))
            self.chunk_size = data_cfg.webdataset.get("chunk_size", 1000)

        self.total_key_count += self.chunk_size * len(self.urls)
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
    def __init__(self, data_cfg, url_dataset,
                 image_transform, text_transform,
                 shared_epoch=None, is_train=True):
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

        self.img_transform = image_transform
        self.text_transform = text_transform

        self.epoch = shared_epoch
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

        webdata_cfg = self.data_cfg.get("webdataset", {})
        self.dataset = (
            WebDataset(
                self.url_dataset,
                load_from_object_store=webdata_cfg.get("object_store"),
                s3_client=self.s3,
                s3_bucket_name=self.bucket,
                local_root_path=self.local_root_path,
                handler=warn_and_continue,
            )
            .compose(detshuffle2(bufsize=chunk_size, epoch=self.epoch))  # Shuffling the buffer
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