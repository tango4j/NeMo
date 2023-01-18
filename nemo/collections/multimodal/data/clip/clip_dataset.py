import torch
from functools import partial
from typing import Any, List, Union, Dict, Optional
from torch.utils.data import Dataset

from nemo.collections.multimodal.data.clip.wds_dataset import WebDatasetUrls, WDSDataset
from nemo.collections.multimodal.data.clip.wds_utils import RandomSamplerIterableDataset
from nemo.collections.multimodal.data.clip.data_samplers import WDSUrlsRandomSampler
from nemo.collections.multimodal.data.clip.clip_augment import image_transform
from nemo.collections.multimodal.data.clip.imagenet_zeroshot_data import openai_imagenet_template, imagenet_classnames
from nemo.collections.vision.data.megatron.image_folder import ImageFolder
from nemo.collections.vision.data.megatron.vit_dataset import RandomSeedDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

def tokenize(texts: Union[str, List[str]], tokenizer: Any, context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    tokenizer:
        Tokenizer loaded in NeMo NeMo
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    texts_is_str = False
    if isinstance(texts, str):
        texts = [texts]
        texts_is_str = True

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    all_tokens = [[bos_id] + tokenizer.text_to_ids(text) + [eos_id] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eos_id
        result[i, :len(tokens)] = torch.tensor(tokens)

    if texts_is_str:
        result = result[0]
    return result


def identical_transform(x):
    return x


def get_preprocess_fns(model_cfg, tokenizer=None):
    # Define transforms
    img_size = (model_cfg.vision.get("img_h"), model_cfg.vision.get("img_w"))
    img_mean = model_cfg.vision.get("img_mean")
    img_std = model_cfg.vision.get("img_std")
    train_image_transform = image_transform(
        img_size,
        is_train=True,
        mean=img_mean,
        std=img_std,
    )
    val_image_transform = image_transform(
        img_size,
        is_train=False,
        mean=img_mean,
        std=img_std,
    )

    text_transform = identical_transform
    if tokenizer is not None:
        text_transform = partial(
            tokenize,
            tokenizer=tokenizer,
            context_length=model_cfg.text.get("max_position_embeddings"),
        )
    return train_image_transform, val_image_transform, text_transform


def build_train_valid_datasets(
        model_cfg,
        consumed_samples,
        tokenizer=None,
):
    train_image_transform, val_image_transform, text_transform = get_preprocess_fns(model_cfg, tokenizer)
    data_cfg = model_cfg.data

    # Create a dataset of WebDataset Urls
    train_url_dataset = WebDatasetUrls(data_cfg)
    val_url_dataset = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("data_path"):
        val_url_dataset = WebDatasetUrls(data_cfg, is_train=False)

    # Create a random sampler to shard, shuffle and resume with Urls
    train_url_sampler = WDSUrlsRandomSampler(
        dataset=train_url_dataset,
        total_urls=len(train_url_dataset),
        chunk_size=train_url_dataset.chunk_size,
        consumed_samples=consumed_samples,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        drop_last=data_cfg.train.get("drop_last", True),
        data_sharding=data_cfg.train.get("data_sharding", True),
    )
    if val_url_dataset is not None:
        val_url_sampler = WDSUrlsRandomSampler(
            dataset=val_url_dataset,
            total_urls=len(val_url_dataset),
            chunk_size=val_url_dataset.chunk_size,
            consumed_samples=0,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=data_cfg.validation.get("drop_last", True),
            data_sharding=data_cfg.validation.get("data_sharding", True),
        )

    # Wrapping the Url dataset with the random sampler
    train_url_dataset = RandomSamplerIterableDataset(
        train_url_dataset,
        train_url_sampler,
        chunk_size=train_url_dataset.chunk_size,
    )

    if val_url_dataset is not None:
        val_url_dataset = RandomSamplerIterableDataset(
            val_url_dataset,
            val_url_sampler,
            chunk_size=val_url_dataset.chunk_size,
        )

    # Create the actual WebDataset IterableDataset
    # Expanding the url to actual samples with shuffling, decoding and transforming
    train_data = WDSDataset(
        data_cfg,
        train_url_dataset,
        image_transform=train_image_transform,
        text_transform=text_transform,
        is_train=True
    )

    val_data = None
    if val_url_dataset is not None:
        val_data = WDSDataset(
            data_cfg,
            train_url_dataset,
            image_transform=val_image_transform,
            text_transform=text_transform,
            is_train=False
        )

    return train_data, val_data


# For zero-shot imagenet validation
def build_imagenet_validation_dataloader(model_cfg, tokenizer=None):
    _, val_image_transform, text_transform = get_preprocess_fns(model_cfg, tokenizer)
    data_cfg = model_cfg.data

    imagenet_val = {}

    imagenet_path = data_cfg.get("imagenet_val")
    if imagenet_path is None:
        return None

    image_dataset = ImageFolder(
        root=imagenet_path,
        transform=val_image_transform,
    )
    # image_dataset = RandomSeedDataset(val_data)
    # image_batch_sampler = MegatronPretrainingBatchSampler(
    #     total_samples=len(image_dataset),
    #     consumed_samples=0,
    #     micro_batch_size=model_cfg.micro_batch_size,
    #     global_batch_size=model_cfg.global_batch_size,
    #     data_parallel_rank=parallel_state.get_data_parallel_rank(),
    #     data_parallel_size=parallel_state.get_data_parallel_world_size(),
    #     drop_last=False, # TODO (yuya): check this
    # )
    imagenet_val["images"] = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=model_cfg.micro_batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    text_dataset = ImagenetClassnameDataset(imagenet_classnames, openai_imagenet_template, text_transform)
    imagenet_val["texts"] = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=text_dataset.num_templates,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )
    return imagenet_val

class ImagenetClassnameDataset(Dataset):
    def __init__(self, classnames, templates, text_transform):
        self.num_templates = len(templates)
        self.samples = []
        for classname in classnames:
            texts = [template(classname) for template in templates]
            self.samples.extend(text_transform(texts))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)