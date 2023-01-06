from functools import partial
from typing import Any, Union, Dict, Optional

from nemo.collections.multimodal.data.clip.wds_dataset import WebDatasetUrls, WDSDataset
from nemo.collections.multimodal.data.clip.wds_utils import RandomSamplerIterableDataset
from nemo.collections.multimodal.data.clip.data_samplers import WDSUrlsRandomSampler
from nemo.collections.multimodal.data.clip.clip_augment import image_transform


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
    if isinstance(texts, str):
        texts = [texts]

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    all_tokens = [[bos_id] + _tokenizer.text_to_ids(text) + [eos_id] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def build_train_valid_datasets(model_cfg, consumed_samples, tokenizer):
    img_size = model_cfg.get("image_size")
    img_mean = model_cfg.get("image_mean")
    img_std = model_cfg.get("image_std")

    # Create a dataset of WebDataset Urls
    train_url_dataset = WebDatasetUrls(model_cfg.data)
    val_url_dataset = None
    if model_cfg.data.get("val") is not None and model_cfg.data.val.get("dataset_info"):
        val_url_dataset = WebDatasetUrls(model_cfg.data, is_train=False)

    # Create a random sampler to shard, shuffle and resume with Urls
    train_url_sampler = WDSUrlsRandomSampler(
        dataset=train_url_dataset,
        total_urls=len(train_url_dataset),
        chunk_size=train_url_dataset.chunk_size,
        consumed_samples=consumed_samples,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        drop_last=model_cfg.train.data.get("drop_last", True),
        data_sharding=model_cfg.train.data.get("data_sharding", True),
    )
    if val_url_dataset is not None:
        val_url_sampler = WDSUrlsRandomSampler(
            dataset=val_url_dataset,
            total_urls=len(val_url_dataset),
            chunk_size=val_url_dataset.chunk_size,
            consumed_samples=0,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=model_cfg.data.val.get("drop_last", True),
            data_sharding=model_cfg.data.val.get("data_sharding", True),
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

    text_transform = partial(
        tokenize,
        tokenizer=tokenizer,
        context_length=model_cfg.get("max_position_embeddings"),
    )

    # Create the actual WebDataset IterableDataset
    # Expanding the url to actual samples with shuffling, decoding and transforming
    train_data = WDSDataset(
        model_cfg.data,
        train_url_dataset,
        image_transform=image_transform(
                img_size,
                is_train=True,
                mean=img_mean,
                std=img_std,
            ),
        text_transform=text_transform,
        is_train=True
    )

    val_data = None
    if val_url_dataset is not None:
        val_data = WDSDataset(
            model_cfg.data,
            train_url_dataset,
            image_transform=image_transform(
                    img_size,
                    is_train=False,
                    mean=img_mean,
                    std=img_std,
                ),
            text_transform=text_transform,
            is_train=False
        )

    return train_data, val_data
