from nemo.collections.multimodal.data.clip.wds_dataset import WebDatasetUrls, WDSDataset
from nemo.collections.multimodal.data.clip.wds_utils import RandomSamplerIterableDataset
from nemo.collections.multimodal.data.clip.data_samplers import WDSUrlsRandomSampler


def build_train_valid_datasets(model_cfg, consumed_samples):
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

    # Create the actual WebDataset IterableDataset
    # Expanding the url to actual samples with shuffling, decoding and transforming
    train_data = WDSDataset(model_cfg.data, train_url_dataset, is_train=True)
    val_data = None
    if val_url_dataset is not None:
        val_data = WDSDataset(model_cfg.data, val_url_dataset, is_train=False)

    return train_data, val_data
