import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import tqdm
from chime7_enhancers import FrontEnd_v1
from gss.core.enhancer import get_enhancer
from lhotse import Recording, SupervisionSet, load_manifest_lazy
from lhotse.audio import set_audio_duration_mismatch_tolerance
from lhotse.cut import CutSet
from lhotse.utils import fastcopy

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


ENHANCER_IMPL_CHOICES = ['gss', 'nemo_v1']


def enhance_cuts(
    enhancer_impl: str,
    cuts_per_recording: str,
    cuts_per_segment: str,
    enhanced_dir: str,
    bss_iterations: int,
    context_duration: float,
    use_garbage_class: bool,
    min_segment_length: float,
    max_segment_length: float,
    max_batch_duration: float,
    max_batch_cuts: int,
    num_buckets: int,
    num_workers: int,
    force_overwrite: bool,
    duration_tolerance: float,
    channels: Optional[str] = None,
    torchaudio_backend: str = 'soundfile',
):
    logger.info('Enhance cuts')
    logger.info('\tenhancer_impl:      %s', enhancer_impl)
    logger.info('\tcuts_per_recording: %s', cuts_per_recording)
    logger.info('\tcuts_per_segment:   %s', cuts_per_segment)
    logger.info('\tenhanced_dir:       %s', enhanced_dir)
    logger.info('\tbss_iterations:     %d', bss_iterations)
    logger.info('\tcontext_duration:   %f', context_duration)
    logger.info('\tuse_garbage_class:  %s', use_garbage_class)
    logger.info('\tmin_segment_length: %f', min_segment_length)
    logger.info('\tmax_segment_length: %f', max_segment_length)
    logger.info('\tmax_batch_duration: %f', max_batch_duration)
    logger.info('\tmax_batch_cuts:     %d', max_batch_cuts)
    logger.info('\tnum_buckets:        %d', num_buckets)
    logger.info('\tnum_workers:        %d', num_workers)
    logger.info('\tforce_overwrite:    %s', force_overwrite)
    logger.info('\tduration_tolerance: %f', duration_tolerance)
    logger.info('\tchannels:           %s', channels)
    logger.info('\ttorchaudio_backend: %s', torchaudio_backend)

    # ########################################
    # Setup as in gss.bin.modes.enhance.cuts_
    # ########################################
    if duration_tolerance is not None:
        set_audio_duration_mismatch_tolerance(duration_tolerance)

    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(exist_ok=True, parents=True)

    cuts = load_manifest_lazy(cuts_per_recording)
    cuts_per_segment = load_manifest_lazy(cuts_per_segment)

    if channels is not None:
        channels = [int(c) for c in channels.split(",")]
        cuts_per_segment = CutSet.from_cuts(fastcopy(cut, channel=channels) for cut in cuts_per_segment)

    # Paranoia mode: ensure that cuts_per_recording have ids same as the recording_id
    cuts = CutSet.from_cuts(cut.with_id(cut.recording_id) for cut in cuts)

    logger.info("Aplying min/max segment length constraints")
    cuts_per_segment = cuts_per_segment.filter(lambda c: c.duration > min_segment_length).cut_into_windows(
        duration=max_segment_length
    )

    # ########################################
    # Initialize enhancer
    # ########################################

    if enhancer_impl == 'gss':
        logger.info("Initializing GSS enhancer")
        enhancer = get_enhancer(
            cuts=cuts,
            bss_iterations=bss_iterations,
            context_duration=context_duration,
            activity_garbage_class=use_garbage_class,
            wpe=True,
        )
    elif enhancer_impl == 'nemo_v1':
        enhancer = FrontEnd_v1(
            stft_fft_length=1024,
            stft_hop_length=256,
            dereverb_prediction_delay=2,
            dereverb_filter_length=10,
            dereverb_num_iterations=3,
            bss_iterations=bss_iterations,
            mc_filter_type='pmwf',
            mc_filter_beta=0,
            mc_filter_rank='one',
            mc_filter_postfilter='ban',
            mc_ref_channel='max_snr',
            use_dtype=torch.cfloat,
            cuts=cuts,
            context_duration=context_duration,
            activity_garbage_class=use_garbage_class,
        )
    else:
        raise NotImplementedError(f'Unknown enhancer implementation: {enhancer_impl}')

    # ########################################
    # Enhance cuts and save audio
    # ########################################

    logger.info(f'Enhancing {len(frozenset(c.id for c in cuts_per_segment))} segments')
    begin = time.time()
    num_errors, out_cuts = enhancer.enhance_cuts(
        cuts=cuts_per_segment,
        exp_dir=enhanced_dir,
        max_batch_duration=max_batch_duration,
        max_batch_cuts=max_batch_cuts,
        num_workers=num_workers,
        num_buckets=num_buckets,
        force_overwrite=force_overwrite,
        torchaudio_backend=torchaudio_backend,
    )
    end = time.time()

    if num_errors > 0:
        logger.error(f'Finished in {end-begin:.2f}s with {num_errors} errors')
    else:
        logger.info(f'Finished in {end-begin:.2f}s')


if __name__ == '__main__':
    """Simulate the interface of gss enhance cuts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--enhancer-impl',
        type=str,
        choices=ENHANCER_IMPL_CHOICES,
        default='nemo_v1',
        help='Implementation of the enhancer, e.g., gss',
    )
    parser.add_argument(
        '--cuts-per-recording', type=str, required=True, help='Path to cuts manifest, e.g., cuts.jsonl.gz',
    )
    parser.add_argument(
        '--cuts-per-segment',
        type=str,
        required=True,
        help='Path to cuts_per_segment manifest, e.g., cuts_per_segment.JOB.jsonl.gz',
    )
    parser.add_argument(
        '--enhanced-dir', type=str, required=True, help='Output dir for enhanced audio',
    )
    parser.add_argument(
        '--bss-iterations', type=int, default=20, help='Number of BSS iterations. Default: 20',
    )
    parser.add_argument(
        '--context-duration', type=float, default=15, help='Context duration in seconds. Default: 15',
    )
    parser.add_argument(
        '--use-garbage-class',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Use garbage class. Default: True',
    )
    parser.add_argument(
        '--min-segment-length', type=float, default=0.0,
    )
    parser.add_argument(
        '--max-segment-length', type=float, default=200,
    )
    parser.add_argument(
        '--max-batch-duration', type=float, default=200,
    )
    parser.add_argument(
        '--max-batch-cuts', type=int, default=1,
    )
    parser.add_argument(
        '--num-buckets', type=int, default=4,
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
    )
    parser.add_argument(
        '--force-overwrite', action='store_true',
    )
    parser.add_argument(
        '--duration-tolerance', type=float, default=3.0,
    )
    parser.add_argument('--channels', type=str, default=None, help='Comma-separated list of channels')
    parser.add_argument(
        '--torchaudio-backend',
        type=str,
        choices=['sox_io', 'soundfile'],
        default='soundfile',  # faster than the defaulut sox_io
        help='Backend used for torchaudio',
    )
    args = parser.parse_args()

    enhance_cuts(
        enhancer_impl=args.enhancer_impl,
        cuts_per_recording=args.cuts_per_recording,
        cuts_per_segment=args.cuts_per_segment,
        enhanced_dir=args.enhanced_dir,
        bss_iterations=args.bss_iterations,
        context_duration=args.context_duration,
        use_garbage_class=args.use_garbage_class,
        min_segment_length=args.min_segment_length,
        max_segment_length=args.max_segment_length,
        max_batch_duration=args.max_batch_duration,
        max_batch_cuts=args.max_batch_cuts,
        num_buckets=args.num_buckets,
        num_workers=args.num_workers,
        force_overwrite=args.force_overwrite,
        duration_tolerance=args.duration_tolerance,
        channels=args.channels,
        torchaudio_backend=args.torchaudio_backend,
    )
