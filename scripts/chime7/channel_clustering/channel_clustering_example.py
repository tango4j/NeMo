import argparse
import os
import torch
from matplotlib import pyplot as plt
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.parts.preprocessing.manifest import get_full_path

from nemo.collections.asr.parts.utils.channel_clustering import channel_cluster_from_coherence


def subband_to_freq(subband, fft_length, sample_rate):
    """Convert subband to freq

    Args:
        subband: subband index

    Returns:
        Frequency in Hz
    """
    num_subbands = fft_length//2 + 1
    assert subband < num_subbands, f'Subband {subband} is out of range, num_subbands {num_subbands}.'
    return subband * sample_rate / fft_length

def main(manifest_file, session_index, duration = 300, offset = 0):

    sample_rate = 16000

    # read manifest
    metadata = read_manifest(manifest_file)

    # get the session
    data = metadata[session_index]
    audio_filepath = get_full_path(data['audio_filepath'], manifest_file=manifest_file)

    # session name
    session_name = os.path.basename(audio_filepath[0]).split('CH')[0].rstrip('_.')

    # load
    audio_segment = AudioSegment.from_file(audio_filepath, target_sr=sample_rate, offset=offset, duration=duration)

    # use NeMo to compute STFT
    audio_signal = torch.tensor(audio_segment.samples.T)
    
    # run clustering
    clusters, mag_coherence = channel_cluster_from_coherence(
                                    audio_signal=audio_signal,
                                    sample_rate=sample_rate,
                                    output_coherence=True,
                                    )
    affinity_mat = clusters[:, None] == clusters[None, :]

    print('num clusters:', len(clusters.unique()))
    print('clusters:', clusters)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mag_coherence)
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    plt.title(session_name)
    plt.subplot(1,2,2)
    plt.imshow(affinity_mat)
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    plt.title('Binarized')
    plt.savefig(os.path.basename(manifest_file).replace('.json', '.png'))


if __name__ == '__main__':
    """
    Example:
        python channel_clustering_example.py --manifest /data/chime7/chime7_official_cleaned/chime6/chime6-dev_manifest.json --session-index 0
        python channel_clustering_example.py --manifest /data/chime7/chime7_official_cleaned/dipco/dipco-dev_manifest.json --session-index 0
        python channel_clustering_example.py --manifest /data/chime7/chime7_official_cleaned/mixer6/mixer6-dev_manifest.json --session-index 0
    """
    parser = argparse.ArgumentParser(description='Channel Clustering Example')
    parser.add_argument(
        '--manifest',
        metavar='manifest',
        type=str,
        required=True,
        help='Path to a manifest file',
    )
    parser.add_argument(
        '--session-index',
        metavar='session_index',
        type=int,
        required=True,
        help='Index of the session in the manifest file',
    )
    parser.add_argument(
        '--duration',
        metavar='duration',
        type=float,
        default=300,
        help='Duration of the segment used for clustering [in seconds]',
    )
    parser.add_argument(
        '--offset',
        metavar='offset',
        type=float,
        default=0,
        help='Duration of the segment used for clustering [in seconds]',
    )
    args = parser.parse_args()

    main(args.manifest, args.session_index, args.duration, args.offset)