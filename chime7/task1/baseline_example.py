

import numpy as np
import soundfile as sf
import torch
import os

from nemo.collections.asr.modules.audio_modules import MaskBasedDereverbWPE
from nemo.collections.asr.modules.audio_preprocessing import AudioToSpectrogram, SpectrogramToAudio
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_mc_audio_rttm_map,
    get_subsegments,
    get_uniqname_from_filepath,
    rttm_to_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
)

# Parameters
fft_length = 512
hop_length = fft_length // 2
filter_length = 8
delay = 2
num_iterations = 3

# input_manifest = "/disk_d/datasets/nemo_chime7_manifests/dipco/mulspk_asr_manifest/dipco-dev.json"
input_manifest = "/disk_d/datasets/nemo_chime7_manifests/mixer6/mulspk_asr_manifest/mixer6-dev.json"
audio_rttm_map_manifest = get_mc_audio_rttm_map(input_manifest)
sample_rate = 16000

# Read all audio files and save them into a numpy array
# Iterate through all audio file keys
for uniq_id, audio_rttm_map_dict in audio_rttm_map_manifest.items():
    session_audio_list = []
    ch_n = len(audio_rttm_map_dict['audio_filepath'])
    min_dur_sec = audio_rttm_map_dict['min_duration']    
    min_dur_sample_n = int(min_dur_sec * sample_rate)
    for audio_filepath in audio_rttm_map_dict['audio_filepath']:
        # Read audio
        audio_ts = AudioSegment.from_file(audio_filepath, target_sr=sample_rate).samples.T
        audio_ts = torch.from_numpy(audio_ts).to('cuda:0')
        session_audio_list.append(audio_ts[:min_dur_sample_n])
    # Construct (C, T) numpy array
    print(f"Concatenating {ch_n} channels into a numpy array of length {min_dur_sec} seconds...")
    # mc_x = np.concatenate(session_audio_list).reshape(-1, min_dur_sample_n)
    mc_x = torch.stack(session_audio_list).reshape(-1, min_dur_sample_n).to('cuda:0')
    # mc_x = np.stack(session_audio_list) # OOM!

    # Add batch dimension, shape (B, C, T)
    mc_x = mc_x[None, ...].to('cuda:0')

    # Prepare analysis and synthesis transforms
    stft = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length).to('cuda:0')
    istft = SpectrogramToAudio(fft_length=fft_length, hop_length=hop_length).to('cuda:0')

    # Analysis transform
    mc_X, _ = stft(input=mc_x)
    mc_X = mc_X[:, :, :, :5000]
    mc_X = mc_X.to('cuda:0')

    import ipdb; ipdb.set_trace()

    # Prepare dereverb instance
    dereverb = MaskBasedDereverbWPE(filter_length=filter_length, prediction_delay=delay, num_iterations=num_iterations).to('cuda:0')
    # Processing
    mc_Y, _ = dereverb(input=mc_X, mask=None)
    mc_Y = mc_Y.to('cuda:0')

    # Synthesis transform: shape (B, C, T)
    mc_y, _ = istft(input=mc_Y)
    mc_y = mc_y.to('cuda:0')


    os.makedirs('output', exist_ok=True)
    import ipdb; ipdb.set_trace()



# Load audio
sample_rate = 16000
x = AudioSegment.from_file(filename, target_sr=sample_rate).samples.T

# Add batch dimension, shape (B, C, T)
x = x[None, ...]

# Prepare analysis and synthesis transforms
stft = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)
istft = SpectrogramToAudio(fft_length=fft_length, hop_length=hop_length)

# Analysis transform
X, _ = stft(input=torch.tensor(x))

# Prepare dereverb instance
dereverb = MaskBasedDereverbWPE(filter_length=filter_length, prediction_delay=delay, num_iterations=num_iterations)
# Processing
Y, _ = dereverb(input=X, mask=None)

# Synthesis transform: shape (B, C, T)
y, _ = istft(input=Y)

os.makedirs('output', exist_ok=True)
# Save audio
sf.write('output/dereverb_output.wav', y.cpu().numpy()[0, ...].T, sample_rate)

