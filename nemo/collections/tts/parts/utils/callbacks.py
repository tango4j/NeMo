import os
import librosa
import soundfile as sf
from pathlib import Path
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import List, Optional

import torch

from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor, save_plot

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


def _get_logger(loggers, logger_type):
    for logger in loggers:
        if isinstance(logger, logger_type):
            return logger.experiment
    raise ValueError(f"Could not find {logger_type} logger in {loggers}.")


def _load_vocoder(vocoder_type, vocoder_checkpoint):
    if vocoder_type == "hifigan":
        from nemo.collections.tts.models import HifiGanModel
        vocoder = HifiGanModel.load_from_checkpoint(vocoder_checkpoint).eval()
    elif vocoder_type == "univnet":
        from nemo.collections.tts.models import UnivNetModel
        vocoder = UnivNetModel.load_from_checkpoint(vocoder_checkpoint).eval()
    else:
        raise ValueError(f"Unknown vocoder type '{vocoder_type}'")

    return vocoder


class VocoderLoggingCallback(Callback):

    def __init__(
        self,
        sample_rate: int,
        data_loader: torch.utils.data.DataLoader,
        output_dir: Path,
        epoch_frequency: int = 1,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        loggers: Optional[List[Logger]] = None,
    ):
        self.sample_rate = sample_rate
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.epoch_frequency = epoch_frequency
        self.log_tensorboard = log_tensorboard
        self.log_wandb = log_wandb

        if log_tensorboard:
            self.tensorboard_logger = _get_logger(loggers, TensorBoardLogger)
        else:
            self.tensorboard_logger = None

        if log_wandb:
            if not HAVE_WANDB:
                raise ValueError("Wandb not installed.")

            self.wandb_logger = _get_logger(loggers, WandbLogger)
        else:
            self.wandb_logger = None

    def _log_audio(self, audio_id, filepath, audio, step):
        sf.write(file=filepath, data=audio, samplerate=self.sample_rate)

        if self.tensorboard_logger:
            self.tensorboard_logger.add_audio(
                tag=audio_id,
                snd_tensor=audio,
                global_step=step,
                sample_rate=self.sample_rate,
            )

        if self.wandb_logger:
            wandb_audio = wandb.Audio(audio, sample_rate=self.sample_rate, caption=audio_id),
            self.wandb_logger.log({audio_id: wandb_audio})

    def on_train_epoch_end(self, trainer: Trainer, vocoder_model: LightningModule):
        epoch = 1 + vocoder_model.current_epoch
        if epoch % self.epoch_frequency != 0:
            return

        log_epoch_dir = self.output_dir / f"epoch_{epoch}"
        log_epoch_dir.mkdir(parents=True, exist_ok=True)

        for batch_dict in self.data_loader:
            audio_filepaths = batch_dict.get("audio_filepaths")
            audio = batch_dict.get("audio").to(vocoder_model.device)
            audio_len = batch_dict.get("audio_lens").to(vocoder_model.device)

            audio_ids = [str(p.with_suffix("")).replace(os.sep, "_") for p in audio_filepaths]
            audio_paths = [log_epoch_dir / s for s in audio_ids]

            spec, spec_len = vocoder_model.audio_to_melspec_precessor(audio, audio_len)

            with torch.no_grad():
                # [batch, 1, time]
                audio_pred = vocoder_model.forward(spec=spec)

            audio_pred = mask_sequence_tensor(tensor=audio_pred, lengths=audio_len)

            for i, (audio_id, audio_path) in enumerate(zip(audio_ids, audio_paths)):
                audio_file = f"{audio_path}.wav"
                audio_pred_i = audio_pred[i][0][:audio_len[i]]
                audio_pred_i = audio_pred_i.cpu().numpy()
                log_audio_id = f"audio_{audio_id}"
                self._log_audio(
                    audio_id=log_audio_id,
                    filepath=audio_file,
                    audio=audio_pred_i,
                    step=vocoder_model.global_step
                )


class FastPitchLoggingCallback(Callback):

    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        output_dir: Path,
        epoch_frequency: int = 1,
        log_audio: bool = False,
        log_audio_gta: bool = False,
        log_spectrogram: bool = True,
        log_alignment: bool = True,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        loggers: Optional[List[Logger]] = None,
        vocoder_type: Optional[str] = None,
        vocoder_checkpoint_path: Optional[Path] = None,
    ):
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.epoch_frequency = epoch_frequency
        self.log_audio = log_audio
        self.log_audio_gta = log_audio_gta
        self.log_spectrogram = log_spectrogram
        self.log_alignment = log_alignment
        self.log_tensorboard = log_tensorboard
        self.log_wandb = log_wandb

        if log_tensorboard:
            self.tensorboard_logger = _get_logger(loggers, TensorBoardLogger)
        else:
            self.tensorboard_logger = None

        if log_wandb:
            if not HAVE_WANDB:
                raise ValueError("Wandb not installed.")

            self.wandb_logger = _get_logger(loggers, WandbLogger)
        else:
            self.wandb_logger = None

        if self.log_audio or self.log_audio_gta:
            if not vocoder_type or not vocoder_checkpoint_path:
                raise ValueError(f"Logging audio requires vocoder type, path. Received: "
                                 f"{vocoder_type}, "
                                 f"{vocoder_checkpoint_path}")
            self.vocoder = _load_vocoder(
                vocoder_type, vocoder_checkpoint_path
            )
        else:
            self.vocoder = None

    def _log_image(self, image_id, filepath, data, step, x_axis, y_axis):
        spec_data = save_plot(
            output_filepath=filepath, data=data, x_axis=x_axis, y_axis=y_axis
        )

        if self.tensorboard_logger:
            self.tensorboard_logger.add_image(
                tag=image_id,
                img_tensor=spec_data,
                global_step=step,
                dataformats="HWC",
            )

        if self.wandb_logger:
            wandb_image = wandb.Image(data, caption=image_id),
            self.wandb_logger.log({image_id: wandb_image})

    def _log_audio(self, audio_id, filepath, audio, step):
        sf.write(file=filepath, data=audio, samplerate=self.vocoder.sample_rate)

        if self.tensorboard_logger:
            self.tensorboard_logger.add_audio(
                tag=audio_id,
                snd_tensor=audio,
                global_step=step,
                sample_rate=self.vocoder.sample_rate,
            )

        if self.wandb_logger:
            wandb_audio = wandb.Audio(audio, sample_rate=self.vocoder.sample_rate, caption=audio_id),
            self.wandb_logger.log({audio_id: wandb_audio})

    def on_train_epoch_end(self, trainer: Trainer, fastpitch_model: LightningModule):
        epoch = 1 + fastpitch_model.current_epoch
        if epoch % self.epoch_frequency != 0:
            return

        log_epoch_dir = self.output_dir / f"epoch_{epoch}"
        log_epoch_dir.mkdir(parents=True, exist_ok=True)

        for batch_dict in self.data_loader:
            audio_filepaths = batch_dict.get("audio_filepaths")
            audio = batch_dict.get("audio").to(fastpitch_model.device)
            audio_lens = batch_dict.get("audio_lens").to(fastpitch_model.device)
            text = batch_dict.get("text").to(fastpitch_model.device)
            text_lens = batch_dict.get("text_lens").to(fastpitch_model.device)
            attn_prior = batch_dict.get("align_prior_matrix", None).to(fastpitch_model.device)
            pitch = batch_dict.get("pitch", None).to(fastpitch_model.device)
            energy = batch_dict.get("energy", None).to(fastpitch_model.device)
            speaker = batch_dict.get("speaker_id", None).to(fastpitch_model.device)

            audio_ids = [str(p.with_suffix("")).replace(os.sep, "_") for p in audio_filepaths]
            audio_paths = [log_epoch_dir / s for s in audio_ids]

            if self.log_spectrogram or self.log_audio:
                with torch.no_grad():
                    mels_pred, mels_pred_len, *_ = fastpitch_model.forward(
                        text=text,
                        input_lens=text_lens,
                        speaker=speaker,
                    )

                if self.log_spectrogram:
                    for i, (audio_id, audio_path) in enumerate(zip(audio_ids, audio_paths)):
                        spec_file = f"{audio_path}_spec.png"
                        spec_i = mels_pred[i][:, :mels_pred_len[i]]
                        spec_i = spec_i.cpu().numpy()
                        spec_id = f"spectrogram_{audio_id}"
                        self._log_image(
                            image_id=spec_id,
                            filepath=spec_file,
                            data=spec_i,
                            step=fastpitch_model.global_step,
                            x_axis="Audio Frames",
                            y_axis="Channels"
                        )

                if self.log_audio:
                    vocoder_input = mels_pred.to(self.vocoder.device)
                    with torch.no_grad():
                        audio_pred = self.vocoder.convert_spectrogram_to_audio(spec=vocoder_input)
                    mels_pred_len_array = mels_pred_len.cpu().numpy()
                    audio_pred_lens = librosa.core.frames_to_samples(
                        mels_pred_len_array,
                        hop_length=fastpitch_model.preprocessor.hop_length
                    )
                    for i, (audio_id, audio_path) in enumerate(zip(audio_ids, audio_paths)):
                        audio_file = f"{audio_path}.wav"
                        audio_pred_i = audio_pred[i][:audio_pred_lens[i]]
                        audio_pred_i = audio_pred_i.cpu().numpy()
                        log_audio_id = f"audio_{audio_id}"
                        self._log_audio(
                            audio_id=log_audio_id,
                            filepath=audio_file,
                            audio=audio_pred_i,
                            step=fastpitch_model.global_step
                        )

            if self.log_alignment or self.log_audio_gta:
                mels, spec_len = fastpitch_model.preprocessor(input_signal=audio, length=audio_lens)
                with torch.no_grad():
                    mels_gta_pred, mels_gta_pred_len, _, _, _, attn_soft, _, _, _, _, _, _ = fastpitch_model.forward(
                        text=text,
                        input_lens=text_lens,
                        pitch=pitch,
                        energy=energy,
                        speaker=speaker,
                        spec=mels,
                        mel_lens=spec_len,
                        attn_prior=attn_prior,
                    )

                if self.log_alignment:
                    for i, (audio_id, audio_path) in enumerate(zip(audio_ids, audio_paths)):
                        align_file = f"{audio_path}_align.png"
                        attn_soft_i = attn_soft[i][0][:mels_gta_pred_len[i], :text_lens[i]]
                        attn_soft_i = attn_soft_i.cpu().numpy().transpose()
                        alignment_id = f"alignment_{audio_id}"
                        self._log_image(
                            image_id=alignment_id,
                            filepath=align_file,
                            data=attn_soft_i,
                            step=fastpitch_model.global_step,
                            x_axis="Audio Frames",
                            y_axis="Text Tokens"
                        )

                if self.log_audio_gta:
                    vocoder_input = mels_gta_pred.to(self.vocoder.device)
                    with torch.no_grad():
                        audio_gta_pred = self.vocoder.convert_spectrogram_to_audio(spec=vocoder_input)
                    mels_gta_pred_len_array = mels_gta_pred_len.cpu().numpy()
                    audio_gta_pred_lens = librosa.core.frames_to_samples(
                        mels_gta_pred_len_array,
                        hop_length=fastpitch_model.preprocessor.hop_length
                    )
                    for i, (audio_id, audio_path) in enumerate(zip(audio_ids, audio_paths)):
                        audio_gta_file = f"{audio_path}_gta.wav"
                        audio_gta_pred_i = audio_gta_pred[i][:audio_gta_pred_lens[i]]
                        audio_gta_pred_i = audio_gta_pred_i.cpu().numpy()
                        log_audio_id = f"audio_gta_{audio_id}"
                        self._log_audio(
                            audio_id=log_audio_id,
                            filepath=audio_gta_file,
                            audio=audio_gta_pred_i,
                            step=fastpitch_model.global_step
                        )
