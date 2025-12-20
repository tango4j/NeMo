# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIBotTTSTextMessage, RTVIBotLLMStartedMessage, RTVIBotLLMStoppedMessage, RTVIBotTTSStartedMessage, RTVIBotTTSStoppedMessage, RTVITextMessageData, RTVIBotTranscriptionMessage, RTVIServerMessage, RTVIServerResponseFrame, RTVIServerMessageFrame

from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import (
    FunctionCallParams,  # TODO(aleix): we shouldn't import `services` from `processors`
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport
from pipecat.utils.string import match_endofsentence

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from pipecat.frames.frames import (
    BotInterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    DataFrame,
    EndFrame,
    EndTaskFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
    TransportMessageUrgentFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
class AudioLogger:
    """
    Utility class for logging audio data and transcriptions during voice agent interactions.

    This logger saves:
    - Audio files in WAV format
    - Transcriptions with metadata in JSON format
    - Session information and metadata

    File structure:
        log_dir/
        ├── session_YYYYMMDD_HHMMSS/
        │   ├── user/
        │   │   ├── 00001_HHMMSS.wav
        │   │   ├── 00001_HHMMSS.json
        │   │   ├── 00002_HHMMSS.wav
        │   │   └── 00002_HHMMSS.json
        │   ├── agent/
        │   │   ├── 00001_HHMMSS.wav
        │   │   ├── 00001_HHMMSS.json
        │   └── session_metadata.json

    Args:
        log_dir: Base directory for storing logs (default: "./audio_logs")
        session_id: Optional custom session ID. If None, auto-generated from timestamp
        enabled: Whether logging is enabled (default: True)

    # 12/19/2025 Note: Stereo conversation recording is implemented, but -0.8 seconds offset needs to be applied to make the session sound synced. 
        TODO: 
        1. Make offset compensated conversation_stereo.wav file: Start time is the first turn's start time.
        2. user_speech.wav in user folder. This is not offset compensated but also start time is the first turn's start time.
        3. Remove the segment based wav files. (Both TTS and STT, make it configurable, but default to False)
        4. Do the testing on Jetson. 
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "./audio_logs",
        session_id: Optional[str] = None,
        enabled: bool = True,
        user_lead_in_frame_count: int = 2,
        user_audio_sample_rate: int = 16000,
    ):
        self.enabled = enabled
        if not self.enabled:
            logger.info("[AudioLogger] AudioLogger is disabled")
            return

        self.log_dir = Path(log_dir)

        # Generate session ID if not provided
        self.session_start_time = datetime.now()
        if session_id is None:
            session_id = f"session_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}"
        self.session_id = session_id
        self.session_dir = self.log_dir / session_id

        # Create directories
        self.user_dir = self.session_dir / "user"
        self.agent_dir = self.session_dir / "agent"

        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        # Counters for file naming (thread-safe)
        self.user_lead_in_frame_count = user_lead_in_frame_count
        self._user_counter = 0
        self._agent_counter = 0
        self._turn_index = 0  # Turn index for conversation turns
        self._current_speaker = None  # Track current speaker for turn transitions
        self._agent_turn_start_time = None  # Captured when BotStartedSpeakingFrame is received
        self._lock = threading.Lock()
        self._staged_metadata = None
        self._staged_audio_data = None

        self.turn_audio_buffer = []
        self.turn_transcription_buffer = []
        # self.lead_in_user_audio_frame_queue = []

        # Global user audio recording (entire conversation)
        self._global_user_audio_buffer: list = []
        self._global_user_speech_filename = "user_speech.wav"
        self._global_user_audio_file = self.user_dir / self._global_user_speech_filename
        self._global_user_audio_sample_rate = user_audio_sample_rate

        # Stereo conversation recording (left=agent, right=user)
        self._stereo_conversation_filename = "conversation_stereo.wav"
        self._stereo_conversation_file = self.session_dir / self._stereo_conversation_filename
        self._stereo_sample_rate = 24000  # TTS sample rate (higher of the two)
        self._stereo_audio_buffer_left: list = []   # Agent audio (left channel)
        self._stereo_audio_buffer_right: list = []  # User audio (right channel)
        self._user_audio_stereo_offset = -0.8  # Offset in seconds to compensate for user audio lag

        # Session metadata
        # agent_entries is a list of lists: each sublist contains segments for one turn
        # e.g., [[seg1, seg2, seg3], [seg4, seg5], ...]  where each [] is a turn
        self.session_metadata = {
            "session_id": session_id,
            "start_time": self.session_start_time.isoformat(),
            "user_entries": [],
            "agent_entries": [],  # List of turns, each turn is a list of segments
        }

        logger.info(f"[AudioLogger] AudioLogger initialized: {self.session_dir}")



    def append_global_user_audio(self, audio_data: bytes):
        """
        Append audio data to the global user audio buffer.
        
        This method should be called for EVERY audio frame received from the user,
        regardless of VAD state, to record the complete conversation audio.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        if not self.enabled:
            return
        
        self._global_user_audio_buffer.append(audio_data)

    def save_global_user_audio(self):
        """
        Save the complete global user audio buffer to user_speech.wav.
        
        This method should be called at session finalization to save the
        entire conversation audio as a single file.
        """
        if not self.enabled:
            return
        
        if not self._global_user_audio_buffer:
            logger.warning("[AudioLogger] No global user audio to save")
            return
        
        try:
            # Join all audio frames
            complete_audio = b"".join(self._global_user_audio_buffer)
            
            # Save to file
            self._save_audio_wav(
                audio_data=complete_audio,
                file_path=self._global_user_audio_file,
                sample_rate=self._global_user_audio_sample_rate,
                num_channels=1,
            )
            
            audio_duration_sec = len(complete_audio) / (self._global_user_audio_sample_rate * 2)
            logger.info(
                f"[AudioLogger] Saved global user audio: {self._global_user_audio_file} "
                f"({audio_duration_sec:.2f} seconds, {len(self._global_user_audio_buffer)} frames)"
            )
            
            # Clear buffer after saving
            self._global_user_audio_buffer = []
            
        except Exception as e:
            logger.error(f"[AudioLogger] Error saving global user audio: {e}")

    def _resample_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """
        Resample audio data to a target sample rate using librosa.
        
        Args:
            audio_data: Audio data as bytes (int16) or numpy array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio as numpy array (float32)
        """
        # Convert bytes to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int16:
            audio_array = audio_data.astype(np.float32) / 32768.0
        else:
            audio_array = audio_data.astype(np.float32)
        
        # Resample if needed
        if orig_sr != target_sr:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
        
        return audio_array

    def _append_to_stereo_conversation(
        self,
        audio_data: Union[bytes, np.ndarray],
        channel: str,
        start_time: float,
        sample_rate: int,
    ):
        """
        Append audio to the stereo conversation buffer at the correct time position.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            channel: "left" for agent, "right" for user
            start_time: Start time in seconds from session start
            sample_rate: Sample rate of the input audio
        """
        if not self.enabled:
            return
        
        try:
            # Resample to stereo sample rate if needed
            audio_float = self._resample_audio(audio_data, sample_rate, self._stereo_sample_rate)
            
            # Calculate the sample position for this audio
            start_sample = int(start_time * self._stereo_sample_rate)
            
            # Get the appropriate buffer
            if channel == "left":
                buffer = self._stereo_audio_buffer_left
            else:
                buffer = self._stereo_audio_buffer_right
            
            # Extend buffer with zeros if needed to reach start position
            current_length = len(buffer)
            if start_sample > current_length:
                buffer.extend([0.0] * (start_sample - current_length))
            
            # Append or overwrite audio samples
            for i, sample in enumerate(audio_float):
                pos = start_sample + i
                if pos < len(buffer):
                    # Mix with existing audio (in case of overlap)
                    buffer[pos] = np.clip(buffer[pos] + sample, -1.0, 1.0)
                else:
                    buffer.append(sample)
            
            logger.debug(
                f"[AudioLogger] Appended {len(audio_float)} samples to {channel} channel "
                f"at position {start_sample} (buffer now {len(buffer)} samples)"
            )
            
        except Exception as e:
            logger.error(f"[AudioLogger] Error appending to stereo conversation: {e}")

    def save_stereo_conversation(self):
        """
        Save the stereo conversation buffer to a WAV file.
        Left channel = Agent, Right channel = User.
        """
        if not self.enabled:
            return
        
        if not self._stereo_audio_buffer_left and not self._stereo_audio_buffer_right:
            logger.warning("[AudioLogger] No stereo conversation audio to save")
            return
        
        try:
            # Pad the shorter buffer with zeros
            max_length = max(len(self._stereo_audio_buffer_left), len(self._stereo_audio_buffer_right))
            
            left_array = np.array(self._stereo_audio_buffer_left, dtype=np.float32)
            right_array = np.array(self._stereo_audio_buffer_right, dtype=np.float32)
            
            # Pad to same length
            if len(left_array) < max_length:
                left_array = np.pad(left_array, (0, max_length - len(left_array)))
            if len(right_array) < max_length:
                right_array = np.pad(right_array, (0, max_length - len(right_array)))
            
            # Create stereo array (interleaved: L, R, L, R, ...)
            stereo_array = np.column_stack((left_array, right_array))
            
            # Convert to int16
            stereo_int16 = (stereo_array * 32767).astype(np.int16)
            
            # Save as WAV
            with wave.open(str(self._stereo_conversation_file), 'wb') as wav_file:  # type: ignore[union-attr]
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self._stereo_sample_rate)
                wav_file.writeframes(stereo_int16.tobytes())
            
            duration_sec = max_length / self._stereo_sample_rate
            logger.info(
                f"[AudioLogger] Saved stereo conversation: {self._stereo_conversation_file} "
                f"({duration_sec:.2f} seconds, {max_length} samples)"
            )
            
        except Exception as e:
            logger.error(f"[AudioLogger] Error saving stereo conversation: {e}")

    def get_time_from_start_of_session(self, timestamp: datetime = None) -> float:
        """Get the time from the start of the session to the given datetime string."""
        # get the time difference in seconds.
        time_diff = (timestamp if timestamp else datetime.now()) - self.session_start_time
        return time_diff.total_seconds()

    def _get_next_counter(self, speaker: str) -> int:
        """Get the next counter value for a speaker in a thread-safe manner."""
        with self._lock:
            if speaker == "user":
                self._user_counter += 1
                return self._user_counter
            else:
                self._agent_counter += 1
                return self._agent_counter

    def get_turn_index(self) -> int:
        """Get the current turn index."""
        return self._turn_index

    def increment_turn_index(self, speaker: str = None) -> int:
        """
        Increment the turn index if the speaker has changed.
        
        Args:
            speaker: "user" or "agent". If provided, only increments
                    if this is different from the current speaker.
                    If None, always increments.
        
        Returns:
            The current turn index after any increment.
        """
        with self._lock:
            if speaker is None:
                # Always increment if no speaker specified
                self._turn_index += 1
                logger.debug(f"[AudioLogger] Turn index incremented to {self._turn_index}")
            elif speaker != self._current_speaker:
                # Only increment if speaker changed
                self._current_speaker = speaker
                self._turn_index += 1
                # Reset agent turn start time when speaker changes
                if speaker == "agent":
                    self._agent_turn_start_time = None
                logger.debug(f"[AudioLogger] Speaker changed to {speaker}, turn index incremented to {self._turn_index}")
            # else: same speaker, no increment
            return self._turn_index

    def set_agent_turn_start_time(self):
        """
        Set the start time for the current agent turn.
        
        This should be called when BotStartedSpeakingFrame is received,
        which indicates the audio is actually starting to play (not just generated).
        This provides more accurate timing than capturing time during TTS generation.
        """
        if not self.enabled:
            return
        
        # Only set if not already set for this turn
        if self._agent_turn_start_time is None:
            self._agent_turn_start_time = self.get_time_from_start_of_session()
            logger.debug(f"[AudioLogger] Agent turn start time set to {self._agent_turn_start_time:.3f}s")

    def get_agent_turn_start_time(self) -> Optional[float]:
        """
        Get the start time for the current agent turn.
        
        Returns:
            The agent turn start time, or None if not set.
        """
        return self._agent_turn_start_time

    def _save_audio_wav(
        self,
        audio_data: Union[bytes, np.ndarray],
        file_path: Path,
        sample_rate: int,
        num_channels: int = 1,
    ):
        """
        Save audio data to a WAV file.

        Args:
            audio_data: Audio data as bytes or numpy array
            file_path: Path to save the WAV file
            sample_rate: Audio sample rate in Hz
            num_channels: Number of audio channels (default: 1)
        """
        try:
            # Convert audio data to bytes if it's a numpy array
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype in [np.float32, np.float64]:
                    # Convert float [-1, 1] to int16 [-32768, 32767]
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype != np.int16:
                    audio_data = audio_data.astype(np.int16)
                audio_bytes = audio_data.tobytes()
            else:
                audio_bytes = audio_data

            # Write WAV file
            with wave.open(str(file_path), 'wb') as wav_file:  # type: ignore[union-attr]
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)

            logger.debug(f"[AudioLogger] Saved audio to {file_path}")
        except Exception as e:
            logger.error(f"[AudioLogger] Error saving audio to {file_path}: {e}")
            raise

    def _save_metadata_json(self, metadata: dict, file_path: Path):
        """Save metadata to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"[AudioLogger] Saved metadata to {file_path}")
        except Exception as e:
            logger.error(f"[AudioLogger] Error saving metadata to {file_path}: {e}")
            raise

    def clear_user_audio_buffer(self):
        """ 
        Clear the user audio buffer if the user stopped speaking detected by VAD.
        """
        # Clear turn buffers if logging wasn't completed (e.g., no final transcription)
        if len(self.turn_audio_buffer) > 0 or len(self.turn_transcription_buffer) > 0:
            logger.debug("[AudioLogger] Clearing turn audio and transcription buffers due to VAD user stopped speaking")
            self.turn_audio_buffer = []
            self.turn_transcription_buffer = []

    def stage_user_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        transcription: str,
        sample_rate: int = 16000,
        num_channels: int = 1,
        is_first_frame: bool = False,
        is_final: bool = True,
        additional_metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Stage log user audio and transcription (from STT).
        This data will be saved when the turn is complete by `log_user_audio` method.

        Args:
            audio_data: Raw audio data as bytes or numpy array
            transcription: Transcribed text
            sample_rate: Audio sample rate in Hz (default: 16000)
            num_channels: Number of audio channels (default: 1)
            is_final: Whether this is a final transcription (default: True)
            additional_metadata: Additional metadata to include

        Returns:
            Dictionary with logged file paths, or None if logging is disabled
        """
        if not self.enabled:
            return None

        try:
            # Get counter and generate filenames
            counter = self._get_next_counter("user")
            timestamp_now = datetime.now()
            base_name = f"{counter:05d}_{timestamp_now.strftime('%H%M%S')}"

            audio_file = self.user_dir / f"{base_name}.wav"
            metadata_file = self.user_dir / f"{base_name}.json"

            # Save audio
            self._staged_audio_data = audio_data

            if is_first_frame or self._staged_metadata is None:
                _start_time = self.get_time_from_start_of_session(timestamp=timestamp_now)
            else:
                # start_time is stored as float (seconds from session start), not ISO string
                _start_time = self._staged_metadata["start_time"]
                
            if isinstance(audio_data, bytes):
                audio_duration_sec = len(audio_data) / (sample_rate * num_channels * 2)
            else:
                audio_duration_sec = len(audio_data) / sample_rate

            _end_time = _start_time + audio_duration_sec

            # Prepare metadata
            self._staged_metadata = {
                "base_name": base_name,
                "counter": counter,
                "turn_index": self._turn_index,
                "speaker": "user",
                "timestamp": timestamp_now.isoformat(),
                "start_time": _start_time,
                "end_time": _end_time,
                "transcription": transcription,
                "is_final": is_final,
                "audio_file": audio_file.name,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "audio_duration_sec": audio_duration_sec,
            }

            if additional_metadata:
                self._staged_metadata.update(additional_metadata)

            return {
                "audio_file": str(audio_file),
                "metadata_file": str(metadata_file),
                "counter": counter,
            }

        except Exception as e:
            logger.error(f"Error logging user audio: {e}")
            return None

    def save_user_audio(self):
        """Save the user audio to the disk."""
        # Safety check: ensure staged metadata and audio data exist
        if self._staged_metadata is None:
            logger.warning("[AudioLogger] Attempted to save user audio but no staged metadata found")
            return
        
        if self._staged_audio_data is None:
            logger.warning("[AudioLogger] Attempted to save user audio but no staged audio data found")
            return
        
        try:
            audio_file = self.user_dir / f"{self._staged_metadata['base_name']}.wav"
            metadata_file = self.user_dir / f"{self._staged_metadata['base_name']}.json"

            self._save_audio_wav(
                audio_data=self._staged_audio_data, file_path=audio_file, sample_rate=self._staged_metadata["sample_rate"]
            )

            self._save_metadata_json(metadata=self._staged_metadata, file_path=metadata_file)
            logger.info(
                f"[AudioLogger] Saved user audio #{self._staged_metadata['counter']}: '{self._staged_metadata['transcription'][:50]}{'...' if len(self._staged_metadata['transcription']) > 50 else ''}'"
            )
            
            # Append to stereo conversation (right channel = user)
            # Apply offset to advance user audio to compensate for processing lag
            user_stereo_start_time = max(0.0, self._staged_metadata["start_time"] + self._user_audio_stereo_offset)
            self._append_to_stereo_conversation(
                audio_data=self._staged_audio_data,
                channel="right",
                start_time=user_stereo_start_time,
                sample_rate=self._staged_metadata["sample_rate"],
            )
            
            # Update session metadata
            with self._lock:
                self.session_metadata["user_entries"].append(self._staged_metadata)
                self._save_session_metadata()

            self.clear_user_audio_buffer()
            
            # Clear staged data after successful save
            self._staged_metadata = None
            self._staged_audio_data = None
        except Exception as e:
            logger.error(f"[AudioLogger] Error saving user audio: {e}")
            raise

    def log_agent_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        text: str,
        sample_rate: int = 22050,
        num_channels: int = 1,
        additional_metadata: Optional[dict] = None,
        tts_generation_time: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Log agent audio and text (from TTS).

        Args:
            audio_data: Generated audio data as bytes or numpy array
            text: Input text that was synthesized
            sample_rate: Audio sample rate in Hz (default: 22050)
            num_channels: Number of audio channels (default: 1)
            additional_metadata: Additional metadata to include
            tts_generation_time: Time when TTS generation started (seconds from session start).
                                Used to calculate actual start_time for first segment of a turn.

        Returns:
            Dictionary with logged file paths, or None if logging is disabled
        """
        if not self.enabled:
            return None

        try:
            # Get counter and generate filenames
            counter = self._get_next_counter("agent")
            timestamp_now = datetime.now()
            base_name = f"{counter:05d}_{timestamp_now.strftime('%H%M%S')}"

            audio_file = self.agent_dir / f"{base_name}.wav"
            metadata_file = self.agent_dir / f"{base_name}.json"

            # Save audio
            self._save_audio_wav(audio_data, audio_file, sample_rate, num_channels)

            # Calculate audio duration
            audio_duration_sec = (
                len(audio_data) / (sample_rate * num_channels * 2)
                if isinstance(audio_data, bytes)
                else len(audio_data) / sample_rate
            )
            
            # Determine start_time based on previous segment in the same turn
            # If this is the first segment of the turn, use tts_generation_time
            # Otherwise, use the previous segment's end_time for sequential playback
            start_time = None
            with self._lock:
                agent_entries = self.session_metadata["agent_entries"]
                # agent_entries is a list of turns, each turn is a list of segments
                if agent_entries and agent_entries[-1]:  # If there's a current turn with segments
                    last_segment = agent_entries[-1][-1]  # Last segment of last turn
                    if last_segment["turn_index"] == self._turn_index:
                        # Same turn - start after previous segment ends
                        start_time = last_segment["end_time"]
            
            if start_time is None:
                # First segment of the turn - use agent_turn_start_time (from BotStartedSpeakingFrame)
                # This is more accurate than tts_generation_time as it reflects actual playback start
                if self._agent_turn_start_time is not None:
                    start_time = self._agent_turn_start_time
                elif tts_generation_time is not None:
                    # Fallback to tts_generation_time if agent_turn_start_time not set
                    start_time = tts_generation_time
                else:
                    start_time = self.get_time_from_start_of_session(timestamp=timestamp_now)
            
            end_time = start_time + audio_duration_sec

            # Prepare metadata
            # cutoff_time is None by default (no interruption)
            # It will be set by set_agent_cutoff_time() if TTS is interrupted
            metadata = {
                "base_name": base_name,
                "counter": counter,
                "turn_index": self._turn_index,
                "speaker": "agent",
                "timestamp": timestamp_now.isoformat(),
                "start_time": start_time,
                "end_time": end_time,
                "cutoff_time": None,  # None means not interrupted; float if interrupted
                "text": text,
                "audio_file": audio_file.name,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "audio_duration_sec": audio_duration_sec,
            }

            if additional_metadata:
                metadata.update(additional_metadata)

            # Save metadata
            self._save_metadata_json(metadata, metadata_file)

            # Append to stereo conversation (left channel = agent)
            self._append_to_stereo_conversation(
                audio_data=audio_data,
                channel="left",
                start_time=start_time,
                sample_rate=sample_rate,
            )

            # Update session metadata
            # agent_entries is a list of turns, each turn is a list of segments
            with self._lock:
                agent_entries = self.session_metadata["agent_entries"]
                # Check if we need to start a new turn or append to existing turn
                if not agent_entries or agent_entries[-1][-1]["turn_index"] != self._turn_index:
                    # Start a new turn (new sublist)
                    agent_entries.append([metadata])
                else:
                    # Append to current turn
                    agent_entries[-1].append(metadata)
                self._save_session_metadata()

            logger.info(f"[AudioLogger] Logged agent audio #{counter}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            return {
                "audio_file": str(audio_file),
                "metadata_file": str(metadata_file),
                "counter": counter,
            }

        except Exception as e:
            logger.error(f"[AudioLogger] Error logging agent audio: {e}")
            return None

    def set_agent_cutoff_time(self, cutoff_time: Optional[float] = None):
        """
        Set the cutoff time for the most recent agent audio entry.
        
        This method should be called when TTS is interrupted by user speech.
        The cutoff_time represents when the agent audio was actually cut off,
        which may be earlier than the natural end_time.
        
        Args:
            cutoff_time: The cutoff time in seconds from session start.
                        If None, uses current time from session start.
        """
        if not self.enabled:
            return
        
        if cutoff_time is None:
            cutoff_time = self.get_time_from_start_of_session()
        
        with self._lock:
            agent_entries = self.session_metadata["agent_entries"]
            if not agent_entries or not agent_entries[-1]:
                logger.warning("[AudioLogger] No agent entries to set cutoff time")
                return
            
            # Get the current turn (last sublist) and update ALL segments in it
            current_turn = agent_entries[-1]
            turn_index = current_turn[0]["turn_index"]
            
            # Update cutoff_time for ALL segments in the current turn
            for segment in current_turn:
                segment["cutoff_time"] = cutoff_time
                # Also update individual JSON files
                try:
                    metadata_file = self.agent_dir / f"{segment['base_name']}.json"
                    self._save_metadata_json(segment, metadata_file)
                except Exception as e:
                    logger.error(f"[AudioLogger] Error updating agent cutoff time for segment: {e}")
            
            # Truncate the stereo buffer (left channel = agent) at the cutoff point
            cutoff_sample = int(cutoff_time * self._stereo_sample_rate)
            if cutoff_sample < len(self._stereo_audio_buffer_left):
                # Zero out agent audio after cutoff point
                for i in range(cutoff_sample, len(self._stereo_audio_buffer_left)):
                    self._stereo_audio_buffer_left[i] = 0.0
                logger.debug(
                    f"[AudioLogger] Truncated agent stereo buffer at sample {cutoff_sample} "
                    f"(cutoff_time={cutoff_time:.3f}s)"
                )
            
            logger.info(
                f"[AudioLogger] Set cutoff_time={cutoff_time:.3f}s for turn {turn_index} "
                f"({len(current_turn)} segments)"
            )
            
            # Save updated session metadata
            self._save_session_metadata()

    def _save_session_metadata(self):
        """Save the session metadata to disk."""
        if not self.enabled:
            return

        try:
            metadata_file = self.session_dir / "session_metadata.json"
            self.session_metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata_json(self.session_metadata, metadata_file)
        except Exception as e:
            logger.error(f"[AudioLogger] Error saving session metadata: {e}")

    def finalize_session(self):
        """Finalize the session and save final metadata."""
        if not self.enabled:
            return

        # Save global user audio before finalizing
        self.save_global_user_audio()
        
        # Save stereo conversation before finalizing
        self.save_stereo_conversation()

        self.session_metadata["end_time"] = datetime.now().isoformat()
        self.session_metadata["total_user_entries"] = self._user_counter
        self.session_metadata["total_agent_segments"] = self._agent_counter
        self.session_metadata["total_agent_turns"] = len(self.session_metadata["agent_entries"])
        self._save_session_metadata()
        logger.info(
            f"[AudioLogger] Session finalized: {self.session_id} "
            f"(User: {self._user_counter}, Agent: {self._agent_counter} segments in "
            f"{len(self.session_metadata['agent_entries'])} turns)"
        )

    def get_session_info(self) -> dict:
        """Get current session information."""
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "user_entries": self._user_counter,
            "agent_entries": self._agent_counter,
            "enabled": self.enabled,
        }

class RTVIAudioLoggerObserver(BaseObserver):
    def __init__(self, audio_logger: AudioLogger):
        super().__init__()
        self._audio_logger = audio_logger

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, TranscriptionFrame) and self._audio_logger:
            self._audio_logger.save_user_audio()
        # Call parent class's on_push_frame method
        await super().on_push_frame(data) 