import glob
import json
import os
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.core.config import hydra_runner
from nemo.utils import logging


from .enhance_cuts import enhance_cuts as nemo_enhance_cuts
from .lhoste_cuts import simple_cut, trim_to_supervisions
from .lhoste_manifests import prepare_chime_manifests
from .mic_rank import get_gss_mic_ranks


def convert_diar_results_to_falign(
    diarization_dir: str, diarization_params: str, output_dir: str, subsets: list = ['dev']
):
    # Assumption:
    # Output of diarization is organized in 3 subdirectories, with each subdirectory corresponding to one scenario (chime6, dipco, mixer6)
    diarization_system = diarization_dir.split('/')[-1]

    if output_dir == "":
        output_dir = f'./alignments/{diarization_system}-{diarization_params}'

    scenario_dirs = glob.glob(diarization_dir + '/*')

    # assert len(scenario_dirs) == 3, f'Expected 3 subdirectories, found {len(scenario_dirs)}'
    none_useful_fields = ['audio_filepath', 'words', 'text', 'duration', 'offset']
    for scenario in ['chime6', 'dipco', 'mixer6']:
        for subset in subsets:
            # Currently, subdirectories don't have a uniform naming scheme
            # Therefore, we pick the subdirectory that has both scenario and subset in its name
            scenario_subset_dir = [sd for sd in scenario_dirs if scenario in sd]

            if len(scenario_subset_dir) == 0:
                print(f'No subdirectory found for {scenario} and {subset}')
                break
            else:
                scenario_subset_dir = scenario_subset_dir[0]

            # Grab manifests from the results of diarization
            manifests_dir = os.path.join(scenario_subset_dir, diarization_params)
            if not os.path.isdir(manifests_dir):
                manifests_dir = os.path.join(scenario_subset_dir, f"pred_jsons_{diarization_params}")

            manifests = list(glob.glob(manifests_dir + '/*.json'))
            if len(manifests) == 0:
                print(f'No manifests found in {manifests_dir}')
                continue

            # Process each manifest
            for manifest in manifests:
                manifest_name = os.path.basename(manifest)
                session_name = (
                    manifest_name.replace(scenario, '')
                    .replace('dev', '')
                    .replace('eval', '')
                    .replace('.json', '')
                    .strip('-')
                )
                new_manifest = os.path.join(output_dir, scenario, subset, session_name + '.json')

                if not os.path.isdir(os.path.dirname(new_manifest)):
                    os.makedirs(os.path.dirname(new_manifest))

                # read manifest
                try:
                    data = read_manifest(manifest)
                except json.decoder.JSONDecodeError:
                    data = json.load(open(manifest, 'r'))

                for item in data:
                    # not required
                    for k in none_useful_fields:
                        if k in item:
                            item.pop(k)
                    # set these to be consistent with the baseline falign manifests
                    item['session_id'] = session_name
                    item['words'] = 'placeholder'

                # dump the list in a json file (not JSONL as our manifests)
                with open(new_manifest, 'w') as f:
                    json.dump(data, f, indent=4, sort_keys=True)


def prepare_nemo_manifests(data_dir: str, audio_type: str = 'flac'):
    """
    Prepare NeMo manifests from GSS outputs
    """
    # Make sure we know scenario
    if 'chime6' in data_dir:
        scenario = 'chime6'
    elif 'dipco' in data_dir:
        scenario = 'dipco'
    elif 'mixer6' in data_dir:
        scenario = 'mixer6'
    else:
        raise ValueError(f'Unknown setup: {data_dir}')

    # Make sure we know subset
    if 'dev' in data_dir and 'eval' in data_dir:
        raise ValueError(f'Unknown subset: both dev and eval are in {data_dir}')
    elif 'dev' in data_dir:
        subset = 'dev'
    elif 'eval' in data_dir:
        subset = 'eval'
    else:
        raise ValueError(f'Unknown subset: {data_dir}')

    # Find all audio files
    audio_files = glob.glob(data_dir + f'/**/*.{audio_type}', recursive=True)
    logging.info(f"Found {len(audio_files)} *.{audio_type} files in {data_dir}")

    session_to_data = defaultdict(list)
    for audio_file in audio_files:
        # Each audio files is named session_id-speaker_id-start_time-end_time.{audio_type}
        # with start and end times in 1/100 seconds
        filename = os.path.basename(audio_file)
        if scenario == 'mixer6':
            # session_id has '-' in it
            parts = filename.replace(f'.{audio_type}', '').split('-')
            session_id = parts[0]  # keep only session, drop dev and mdm
            speaker_id = parts[-2]
            start_end_time = parts[-1]
        else:
            session_id, speaker_id, start_end_time = filename.replace(f'.{audio_type}', '').split('-')
        start_time, end_time = start_end_time.split('_')
        start_time = int(start_time) / 100
        end_time = int(end_time) / 100
        session_to_data[session_id].append(
            {
                'speaker': speaker_id,
                'session_id': session_id,
                'start_time': str(start_time),
                'end_time': str(end_time),
                'audio_filepath': os.path.relpath(path=audio_file, start=data_dir),
            }
        )

    for session_id, data in session_to_data.items():
        manifest_file = os.path.join(data_dir, f'{scenario}-{subset}-{session_id}.json')
        write_manifest(manifest_file, data)


def run_gss_process(cfg):
    diar_output_dir = Path(cfg.diar_base_dir, cfg.diar_config)
    gss_output_dir = Path(cfg.gss_output_dir, "processed", f"{cfg.diar_config}-{cfg.diar_param}")
    gss_output_dir.mkdir(parents=True, exist_ok=True)
    alignments_output_dir = Path(cfg.gss_output_dir, "alignments", f"{cfg.diar_config}-{cfg.diar_param}")
    alignments_output_dir.mkdir(parents=True, exist_ok=True)
    manifests_output_dir = Path(cfg.gss_output_dir, "manifests/lhotse", f"{cfg.diar_config}-{cfg.diar_param}")
    manifests_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Convert NeMo diarization output to falign format")
    convert_diar_results_to_falign(
        diarization_dir=str(diar_output_dir),
        diarization_params=cfg.diar_param,
        output_dir=str(alignments_output_dir),
        subsets=cfg.subsets,
    )
    outputs = []
    for scenario in cfg.scenarios:
        for subset in cfg.subsets:
            # Prepare lhotse manifests from diarization output
            # NOTE:
            # Unfortunately, this scripts runs for "mdm" and "ihm" for the dev set.
            # This means it will always fail for "ihm", since we only have diarization output for "mdm".
            # It's safe to ignore the error, since we only need the "mdm" manifests.
            # The error in question: "AssertionError: No recordings left after fixing the manifests."
            # If the error appears twice for the same scenario, then something is wrong.
            #
            # NOTE 2:
            # The script will ignore segments shorter than 0.2 seconds.
            logging.info(f"Prepare manifests for {scenario}/{subset}...")
            alignments_dir = alignments_output_dir / scenario
            alignments_dir.mkdir(parents=True, exist_ok=True)
            prepare_chime_manifests(
                data_root=str(cfg.chime_data_root),
                diar_json=str(alignments_dir),
                scenario=scenario,
                subset=subset,
                output_root=str(manifests_output_dir),
                ignore_shorter=cfg.preprocess.ignore_shorter,
                text_norm=cfg.preprocess.text_norm,
            )

            manifest_dir = manifests_output_dir / scenario / subset
            manifest_dir.mkdir(parents=True, exist_ok=True)

            exp_dir = gss_output_dir / scenario / subset
            exp_dir.mkdir(parents=True, exist_ok=True)

            logging.info("Stage 0: Select a subset of channels")
            get_gss_mic_ranks(
                recordings=str(manifest_dir / f"{scenario}-mdm_recordings_{subset}.jsonl.gz"),
                supervisions=str(manifest_dir / f"{scenario}-mdm_supervisions_{subset}.jsonl.gz"),
                output_filename=str(exp_dir / f"{scenario}_{subset}_selected"),
                top_k=cfg.gss.top_k_channels,
                num_workers=cfg.num_workers,
            )

            recordings = str(exp_dir / f"{scenario}_{subset}_selected_recordings.jsonl.gz")
            supervisions = str(exp_dir / f"{scenario}_{subset}_selected_supervisions.jsonl.gz")
            cuts_manifest = str(exp_dir / f"cuts.jsonl.gz")
            logging.info("Stage 1: Prepare cut set")
            simple_cut(
                output_cut_manifest=cuts_manifest,
                recording_manifest=recordings,
                supervision_manifest=supervisions,
                force_eager=True,
            )

            logging.info("Stage 2: Trim cuts to supervisions (1 cut per supervision)")
            cuts_seg_manifest = str(exp_dir / f"cuts_per_segment.jsonl.gz")
            trim_to_supervisions(
                cuts=cuts_manifest, output_cuts=cuts_seg_manifest, keep_overlapping=False,
            )

            logging.info("Stage 3: Run GSS and prepare nemo manifests")
            enhanced_dir = str(exp_dir.absolute())
            nemo_enhance_cuts(
                cuts_per_recording=cuts_manifest,
                cuts_per_segment=cuts_seg_manifest,
                enhanced_dir=enhanced_dir,
                num_workers=cfg.num_workers,
                bss_iterations=cfg.gss.bss_iterations,
                context_duration=cfg.gss.context_duration,
                use_garbage_class=cfg.gss.use_garbage_class,
                min_segment_length=cfg.gss.min_segment_length,
                max_segment_length=cfg.gss.max_segment_length,
                max_batch_duration=cfg.gss.max_batch_duration,
                max_batch_cuts=cfg.gss.max_batch_cuts,
                num_buckets=cfg.gss.num_buckets,
                force_overwrite=cfg.gss.force_overwrite,
                duration_tolerance=cfg.gss.duration_tolerance,
                channels=cfg.gss.channels,
                torchaudio_backend=cfg.gss.torchaudio_backend,
                dereverb_filter_length=cfg.gss.dereverb_filter_length,
                mc_mask_min_db=cfg.gss.mc_mask_min_db,
                mc_postmask_min_db=cfg.gss.mc_postmask_min_db,
                dereverb_prediction_delay=cfg.gss.dereverb_prediction_delay,
                dereverb_num_iterations=cfg.gss.dereverb_num_iterations,
                mc_filter_type=cfg.gss.mc_filter_type,
                mc_filter_num_iterations=cfg.gss.mc_filter_num_iterations,
                mc_filter_postfilter=cfg.gss.mc_filter_postfilter,
            )

            prepare_nemo_manifests(enhanced_dir, cfg.audio_type)
            logging.info(f"NeMo manifests saved to: {enhanced_dir}")
            outputs.append(enhanced_dir)
    return outputs


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    outputs = run_gss_process(cfg)
    logging.info(f"NeMo manifests saved to: {outputs}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
