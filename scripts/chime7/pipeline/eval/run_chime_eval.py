# script adapted from https://github.com/espnet/espnet/blob/master/egs2/chime7_task1/asr1/local/da_wer_scoring.py
import argparse
import glob
import json
import os
import pickle
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import jiwer
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict
from pyannote.core.utils.types import Label
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.matcher import HungarianMapper
from pyannote.metrics.segmentation import Annotation, Segment, Timeline
from pyannote.metrics.types import MetricComponents
from tabulate import tabulate
from tqdm import tqdm

from nemo.core.config import hydra_runner
from nemo.utils import logging


def compute_der(df_or_dict):
    if isinstance(df_or_dict, dict):
        der = (df_or_dict["false alarm"] + df_or_dict["missed detection"] + df_or_dict["confusion"]) / df_or_dict[
            "total"
        ]
    elif isinstance(df_or_dict, pd.DataFrame):
        der = (
            df_or_dict["false alarm"].sum() + df_or_dict["missed detection"].sum() + df_or_dict["confusion"].sum()
        ) / df_or_dict["total"].sum()
    else:
        raise NotImplementedError

    return der


def compute_jer(df_or_dict):
    if isinstance(df_or_dict, dict):
        jer = df_or_dict["speaker error"] / df_or_dict["speaker count"]
    elif isinstance(df_or_dict, pd.DataFrame):
        jer = df_or_dict["speaker error"].sum() / df_or_dict["speaker count"].sum()
    else:
        raise NotImplementedError

    return jer


class DERComputer(IdentificationErrorRate):
    """Modified from https://github.com/pyannote/
    pyannote-metrics/blob/14af03ca61527621cfc0a3ed7237cc2969681915/
    pyannote/metrics/diarization.py.
    Basically we want to avoid running multiple times (time intensive) uemify and
    the optimal mapping functions as these are also required for JER computation.
    Optimal mapping is also reused for deriving the WER.
    We exposed these functions in this class so the results can be reused."""

    def __init__(self, collar: float = 0.0, skip_overlap: bool = False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = HungarianMapper()

    def get_optimal_mapping(self, reference: Annotation, hypothesis: Annotation,) -> Dict[Label, Label]:
        mapping = self.mapper_(hypothesis, reference)
        mapped = hypothesis.rename_labels(mapping=mapping)
        return mapped, mapping

    def get_uemified(self, reference, hypothesis, uem: Optional[Timeline] = None):
        reference, hypothesis, uem = self.uemify(
            reference, hypothesis, uem=uem, collar=self.collar, skip_overlap=self.skip_overlap, returns_uem=True,
        )

        return reference, hypothesis, uem

    def compute_der(self, reference: Annotation, mapped: Annotation, uem: Optional[Timeline] = None, **kwargs):
        components = super(DERComputer, self).compute_components(
            reference, mapped, uem=uem, collar=0.0, skip_overlap=False, **kwargs
        )
        der = compute_der(components)
        components.update({"diarization error rate": der})
        return components


class JERComputer(DiarizationErrorRate):
    """Modified from https://github.com/pyannote/
    pyannote-metrics/blob/14af03ca61527621cfc0a3ed7237cc2969681915/
    pyannote/metrics/diarization.py.
    This class assumes that optimal mapping and uemify have been already applied.
    We apply them before for DER computation and reuse them here."""

    def __init__(self, collar=0.0, skip_overlap=False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = HungarianMapper()

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "speaker count",
            "speaker error",
        ]

    def compute_jer(self, reference, hypothesis, mapping):
        detail = self.init_components()
        for ref_speaker in reference.labels():
            hyp_speaker = mapping.get(ref_speaker, None)
            if hyp_speaker is None:
                jer = 1.0
            else:
                r = reference.label_timeline(ref_speaker)
                h = hypothesis.label_timeline(hyp_speaker)
                total = r.union(h).support().duration()
                fa = h.duration() - h.crop(r).duration()
                miss = r.duration() - r.crop(h).duration()
                jer = (fa + miss) / total
            detail["speaker count"] += 1
            detail["speaker error"] += jer

        jer = compute_jer(detail)
        detail.update({"Jaccard error rate": jer})
        return detail


def compute_wer(df_or_dict):
    if isinstance(df_or_dict, dict):
        wer = (df_or_dict["substitutions"] + df_or_dict["deletions"] + df_or_dict["insertions"]) / (
            df_or_dict["substitutions"] + df_or_dict["deletions"] + df_or_dict["hits"]
        )
    elif isinstance(df_or_dict, pd.DataFrame):
        wer = (df_or_dict["substitutions"].sum() + df_or_dict["deletions"].sum() + df_or_dict["insertions"].sum()) / (
            df_or_dict["substitutions"].sum() + df_or_dict["deletions"].sum() + df_or_dict["hits"].sum()
        )
    else:
        raise NotImplementedError

    return wer


def compute_diar_errors(hyp_segs, ref_segs, uem_boundaries=None, collar=0.5):
    # computing all diarization errors for each session here.
    # find optimal mapping too, which will then be used to find the WER.
    if uem_boundaries is not None:
        uem = Timeline([Segment(start=uem_boundaries[0], end=uem_boundaries[-1])])
    else:
        uem = None

    def to_annotation(segs):
        out = Annotation()
        for s in segs:
            speaker = s["speaker"]
            start = float(s["start_time"])
            end = float(s["end_time"])
            out[Segment(start, end)] = speaker
        return out

    hyp_annotation = to_annotation(hyp_segs)
    ref_annotation = to_annotation(ref_segs)

    der_computer = DERComputer(collar=collar, skip_overlap=False)
    reference, hypothesis, uem = der_computer.get_uemified(ref_annotation, hyp_annotation, uem=uem)
    mapped, mapping = der_computer.get_optimal_mapping(reference, hypothesis)
    der_score = der_computer.compute_der(reference, mapped, uem=uem)
    # avoid uemify again with custom class
    jer_compute = JERComputer(collar=collar, skip_overlap=False)  # not optimal computationally
    jer_score = jer_compute.compute_jer(reference, hypothesis, {v: k for k, v in mapping.items()})

    # error analysis here
    error_compute = IdentificationErrorAnalysis(collar=collar, skip_overlap=False)
    reference, hypothesis, errors = error_compute.difference(ref_annotation, hyp_annotation, uem=uem, uemified=True)

    return mapping, der_score, jer_score, reference, hypothesis, errors


def log_asr(output_folder, hyp_segs, ref_segs):
    """
    Dump re-ordered hypothesis as JSON files to allow for analyze errors.
    This is done for each session.
    """

    def flatten_segs(spk2utts):
        all = []
        for k in spk2utts.keys():
            all.extend(spk2utts[k])
        return all

    hyp_segs = flatten_segs(hyp_segs)
    ref_segs = flatten_segs(ref_segs)
    hyp_segs = sorted(hyp_segs, key=lambda x: float(x["start_time"]))
    ref_segs = sorted(ref_segs, key=lambda x: float(x["start_time"]))
    with open(os.path.join(output_folder, "hyp_reordered.json"), "w") as f:
        json.dump(hyp_segs, f, indent=4)
    with open(os.path.join(output_folder, "ref.json"), "w") as f:
        json.dump(ref_segs, f, indent=4)


def compute_asr_errors(output_folder, hyp_segs, ref_segs, mapping=None, uem=None):
    if mapping is not None:  # using diarization
        hyp_segs_reordered = []
        for s in hyp_segs:
            new_segment = deepcopy(s)
            c_speaker = new_segment["speaker"]
            if c_speaker not in mapping.keys():
                mapping[c_speaker] = "FA_" + c_speaker  # false speaker
            new_segment["speaker"] = mapping[c_speaker]
            hyp_segs_reordered.append(new_segment)
        hyp_segs = hyp_segs_reordered

    def spk2utts(segs, uem=None):
        st_uem, end_uem = uem
        spk2utt = OrderedDict({k["speaker"]: [] for k in segs})
        for s in segs:
            start = float(s["start_time"])
            end = float(s["end_time"])
            # discard segments whose end is in the uem.
            if uem is not None and (end < st_uem or start > end_uem):
                continue
            spk2utt[s["speaker"]].append(s)

        for spk in spk2utt.keys():
            spk2utt[spk] = sorted(spk2utt[spk], key=lambda x: float(x["start_time"]))
        return spk2utt

    hyp = spk2utts(hyp_segs, uem)
    ref = spk2utts(ref_segs, uem)

    log_asr(output_folder, hyp, ref)

    if mapping is not None:
        # check if they have same speakers
        false_speakers = set(hyp.keys()).difference(set(ref.keys()))
        if false_speakers:
            for f_spk in list(false_speakers):
                ref[f_spk] = [{"words": "", "speaker": f_spk}]
        missed_speakers = set(ref.keys()).difference(set(hyp.keys()))
        if missed_speakers:
            for m_spk in list(missed_speakers):
                hyp[m_spk] = [{"words": "", "speaker": m_spk}]

    tot_stats = {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
    speakers_stats = []
    for spk in ref.keys():
        cat_refs = " ".join([x["words"] for x in ref[spk]])
        cat_hyps = " ".join([x["words"] for x in hyp[spk]])
        if len(cat_refs) == 0:
            # need this because jiwer cannot handle empty refs
            ldist = {
                "hits": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": len(cat_hyps.split()),
            }
        else:
            wordsout = jiwer.process_words(cat_refs, cat_hyps)
            ldist = {k: v for k, v in wordsout.__dict__.items() if k in tot_stats.keys()}

        ldist.update(
            {"speaker": spk, "tot utterances ref": len(ref[spk]), "tot utterances hyp": len(hyp[spk]),}
        )
        speakers_stats.append(ldist)
        for k in tot_stats.keys():
            tot_stats[k] += ldist[k]

    c_wer = compute_wer(tot_stats)
    tot_stats.update({"wer": c_wer})

    return tot_stats, speakers_stats


def log_diarization(sess_folder, reference, hypothesis, errors):
    """
    Logging diarization output to the specified output folder for each session.
    This is useful for analyzing errors.
    """

    with open(os.path.join(sess_folder, "diar_errors_summary.txt"), "w") as f:
        print(errors.chart(), file=f)

    # save to disk as you may want to use them to analyze errors.
    # pyannote has some useful visualization features.
    with open(os.path.join(sess_folder, "diar_errors_pyannote.pickle"), "wb") as handle:
        pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(sess_folder, "diar_ref_pyannote.pickle"), "wb") as handle:
        pickle.dump(reference, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(sess_folder, "diar_hyp_pyannote.pickle"), "wb") as handle:
        pickle.dump(hypothesis, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(sess_folder, "errors.rttm"), "w") as f:
        f.write(errors.to_rttm())
    with open(os.path.join(sess_folder, "reference.rttm"), "w") as f:
        f.write(reference.to_rttm())
    with open(os.path.join(sess_folder, "hypothesis.rttm"), "w") as f:
        f.write(hypothesis.to_rttm())


def score(
    hyp_json,
    reference_jsons,
    scenario_tag,
    output_folder,
    uem_file=None,
    collar=0.5,
    use_diarization=True,
    allow_subset_eval=True,
):
    scenario_dir = os.path.join(output_folder, scenario_tag)
    Path(scenario_dir).mkdir(parents=True, exist_ok=True)

    if uem_file is not None:
        with open(uem_file, "r") as f:
            lines = f.readlines()
        lines = [x.rstrip("\n") for x in lines]
        uem2sess = {}

        for x in lines:
            items = x.split(" ")
            if len(items) == 3:
                sess_id, start, stop = items
            else:
                sess_id, _, start, stop = items
            uem2sess[sess_id] = (float(start), float(stop))

    # load all reference jsons
    refs = []
    for j in reference_jsons:
        with open(j, "r") as f:
            segments = json.load(f)
            if scenario_tag == "mixer6":
                for i in range(len(segments)):
                    if "session_id" not in segments[i]:
                        segments[i]["session_id"] = Path(j).stem  # add session id if not exists
            refs.extend(segments)

    split_tag = "_" if scenario_tag != "mixer6" else None
    hyps = []
    for j in hyp_json:
        hyps.extend(parse_nemo_json(j, split_tag=split_tag))

    def get_sess2segs(segments):
        out = {}
        for x in segments:
            c_sess = x["session_id"]
            if c_sess not in out.keys():
                out[c_sess] = []
            out[c_sess].append(x)
        return out

    h_sess2segs = get_sess2segs(hyps)
    r_sess2segs = get_sess2segs(refs)

    intersection_list = set(h_sess2segs.keys()).intersection(set(r_sess2segs.keys()))

    if not (h_sess2segs.keys() == r_sess2segs.keys()):
        if allow_subset_eval:
            ### To make subset evaluation without errors
            r_sess2segs_new = {}
            for key in r_sess2segs.keys():
                if key in intersection_list:
                    r_sess2segs_new[key] = r_sess2segs[key]
            r_sess2segs = r_sess2segs_new
            # raise Warning(f"Subset evaluation is being performed. {len(r_sess2segs.keys())} sessions are being evaluated.")
        else:
            raise RuntimeError(
                "Hypothesis JSON does not have all sessions as in the reference JSONs."
                "The sessions that are missing: {}".format(set(h_sess2segs.keys()).difference(set(r_sess2segs.keys())))
            )

    all_sess_stats = []
    all_spk_stats = []
    sessions = list(r_sess2segs.keys())
    for indx in tqdm(range(len(sessions))):
        session = sessions[indx]
        hyp_segs = sorted(h_sess2segs[session], key=lambda x: float(x["start_time"]))
        ref_segs = sorted(r_sess2segs[session], key=lambda x: float(x["start_time"]))
        # compute diarization error and best permutation here
        if uem_file is not None:
            c_uem = uem2sess[session]
        else:
            c_uem = None

        sess_dir = os.path.join(scenario_dir, session)
        Path(sess_dir).mkdir(exist_ok=True)

        if use_diarization:
            (mapping, der_score, jer_score, reference, hypothesis, errors,) = compute_diar_errors(
                hyp_segs, ref_segs, c_uem, collar=collar
            )

            log_diarization(sess_dir, reference, hypothesis, errors)
            # save ref hyps and errors in a folder
            # save also compatible format for audacity
            c_sess_stats = {
                "session id": session,
                "scenario": scenario_tag,
                "num spk hyp": len(hypothesis.labels()),
                "num spk ref": len(reference.labels()),
                "tot utterances hyp": len(hypothesis),
                "tot utterances ref": len(reference),
            }

            c_sess_stats.update({k: v for k, v in der_score.items()})
            c_sess_stats.update({k: v for k, v in jer_score.items()})
            asr_err_sess, asr_err_spk = compute_asr_errors(sess_dir, hyp_segs, ref_segs, mapping, uem=c_uem)

        else:
            if not len(hyp_segs) == len(ref_segs):
                warnings.warn(
                    "If oracle diarization was used, "
                    "I expect the hypothesis to have the same number "
                    "of utterances as the "
                    "reference. Have you discarded some utterances "
                    "(e.g. too long) ? "
                    "These will be counted as deletions so be careful !"
                )
            asr_err_sess, asr_err_spk = compute_asr_errors(sess_dir, hyp_segs, ref_segs, mapping=None, uem=c_uem)
            c_sess_stats = {
                "session id": session,
                "scenario": scenario_tag,
                "num spk hyp": len(set([x["speaker"] for x in hyp_segs])),
                "num spk ref": len(set([x["speaker"] for x in ref_segs])),
                "tot utterances hyp": len(hyp_segs),
                "tot utterances ref": len(ref_segs),
            }
            # add to each speaker, session id and scenario
        [x.update({"session_id": session, "scenario": scenario_tag}) for x in asr_err_spk]
        c_sess_stats.update(asr_err_sess)
        all_sess_stats.append(c_sess_stats)
        all_spk_stats.extend(asr_err_spk)

    sess_df = pd.DataFrame(all_sess_stats)
    # pretty print because it may be useful
    print(tabulate(sess_df, headers="keys", tablefmt="psql"))
    # accumulate for all scenario
    scenario_wise_df = sess_df.sum(0).to_frame().transpose()
    scenario_wise_df["scenario"] = scenario_tag
    # need to recompute these
    scenario_wer = compute_wer(sess_df)
    scenario_wise_df["wer"] = scenario_wer
    if use_diarization:
        scenario_der = compute_der(sess_df)
        scenario_wise_df["diarization error rate"] = scenario_der
        scenario_jer = compute_jer(sess_df)
        scenario_wise_df["Jaccard error rate"] = scenario_jer
    del scenario_wise_df["session id"]  # delete session

    return scenario_wise_df, all_sess_stats, all_spk_stats


def parse_nemo_json(json_file, split_tag=None):
    hyp_segs = []
    with open(json_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "session_id" not in entry:
                audio_file = entry["audio_filepath"]
                if isinstance(audio_file, list):
                    audio_file = audio_file[0]
                session_id = Path(audio_file).stem
                if "_CH" in session_id:
                    session_id = session_id.split("_CH")[0]
                if split_tag:
                    session_id = session_id.split(split_tag)[0]
                entry["session_id"] = session_id
            hyp_segs.append(
                {
                    "speaker": entry["speaker"],
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"],
                    "words": entry["pred_text"],
                    "session_id": entry["session_id"],
                    "audio_filepath": entry["audio_filepath"],
                }
            )
    return hyp_segs


def run_chime_evaluation(cfg):
    if not hasattr(jiwer, "process_words"):
        raise RuntimeError("Please update jiwer package to 3.0.0 version.")

    for subset in cfg.subsets:
        asr_output_dir = Path(cfg.asr_output_dir) / subset
        eval_output_dir = Path(cfg.eval_output_dir) / subset
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_cfg = OmegaConf.to_container(cfg.eval, resolve=True)
        eval_cfg = DictConfig(eval_cfg)
        with open_dict(eval_cfg):
            eval_cfg.output_folder = str(eval_output_dir)
            eval_cfg.hyp_folder = str(asr_output_dir)
            eval_cfg.partition = subset

        spk_wise_df = []
        sess_wise_df = []
        scenario_wise_df = []
        scenarios = ["chime6", "dipco", "mixer6"]
        Path(eval_cfg.output_folder).mkdir(exist_ok=True)
        for indx, scenario in enumerate(scenarios):
            # hyp_json = os.path.join(eval_cfg.hyp_folder, scenario + ".json")
            hyp_json = glob.glob(os.path.join(eval_cfg.hyp_folder, scenario, "*.json",))

            if len(hyp_json) == 0 and bool(eval_cfg.ignore_missing):
                warnings.warn("I cannot find {}, so I will skip {} scenario".format(hyp_json, scenario))
                warnings.warn("Macro scores will not be computed.")
                continue

            print("###################################################")
            print("### Scoring {} Scenario ###########################".format(scenario))
            print("###################################################")
            reference_json = glob.glob(
                os.path.join(eval_cfg.dasr_root, scenario, "transcriptions_scoring", eval_cfg.partition, "*.json",)
            )
            uem = os.path.join(eval_cfg.dasr_root, scenario, "uem", eval_cfg.partition, "all.uem")
            assert len(reference_json) > 0, "Reference JSONS not found, is the path {} correct ?".format(
                os.path.join(eval_cfg.dasr_root, scenario, "transcriptions_scoring", eval_cfg.partition, "*.json",)
            )
            scenario_stats, all_sess_stats, all_spk_stats = score(
                hyp_json,
                reference_json,
                scenario,
                eval_cfg.output_folder,
                uem,
                collar=float(eval_cfg.collar / 1000),
                use_diarization=eval_cfg.diarization,
            )
            sess_wise_df.extend(all_sess_stats)
            spk_wise_df.extend(all_spk_stats)
            scenario_wise_df.append(scenario_stats)

        sess_wise_df = pd.DataFrame(sess_wise_df)
        spk_wise_df = pd.DataFrame(spk_wise_df)
        scenario_wise_df = pd.concat(scenario_wise_df, axis=0)
        sess_wise_df.to_csv(os.path.join(eval_cfg.output_folder, "sessions_stats.csv"))
        spk_wise_df.to_csv(os.path.join(eval_cfg.output_folder, "speakers_stats.csv"))
        scenario_wise_df.to_csv(os.path.join(eval_cfg.output_folder, "scenarios_stats.csv"))

        # compute scenario-wise metrics
        print("###################################################")
        print("### Metrics for all Scenarios ###")
        print("###################################################")
        print(tabulate(scenario_wise_df, headers="keys", tablefmt="psql"))
        # if not skip_macro:
        print("####################################################################")
        print("### Macro-Averaged Metrics across all Scenarios (Ranking Metric) ###")
        print("####################################################################")
        macro_avg = scenario_wise_df.drop("scenario", axis="columns").mean(0).to_frame().T
        macro_avg.insert(0, "scenario", "macro-average")
        macro_wer = macro_avg['wer'][0]
        print(tabulate(macro_avg, headers="keys", tablefmt="psql"))
        with open(os.path.join(eval_cfg.output_folder, "macro_wer.txt"), "w") as f:
            f.write(str(macro_wer))
        return macro_wer


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    run_chime_evaluation(cfg)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
