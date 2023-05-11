
import argparse
from pathlib import Path
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

def main(args):
    for manifest_file in Path(args.input_manifest).glob("*.json"):
        manifest_data = read_manifest(manifest_file)
        split_tag = "_" if args.dataset != "mixer6" else None
        print(f"Updating manifest file: {manifest_file}")
        for i, entry in enumerate(manifest_data):
            if "session_id" not in entry:
                audio_file = entry["audio_filepath"]
                if isinstance(audio_file, list):
                    audio_file = audio_file[0]
                session_id = Path(audio_file).stem
                if "_CH" in session_id:
                    session_id = session_id.split("_CH")[0]
                if split_tag:
                    session_id = session_id.split(split_tag)[0]
                manifest_data[i]["session_id"] = session_id
            manifest_data[i]["words"] = "placeholder"
            manifest_data[i]["text"] = "placeholder"
        
        output_file = Path(args.output_dir) / f"{manifest_file.name.split('-')[-1]}"
        write_manifest(output_file, manifest_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest", type=str, required=True, help="Input manifest file",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name", choices=["chime6", "dipco", "mixer6"],
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)