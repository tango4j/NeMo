"""
normalize text for final submission
"""

import re
import json
import argparse
import glob
import os
def read_manifest(manifest):
    data = []
    # try:
    #     f = open(manifest.get(), 'r', encoding='utf-8')
    # except:
    #     raise Exception(f"Manifest file could not be opened: {manifest}")
    # for line in f:
    #     item = json.loads(line)
    #     data.append(item)
    # f.close()
    with open(manifest, 'r') as f:
        for line in f:
            json_line = json.loads(line)  # parse json from line
            words = json_line['pred_text'] 
            del json_line['pred_text'], json_line['text'], json_line['audio_filepath']
            json_line['words'] = words
            data.append(json_line)
    return data

def write_manifest(output_path, target_manifest):
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile)
            outfile.write('\n')
            
def dump_json(output_path, target_manifest):
    with open(output_path, "w") as f:
        # f.write(json.dumps(entry, indent=4))
        json.dump(target_manifest, f, indent=4)
        # for entry in target_manifest:
        #     # Convert dictionary to JSON format with indent and write it to file
        #     f.write(json.dumps(entry, indent=4))
        #     # Write a newline character after each line
        #     f.write('\n')
            
def parse_args():
    parser = argparse.ArgumentParser(description='normalize text for final submission')
    parser.add_argument('--input_fp', type=str, required=True, help='input manifest file')
    args = parser.parse_args()
    return args

# def main(args):
def norm_text(input_fp, output_fp):
    manifest_items = read_manifest(input_fp)
    for item in manifest_items:
        # replace multiple spaces with single space
        item['words'] = re.sub(' +', ' ', item['words'])
        # replace '\u2047' with ''
        item['words'] = item['words'].replace('\u2047', '')
        # replace isolated aw with oh
        item['words'] = item['words'].replace(' aw ', ' oh ')
    
    # write manifest
    # write_manifest(output_fp, manifest_items)
    dump_json(output_fp, manifest_items)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nemo chime7 output folder to final submission format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input file path")
    parser.add_argument("--sub_json_foldername", type=str, required=True, help="sub_json_foldername")
    args = parser.parse_args()
    
    # args = parse_args()
    # original_folder = "/disk_b/datasets/chime7_final_submission/final_submission_nemo_json/main/system1/dev"
    original_folder = args.input_dir
    # output_folder = "/disk_b/datasets/chime7_final_submission/final_submission_nemo_json_normalized"
    output_folder = original_folder.replace(f"{args.sub_json_foldername}", f"{args.sub_json_foldername}_normalized")
    # glob folder to get the file list
    file_list = glob.glob(original_folder + "/**/*.json", recursive=True)
    # loop all the files
    for input_fp in file_list:
        # get basename from input_fp
        output_fp = input_fp.replace(original_folder, output_folder)
        up_folder = os.path.dirname(output_fp)
        os.makedirs(up_folder,  exist_ok=True)
        print(f"Normalizing \n {input_fp} \n to \n {output_fp}")
        norm_text(input_fp, output_fp)