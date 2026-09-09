"""Partition the corpus into training and evaluation utterance lists.

For each (talker x speaking style) pair, five utterances are drawn for the
evaluation list and the rest go to training. Writes `train.list` and `dev.list`
, one audio path per line.

Note that the split is at the level of utterances, not talkers: every talker
appears in both lists. 

    python split.py --audio-dir <corpus>/audio --out-dir lists_out \
        --exclude lists/excluded.txt --exclude-first
"""
import argparse

import os
import random
import re
from collections import defaultdict

def split_audio_files(audio_dir="./audio", dev_list_file="dev.list",
                      train_list_file="train.list", path_prefix="./audio",
                      exclude=(), seed=1111, exclude_first=False):
    """
    Split audio files into train and dev sets.
    For each speaker (s2-s55) and each type (l_ or p_), 
    randomly select 5 files for dev set, rest goes to train set.
    """
    
    # Set random seed 
    random.seed(seed)
    
    # Check if audio directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        return
    
    # Get all wav files
    wav_files = []
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            wav_files.append(file)
    
    print(f"Found {len(wav_files)} wav files")
    
    # Group files by speaker and type
    # Structure: {speaker: {type: [files]}}
    grouped_files = defaultdict(lambda: defaultdict(list))
    
    # Regex pattern to extract speaker and type
    # Pattern: s{number}_{type}_
    pattern = r'^(s\d+)_(l_|p_)'
    
    if exclude_first:
        skip_names = {os.path.basename(p) for p in exclude}
        wav_files = [f for f in wav_files if f not in skip_names]
        print(f"Excluded {len(skip_names)} utterance(s) before the draw")

    for file in wav_files:
        match = re.match(pattern, file)
        if match:
            speaker = match.group(1)  # e.g., "s2", "s3", etc.
            file_type = match.group(2)  # "l_" or "p_"
            grouped_files[speaker][file_type].append(file)
        else:
            print(f"Warning: File '{file}' doesn't match expected pattern")
    
    # Lists to store file paths
    dev_files = []
    train_files = []
    
    # Process each speaker and type combination
    for speaker in sorted(grouped_files.keys()):
        for file_type in sorted(grouped_files[speaker].keys()):
            files = grouped_files[speaker][file_type]
            
            print(f"Speaker: {speaker}, Type: {file_type}, Total files: {len(files)}")
            
            if len(files) < 5:
                print(f"Warning: Only {len(files)} files for {speaker}_{file_type}, using all for dev")
                dev_count = len(files)
            else:
                dev_count = 5
            
            # Randomly select files for dev set
            dev_selected = random.sample(files, dev_count)
            dev_files.extend([os.path.join(path_prefix, f) for f in dev_selected])
            
            # Remaining files go to train set
            train_selected = [f for f in files if f not in dev_selected]
            train_files.extend([os.path.join(path_prefix, f) for f in train_selected])
            
            print(f"  Dev: {len(dev_selected)}, Train: {len(train_selected)}")
    
    # Drop the excluded utterances. 
    if exclude and not exclude_first:
        dropped = sum(1 for p in dev_files + train_files if p in exclude)
        dev_files = [p for p in dev_files if p not in exclude]
        train_files = [p for p in train_files if p not in exclude]
        print(f"Excluded {dropped} utterance(s)")

    # Write dev list
    with open(dev_list_file, 'w') as f:
        for file_path in sorted(dev_files):
            f.write(file_path + '\n')
    
    # Write train list
    with open(train_list_file, 'w') as f:
        for file_path in sorted(train_files):
            f.write(file_path + '\n')
    
    print(f"\nSplit completed!")
    print(f"Dev files: {len(dev_files)} (saved to {dev_list_file})")
    print(f"Train files: {len(train_files)} (saved to {train_list_file})")
    
    # Print summary by speaker and type
    print("\nSummary by speaker and type:")
    for speaker in sorted(grouped_files.keys()):
        for file_type in sorted(grouped_files[speaker].keys()):
            files = grouped_files[speaker][file_type]
            dev_count = min(5, len(files))
            train_count = len(files) - dev_count
            print(f"{speaker}_{file_type}: {dev_count} dev, {train_count} train")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audio-dir", required=True, help="the corpus audio directory")
    ap.add_argument("--out-dir", required=True, help="where dev.list and train.list go")
    ap.add_argument("--seed", type=int, default=1111,
                    help="the draw is seeded with this")
    ap.add_argument("--exclude", default=None,
                    help="file of utterance paths to drop")
    ap.add_argument("--exclude-first", action="store_true",
                    help="drop them before the draw")
    ap.add_argument("--path-prefix", default="./audio",
                    help="prefix written into the lists, independent of --audio-dir "
                         "so the output matches the shipped lists whatever path you pass")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    skip = set()
    if a.exclude:
        skip = {ln.strip() for ln in open(a.exclude)
                if ln.strip() and not ln.lstrip().startswith("#")}
    split_audio_files(a.audio_dir,
                      os.path.join(a.out_dir, "dev.list"),
                      os.path.join(a.out_dir, "train.list"),
                      a.path_prefix, skip, a.seed, a.exclude_first)
