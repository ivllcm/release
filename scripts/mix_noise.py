"""Mix each Lombard GRID utterance with an interferer at five fixed SNRs.

Two interference conditions, assigned to alternating utterances so that each
utterance appears under exactly one of them:

  multispeaker  another utterance drawn at random from the same list
  nonspeech     typical background noise, obtained as (noisy - clean) from a separately
                licensed corpus of domestic recordings. That corpus cannot be
                redistributed, so the `nonspeech` half of the output is absent
                from this release; the code for it is kept here unchanged.

Both conditions are produced in a single pass, and **both require the
background-noise corpus** -- not only `nonspeech`. The noise file drawn for an
utterance supplies the pairing id that appears in every output filename, and
the background track is computed on each iteration regardless of which branch
writes the mixture. There is no multispeaker-only mode.

The dev and test sets use the same clean utterances, but different noise samples 
are mixed with them, resulting in different noisy inputs while preserving the same
underlying target speech.

    python mix_noise.py --lists lists/train.list lists/dev.list lists/dev.list\
        --audio-root <corpus> --out-dir out/train out/dev out/test\
        --noise-noisy <noise>/noisy --noise-clean <noise>/clean

"""
import argparse
import os
import random
import shutil

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Background-noise source. Paths are relative to the corpus root; the noise is
# recovered as (noisy - clean) per file, so both directories are needed.
NOISE_NOISY_DIR = "../../CHiME2/dev/0dB/noisy"
NOISE_CLEAN_DIR = "../../CHiME2/dev/0dB/clean"

SNR_LEVELS = [-10, -5, 0, 5, 10]
SEED = 100

def load_audio(file_path, sr=16000):
    """Load audio file and resample to target sample rate"""
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def save_audio(audio, file_path, sr=16000):
    """Save audio to file as 16bit PCM WAV"""
    try:
        # Ensure audio is in range [-1, 1] before converting to int16
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(file_path, audio, sr, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False

def calculate_snr(signal, noise):
    """Calculate SNR between signal and noise"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def adjust_noise_level(signal, noise, target_snr_db):
    """Adjust noise level to achieve target SNR"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return noise
    
    # Calculate current SNR
    current_snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Calculate required noise scaling factor
    target_snr_linear = 10 ** (target_snr_db / 10)
    current_snr_linear = 10 ** (current_snr_db / 10)
    
    # Scale noise to achieve target SNR
    noise_scaling_factor = np.sqrt(current_snr_linear / target_snr_linear)
    adjusted_noise = noise * noise_scaling_factor
    
    return adjusted_noise

def mix_audio_with_noise(clean_audio, noise_audio, target_snr_db, noise_audio_clean=None):
    """Mix clean audio with noise at target SNR"""
    # Ensure both audios have the same length
    min_length = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_length]
    noise_audio = noise_audio[:min_length]
    if noise_audio_clean is not None:
        noise_audio_clean = noise_audio_clean[:min_length]
    
    # Adjust noise level to achieve target SNR
    adjusted_noise = adjust_noise_level(clean_audio, noise_audio, target_snr_db)
    
    # Mix audio
    mixed_audio = clean_audio + adjusted_noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val * 0.95
    
    return mixed_audio, adjusted_noise, noise_audio_clean


def get_noise_files(noise_dir=None, clean_dir=None):
    """Get list of noise files"""
    noise_dir = noise_dir or NOISE_NOISY_DIR
    clean_dir = clean_dir or NOISE_CLEAN_DIR
    if not os.path.exists(noise_dir):
        # Needed for both conditions, not just `nonspeech` -- see the note in
        # the module docstring.
        raise SystemExit(
            "Background-noise corpus not found at '{}'.\n"
            "Both interference conditions are produced in one pass and both "
            "need it: the drawn noise file supplies the pairing id that goes "
            "into every output filename, and the background track is computed "
            "on every iteration. Set NOISE_NOISY_DIR / NOISE_CLEAN_DIR at the "
            "top of this file.".format(noise_dir))
    
    noise_files = []
    for file in os.listdir(noise_dir):
        if file.endswith('.wav'):
            noise_files.append(os.path.join(noise_dir, file))
    clean_files = []
    for file in os.listdir(clean_dir):
        if file.endswith('.wav'):
            clean_files.append(os.path.join(clean_dir, file))
    
    print(f"Found {len(noise_files)} noise files")
    print(f"Found {len(clean_files)} clean files")
    return noise_files, clean_files


def process_audio_list(list_file, output_dir, noise_files, clean_files, snr_levels,
                       use_same_noise=False, audio_root="."):
    """Process audio files from a list file"""
    if not os.path.exists(list_file):
        print(f"Error: List file '{list_file}' not found!")
        return

    # Create output directories
    clean_dir = os.path.join(output_dir, "clean")
    noisy_dir = os.path.join(output_dir, "noisy")

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # Read audio file paths
    with open(list_file, 'r') as f:
        # List entries are relative to the corpus root; audio_root says where
        # that is, so the generator can be run from anywhere.
        audio_files = [os.path.normpath(os.path.join(audio_root, line.strip()))
                       for line in f if line.strip()]

    print(f"Processing {len(audio_files)} files from {list_file}")

    processed_count = 0
    clean_audio_num = 0

    for audio_file in tqdm(audio_files, desc=f"Processing {os.path.basename(list_file)}"):

        if not os.path.exists(audio_file):
            print(f"Warning: Audio file '{audio_file}' not found, skipping...")
            continue

        # Load clean audio
        clean_audio = load_audio(audio_file)
        if clean_audio is None:
            continue

        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_file))[0]

        # If use_same_noise is True, select one noise file for all SNR levels
        if use_same_noise:
            # Select one random noise file that is long enough for all SNR levels
            max_attempts = 10
            noise_file = None
            noise_audio = None
            noise_name = None

            for attempt in range(max_attempts):
                noise_file = random.choice(noise_files)
                noise_audio = load_audio(noise_file)
                noise_audio_clean = load_audio(noise_file.replace("noisy", "clean"))
                # noise_audio_clean will be selected from audio_files below
                if noise_audio is None:
                    continue

                # Check if noise is long enough
                if len(noise_audio) >= len(clean_audio):
                    noise_name = os.path.splitext(os.path.basename(noise_file))[0]
                    break

            # If no suitable noise file found, skip this audio file
            if noise_audio is None or len(noise_audio) < len(clean_audio):
                print(f"Warning: Could not find suitable noise file for {base_name}, skipping...")
                continue

        # For each SNR level, mix with noise
        for snr_db in tqdm(snr_levels, desc=f"SNR levels for {os.path.basename(audio_file)}", leave=False):
            # If not using same noise, select random noise file for each SNR level
            if not use_same_noise:
                # Select random noise file that is long enough
                max_attempts = 10  # Limit attempts to avoid infinite loop
                noise_file = None
                noise_audio = None
                noise_name = None

                for attempt in range(max_attempts):
                    noise_file = random.choice(noise_files)
                    noise_audio = load_audio(noise_file)
                    noise_audio_clean = load_audio(noise_file.replace("noisy", "clean"))

                    # noise_audio_clean will be selected from audio_files below
                    if noise_audio is None:
                        continue

                    # Check if noise is long enough
                    if len(noise_audio) >= len(clean_audio):
                        noise_name = os.path.splitext(os.path.basename(noise_file))[0]
                        break

                # If no suitable noise file found after max attempts, skip this SNR level
                if noise_audio is None or len(noise_audio) < len(clean_audio):
                    print(f"Warning: Could not find suitable noise file for {base_name} at {snr_db}dB, skipping...")
                    continue

            # Select noise_audio_clean from a random audio file in audio_files (excluding the current file)
            possible_clean_files = [f for f in audio_files if f != audio_file and os.path.exists(f)]
            if not possible_clean_files:
                print(f"Warning: No other clean files available for mixing for {base_name}, skipping...")
                continue
            random_clean_file = random.choice(possible_clean_files)
            noise_audio_other_speech = load_audio(random_clean_file)
            if noise_audio_other_speech is None:
                print(f"Warning: Could not load random clean file {random_clean_file} for mixing, skipping...")
                continue

            # random padding or cutting of noise_audio_other_speech
            rand_op = random.choice(['pad16000', 'cut16000', 'pad8000', 'cut8000', 'pad4000', 'cut4000'])
            if rand_op == 'pad16000':
                noise_audio_other_speech = np.pad(noise_audio_other_speech, (16000, 0), mode='constant')
            elif rand_op == 'cut16000':
                if len(noise_audio_other_speech) > 16000:
                    noise_audio_other_speech = noise_audio_other_speech[16000:]
                else:
                    noise_audio_other_speech = np.zeros_like(noise_audio_other_speech)
            elif rand_op == 'pad8000':
                noise_audio_other_speech = np.pad(noise_audio_other_speech, (8000, 0), mode='constant')
            elif rand_op == 'cut8000':
                if len(noise_audio_other_speech) > 8000:
                    noise_audio_other_speech = noise_audio_other_speech[8000:]
                else:
                    noise_audio_other_speech = np.zeros_like(noise_audio_other_speech)
            elif rand_op == 'pad4000':
                noise_audio_other_speech = np.pad(noise_audio_other_speech, (4000, 0), mode='constant')
            elif rand_op == 'cut4000':
                if len(noise_audio_other_speech) > 4000:
                    noise_audio_other_speech = noise_audio_other_speech[4000:]
                else:
                    noise_audio_other_speech = np.zeros_like(noise_audio_other_speech)

            # Make sure noise_audio_clean is at least as long as clean_audio
            if len(noise_audio_other_speech) < len(clean_audio):
                # Pad with zeros if too short
                pad_length = len(clean_audio) - len(noise_audio_other_speech)
                noise_audio_other_speech = np.pad(noise_audio_other_speech, (0, pad_length), mode='constant')
            else:
                noise_audio_other_speech = noise_audio_other_speech[:len(clean_audio)]

            # Mix audio with noise
            noise_audio_bg = noise_audio - noise_audio_clean

            if clean_audio_num % 2 == 0:
                mixed_audio_nonspeechnoise, adjusted_noise, noise_audio_clean_cliped = mix_audio_with_noise(
                    clean_audio, noise_audio_bg, snr_db, noise_audio_clean=noise_audio_clean)
            else:
                mixed_audio_speechnoise, adjusted_noise, _ = mix_audio_with_noise(
                    clean_audio, noise_audio_other_speech, snr_db)

            # Create output filenames with 'm' for negative SNR
            snr_str = f"m{abs(snr_db)}" if snr_db < 0 else str(snr_db)

            if clean_audio_num % 2 == 0:
                output_filename_nonspeechnoise = f"{base_name}_{snr_str}dB_{noise_name}_nonspeech.wav"
                clean_output_path_nonspeech = os.path.join(clean_dir, output_filename_nonspeechnoise)
                noisy_output_path_nonspeech = os.path.join(noisy_dir, output_filename_nonspeechnoise)
            else:
                output_filename_speechnoise = f"{base_name}_{snr_str}dB_{noise_name}_multispeaker.wav"
                clean_output_path_speech = os.path.join(clean_dir, output_filename_speechnoise)
                noisy_output_path_speech = os.path.join(noisy_dir, output_filename_speechnoise)

            output_filename_noisevoice = f"{base_name}_{snr_str}dB_{noise_name}_VN.wav"
            noisy_output_path_noisevoice = os.path.join(noisy_dir, output_filename_noisevoice)
            save_audio(adjusted_noise, noisy_output_path_noisevoice)

            # Copy original clean audio file directly, but ensure 16bit PCM format
            # Load and save as 16bit PCM
            clean_audio_for_save = load_audio(audio_file)

            if clean_audio_for_save is not None:
                if clean_audio_num % 2 == 0:
                    save_audio(clean_audio_for_save, clean_output_path_nonspeech)
                else:
                    save_audio(clean_audio_for_save, clean_output_path_speech)
            else:
                # If failed to load, fallback to copy (may not guarantee 16bit PCM)
                if clean_audio_num % 2 == 0:
                    shutil.copy2(audio_file, clean_output_path_nonspeech)
                else:
                    shutil.copy2(audio_file, clean_output_path_speech)
                print(f"Warning: Could not load {audio_file}, copied instead")

            # Save mixed audio as 16bit PCM
            if clean_audio_num % 2 == 0:
                save_audio(mixed_audio_nonspeechnoise, noisy_output_path_nonspeech)
            else:
                save_audio(mixed_audio_speechnoise, noisy_output_path_speech)

            processed_count += 1

        clean_audio_num += 1
    print(f"Completed processing {list_file}: {processed_count} files generated")



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lists", nargs="+", default=["train.list", "dev.list"],
                    help="utterance lists produced by split.py, training first")
    ap.add_argument("--audio-root", default=".",
                    help="corpus root the list entries are relative to")
    ap.add_argument("--out-dir", nargs="+", required=True,
                    help="one output directory per list; each receives clean/ and noisy/")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="the draws are seeded with this")
    ap.add_argument("--noise-noisy", default=None, help="overrides NOISE_NOISY_DIR")
    ap.add_argument("--noise-clean", default=None, help="overrides NOISE_CLEAN_DIR")
    args = ap.parse_args()

    if len(args.out_dir) != len(args.lists):
        raise SystemExit("--out-dir needs one directory per entry in --lists")

    random.seed(args.seed)
    noise_files, clean_files = get_noise_files(args.noise_noisy, args.noise_clean)

    for i, lst in enumerate(args.lists):
        if not os.path.exists(lst):
            print("Warning: {} not found".format(lst))
            continue
        stem = os.path.splitext(os.path.basename(lst))[0]
        out = args.out_dir[i]
        # The evaluation list reuses one noise file across an utterance's five
        # SNRs; the training list draws a fresh one at every level.
        same = (stem != "train")
        print("\n=== Processing {} -> {} ===".format(lst, out))
        process_audio_list(lst, out, noise_files, clean_files, SNR_LEVELS,
                           use_same_noise=same, audio_root=args.audio_root)

    print("\nNoise mixing completed!")


if __name__ == "__main__":
    main()
