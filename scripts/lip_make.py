"""Crop the frontal videos down to the target talker's lip region.

A fixed 180x180 window at the center of each 720x480 frame, nudged per talker
by a hand-tuned offset, resized to 64x64 and written at 25 fps. No face tracker
is involved, so the crop stays put for the whole utterance -- the talkers are
seated and filmed head-on.

    python lip_make.py --front-dir <corpus>/front --out-dir lip_out
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re

def crop_video_roi(input_path, output_path, roi_x, roi_y, roi_width, roi_height, output_width, output_height):
    """
    Crop a video to extract ROI and save as MP4 with resizing
    
    Args:
        input_path: Path to input .mov file
        output_path: Path to output .mp4 file
        roi_x, roi_y: Top-left corner of ROI
        roi_width, roi_height: Size of ROI to crop
        output_width, output_height: Size to resize the cropped ROI
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set output frame rate to 25fps
        output_fps = 25
        
        # Create video writer with output size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            cap.release()
            return False
        
        # Process frames
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame to ROI
            cropped_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            
            # Resize cropped frame to output size
            resized_frame = cv2.resize(cropped_frame, (output_width, output_height))
            
            # Write resized frame
            out.write(resized_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def get_speaker_offsets():
    """
    Define center offsets for each speaker (s2 to s55)
    Returns a dictionary with speaker offsets: {speaker: (offset_x, offset_y)}
    """
    # Default offsets - you can modify these for each speaker
    # Format: 's{number}': (offset_x, offset_y)
    # Positive offset_x moves ROI right, positive offset_y moves ROI down
    speaker_offsets = {}
    
    # Initialize all speakers with default offset (0, 0)
    for i in range(2, 56):  # s2 to s55
        speaker_offsets[f's{i}'] = (0, 0)
    
    # Modify specific speakers if needed
    speaker_offsets['s2'] = (-4, 50)   
    speaker_offsets['s3'] = (30, 0)   
    speaker_offsets['s4'] = (0, 60)   
    speaker_offsets['s5'] = (33, 79)   
    speaker_offsets['s6'] = (24, 40)   
    speaker_offsets['s7'] = (-5, 90)   
    speaker_offsets['s8'] = (-77, 88)   
    speaker_offsets['s9'] = (-33, 50)   
    speaker_offsets['s10'] = (62, 88)   
    speaker_offsets['s11'] = (23, 92)   
    speaker_offsets['s12'] = (15, 93)   
    speaker_offsets['s13'] = (45, 83)   
    speaker_offsets['s14'] = (10, 90)   
    speaker_offsets['s15'] = (10, 50)   
    speaker_offsets['s16'] = (25, 65)   
    speaker_offsets['s17'] = (15, 105)   
    speaker_offsets['s18'] = (40, 80)   
    speaker_offsets['s19'] = (25, 70)   
    speaker_offsets['s20'] = (-14, 85)   
    speaker_offsets['s21'] = (20, 85)   
    speaker_offsets['s22'] = (15, 75)   
    speaker_offsets['s23'] = (15, 33)   
    speaker_offsets['s24'] = (25, 95)   
    speaker_offsets['s25'] = (10, 120)   
    speaker_offsets['s26'] = (30, 124)   
    speaker_offsets['s27'] = (10, 95)   
    speaker_offsets['s28'] = (70, 100)   
    speaker_offsets['s29'] = (80, 67)   
    speaker_offsets['s30'] = (15, 60)   
    speaker_offsets['s31'] = (10, 12)   
    speaker_offsets['s32'] = (0, 60)   
    speaker_offsets['s33'] = (30, 85)   
    speaker_offsets['s34'] = (25, 85)   
    speaker_offsets['s35'] = (-20, 75)   
    speaker_offsets['s36'] = (0, 95)   
    speaker_offsets['s37'] = (0, 105)   
    speaker_offsets['s38'] = (-10, 70)   
    speaker_offsets['s39'] = (15, 90)   
    speaker_offsets['s40'] = (0, 75)   
    speaker_offsets['s41'] = (-10, 35)   
    speaker_offsets['s42'] = (-10, 70)   
    speaker_offsets['s43'] = (-5, 135)   
    speaker_offsets['s44'] = (40, 55)   
    speaker_offsets['s45'] = (25, 85)   
    speaker_offsets['s46'] = (30, 80)   
    speaker_offsets['s47'] = (0, 70)   
    speaker_offsets['s48'] = (5, 80)   
    speaker_offsets['s49'] = (10, 120)   
    speaker_offsets['s50'] = (-15, 75)   
    speaker_offsets['s51'] = (-15, 100)   
    speaker_offsets['s52'] = (-15, 110)   
    speaker_offsets['s53'] = (9, 115)   
    speaker_offsets['s54'] = (-5, 95)   
    speaker_offsets['s55'] = (45, 90)   
    return speaker_offsets

def extract_speaker_from_filename(filename):
    """
    Extract speaker ID from filename (e.g., 's2_l_001.mov' -> 's2')
    """
    match = re.match(r'^(s\d+)', filename)
    if match:
        return match.group(1)
    return None

def process_front_videos(front_dir="./front", output_dir="./front_lip_64by64", limit=0):
    """
    Process all .mov files in the front folder, crop to lip ROI and save as .mp4
    """
    
    # ROI parameters (center of 720x480 video)
    video_width = 720
    video_height = 480
    roi_width = 180
    roi_height = 180
    
    # Output size (resized from ROI)
    output_width = 64
    output_height = 64
    
    # Get speaker-specific offsets
    speaker_offsets = get_speaker_offsets()
    
    # Base ROI position (center of video)
    base_roi_x = (video_width - roi_width) // 2
    base_roi_y = (video_height - roi_height) // 2
    
    print(f"Base ROI: x={base_roi_x}, y={base_roi_y}, width={roi_width}, height={roi_height}")
    print(f"Output size: {output_width}x{output_height}")
    print(f"Speaker offsets loaded: {len(speaker_offsets)} speakers")
    
    # Check if front directory exists
    if not os.path.exists(front_dir):
        print(f"Error: Front directory '{front_dir}' not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .mov files
    mov_files = []
    for file in os.listdir(front_dir):
        if file.endswith('.mov'):
            mov_files.append(file)
    
    mov_files.sort()
    if limit:
        mov_files = mov_files[:limit]
    print(f"Found {len(mov_files)} .mov files in {front_dir}")
    
    if len(mov_files) == 0:
        print("No .mov files found!")
        return
    
    #rocess all video files
    successful_count = 0
    failed_count = 0
    
    for mov_file in tqdm(mov_files, desc="Processing videos"):
        input_path = os.path.join(front_dir, mov_file)
        
        # Extract speaker from filename
        speaker = extract_speaker_from_filename(mov_file)
        if speaker is None:
            print(f"Warning: Could not extract speaker from {mov_file}, using default offset")
            speaker = 's2'  # Default fallback
        
        # Get speaker-specific offset
        offset_x, offset_y = speaker_offsets.get(speaker, (0, 0))
        
        # Calculate adjusted ROI position
        roi_x = base_roi_x + offset_x
        roi_y = base_roi_y + offset_y
        
        # Ensure ROI stays within video bounds
        roi_x = max(0, min(roi_x, video_width - roi_width))
        roi_y = max(0, min(roi_y, video_height - roi_height))
        
        # Create output filename (change extension from .mov to .mp4)
        output_filename = os.path.splitext(mov_file)[0] + '.mp4'
        output_path = os.path.join(output_dir, output_filename)
        
        # Crop video
        if crop_video_roi(input_path, output_path, roi_x, roi_y, roi_width, roi_height, output_width, output_height):
            successful_count += 1
        else:
            failed_count += 1
    
    print(f"\nProcessing completed!")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--front-dir", required=True, help="the corpus frontal videos")
    ap.add_argument("--out-dir", required=True, help="where the crops go")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N videos")
    a = ap.parse_args()
    process_front_videos(a.front_dir, a.out_dir, a.limit)
