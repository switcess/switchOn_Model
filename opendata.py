import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

NUM_FRAMES = 180

def preprocess_video(frames):
    processed_frames = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (64, 64))
        processed_frames.append(frame_resized)
    return np.stack(processed_frames, axis=0)  # Stack along depth axis for 3D CNN input

def pad_or_truncate_video(video, target_length):
    """Pads or truncates a video to match the target number of frames."""
    num_frames = len(video)
    
    if num_frames > target_length:
        return np.array(video[:target_length])
    elif num_frames < target_length:
        padding = np.zeros((target_length - num_frames, 64, 64, 3), dtype=np.uint8)  # Padding with zeros
        return np.concatenate((video, padding), axis=0)
    else:
        return np.array(video)

def parse_xml_for_labels(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    start_frames = []
    end_frames = []

    # Process tracks to find start and end frames
    for track in root.findall('track'):
        label = track.get('label')
        if label == 'theft_start':
            start_frames.extend([int(box.get('frame')) for box in track.findall('box')])
        elif label == 'theft_end':
            end_frames.extend([int(box.get('frame')) for box in track.findall('box')])

    return start_frames, end_frames

def load_data(data_dir):
    videos = []
    labels = []
    
    xml_path = os.path.join(data_dir, 'datalabel/shoplifting')  # XML 파일 경로
    video_path = os.path.join(data_dir, 'sourcedata/shoplifting')  # 비디오 파일 경로

    # Iterate through XML files in the directory
    for filename in os.listdir(xml_path):
        if filename.endswith('.xml'):
            # Parse XML
            xml_file = os.path.join(xml_path, filename)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Extract metadata
            video_name = root.find('meta/source').text
            source = os.path.join(video_path, video_name)
            theft_start, theft_end = parse_xml_for_labels(xml_file)
            
            # Load video
            if os.path.exists(source):
                cap = cv2.VideoCapture(source)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Initialize video and frame labels
                video = []
                frame_labels = np.zeros(total_frames, dtype=int)

                for frame_idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize the frame to a consistent size (e.g., 64x64)
                    frame_resized = cv2.resize(frame, (64, 64))
                    video.append(frame_resized)
                    
                    # Label frames between theft_start and theft_end as 1
                    if any(start <= frame_idx <= end for start in theft_start for end in theft_end):
                        frame_labels[frame_idx] = 1

                cap.release()

                if video:
                    # Pad or truncate video to the fixed number of frames
                    video = pad_or_truncate_video(video, NUM_FRAMES)
                    frame_labels = frame_labels[:NUM_FRAMES]  # Truncate labels to match video length
                    
                    # Add the video and labels after processing the entire video
                    videos.append(video)  # [num_frames, height, width, channels]
                    labels.append(frame_labels)  # [num_frames]
    
    # Convert to numpy arrays
    videos = np.array(videos)  # Shape: [num_videos, num_frames, height, width, channels]
    labels = np.array(labels)  # Shape: [num_videos, num_frames]
    
    return videos, labels
