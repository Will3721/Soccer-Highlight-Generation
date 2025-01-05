import numpy as np
import av
import torch
import pickle
from transformers import VideoMAEImageProcessor
from transformers import TimesformerForVideoClassification
from google.colab import drive
import gc
import json
import os
drive.mount('/content/drive')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fps = 25
seconds_per_clip = 5
overlap = 2.5
frames_per_clip = int(fps * seconds_per_clip)
overlap_frames = int(fps * overlap)

root_directory = "/content/drive/Shareddrives/ESE_546_Final_Project/SoccerNet_Files"
output_directory = "/content/drive/Shareddrives/ESE_546_Final_Project/Processed_Tokens_Final"

os.makedirs(output_directory, exist_ok=True)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def process_segment(video, clips, image_processor):
  processed = []
  with torch.no_grad():
    for i, (start, end) in enumerate(clips):
        print(f"Processing clip from frame {start} to {end}")
        inputs = image_processor(video[start:end], return_tensors="pt")
        processed.append(inputs['pixel_values'])
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
  return processed

def predict_tokens(model, clips):
  tokens = []
  with torch.no_grad():
    for clip in clips:
      outputs = model(clip.to(device))
      logits = outputs.logits
      tokens.append(torch.argmax(outputs.logits).item())
  return tokens

def save_tokens(tokens, filename):
  with open(filename, 'w') as f:
    json.dump(tokens, f)

def process_video(video_path, output_path):
  try:
    for starting_frame_to_process, ending_frame_to_process, half in [(0, 33750, 1), (33750, 67500, 2)]:
      container = av.open(video_path)
      image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
      model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600").to(device)
      video = list(read_video_pyav(container, list(range(starting_frame_to_process, ending_frame_to_process))))

      total_frames = ending_frame_to_process - starting_frame_to_process
      clips = []
      start_frame = 0

      while start_frame + frames_per_clip <= total_frames:
          end_frame = start_frame + frames_per_clip
          if (end_frame - start_frame) % 4 != 0:
              end_frame = start_frame + ((end_frame - start_frame) // 4) * 4

          clips.append((start_frame, end_frame))
          start_frame = max(0, end_frame - overlap_frames)

      processed_segment = process_segment(video, clips, image_processor)
      predicted = predict_tokens(model, processed_segment)

      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      output_path = output_path.replace('.json', f'_{half}.json')
      with open(output_path, 'w') as f:
          json.dump({'tokens': predicted}, f)
      print(f"Processed {video_path} and saved the results to {output_path}.")

      # Clean up to free memory
      del container, video, processed_segment, predicted, image_processor, model, clips
      torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

  except Exception as e:
    print(f"Error processing {video_path}: {e}")

def get_files_to_process(root_dir, output_dir):
    files_to_process = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get relative path to maintain directory structure
        relative_path = os.path.relpath(dirpath, root_dir)
        output_subdir = os.path.join(output_dir, relative_path)

        # Count files in the corresponding output directory
        if os.path.exists(output_subdir):
            num_files_in_output = len([f for f in os.listdir(output_subdir) if os.path.isfile(os.path.join(output_subdir, f))])
        else:
            num_files_in_output = 0

        # If the output directory does not have exactly 4 files, add files from root_directory for processing
        if num_files_in_output != 4:
            for filename in filenames:
                files_to_process.append(os.path.join(dirpath, filename))

    return files_to_process

files_to_process = get_files_to_process(root_directory, output_directory)

# Process only the filtered files
for file_path in files_to_process:
    # Compute relative path and corresponding output path
    relative_path = os.path.relpath(file_path, root_directory)
    output_path = os.path.join(output_directory, relative_path + ".json")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process the video
    print(f"Processing: {file_path}")
    process_video(file_path, output_path)