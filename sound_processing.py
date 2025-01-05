import os
import subprocess
from pydub import AudioSegment
import pandas as pd

def process_video_sound(video_path, output_dir, output_csv_name):
    loudness_data = []
    audio_path = "/tmp/temp_audio.wav"

    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        audio = AudioSegment.from_file(audio_path, format="wav")
        duration_ms = len(audio)

        # Break audio into 5-second chunks with 2.5-second overlap
        segment_duration_ms = 5 * 1000
        step_size_ms = int(segment_duration_ms / 2)

        for start_time_ms in range(0, duration_ms, step_size_ms):
            end_time_ms = min(start_time_ms + segment_duration_ms, duration_ms)
            segment = audio[start_time_ms:end_time_ms]
            loudness = segment.dBFS

            loudness_data.append({
                "start_time": start_time_ms / 1000,
                "end_time": end_time_ms / 1000,
                "loudness": loudness
            })

    except Exception as e:
        print(f"Error processing audio for {video_path}: {e}")
        return

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    if loudness_data:
        loudness_df = pd.DataFrame(loudness_data)
        loudness_csv_path = os.path.join(output_dir, output_csv_name)
        loudness_df.to_csv(loudness_csv_path, index=False)
        print(f"Loudness data saved: {loudness_csv_path}")

base_dir = "/content/drive/Shareddrives/ESE_546_Final_Project/SoccerNet_Files/england_epl"
audio_output_dir = '/content/drive/Shareddrives/ESE_546_Final_Project/audio_segments'
os.makedirs(audio_output_dir, exist_ok=True)

for dirpath, dirnames, filenames in os.walk(base_dir):
    for file in filenames:
        print(file)
        if file in ['1_224p.mkv', '2_224p.mkv']:
            print('hello')
            video_path = os.path.join(dirpath, file)
            game_folder = os.path.basename(dirpath)
            output_dir = os.path.join(audio_output_dir, game_folder)
            os.makedirs(output_dir, exist_ok=True)

            if file == '1_224p.mkv':
                output_csv_name = "loudnessi.csv"
            elif file == '2_224p.mkv':
                output_csv_name = "loudnessii.csv"

            process_video_sound(video_path, output_dir, output_csv_name)