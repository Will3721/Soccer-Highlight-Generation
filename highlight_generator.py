import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import os
import json
import csv
import numpy as np

def process_game_data(base_dir):
    """
    Processes all JSON and CSV files in directories containing soccer game data.

    Args:
        base_dir (str): Base directory containing subdirectories with JSON and CSV files.

    Returns:
        dict: A dictionary where keys are game directories and values are lists of tuples.
              Each tuple contains a token and its corresponding normalized loudness.
    """
    def z_score_normalize(values):
        mean = np.mean(values)
        std = np.std(values)
        return [(x - mean) / std for x in values]

    game_data = {}  # Store processed data per game

    # Traverse the base directory
    for root, dirs, files in os.walk(base_dir):
        json_files = sorted([f for f in files if f.endswith('.json')])
        csv_files = sorted([f for f in files if f.endswith('.csv')])

        if len(json_files) != 4 or len(csv_files) != 2:
            print(f"Skipping {root}: Missing files")
            continue

        tokens = []
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            try:
                with open(json_path, 'r') as file:
                    data = json.load(file)
                    tokens.extend(data.get("tokens", []))
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

        loudness_values = []
        segments = []
        for csv_file in csv_files:
            csv_path = os.path.join(root, csv_file)
            try:
                with open(csv_path, mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        start_time = float(row['start_time'])
                        loudness = float(row['loudness'])
                        end_time = start_time + 5.0  # Calculate end time for segments
                        segments.append((start_time, end_time))
                        loudness_values.append(loudness)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        # Normalize loudness values
        normalized_loudness = z_score_normalize(loudness_values)

        # Ensure tokens and loudness align correctly
        if len(tokens) != len(normalized_loudness):
            print(f"Warning: Token count ({len(tokens)}) does not match loudness count ({len(normalized_loudness)}) in {root}")

        # Combine tokens and loudness into a sequential list
        combined_data = list(zip(tokens, normalized_loudness))

        # Store the combined data for this game
        game_data[root] = combined_data

    return game_data

# Example usage
base_dir = '/content/drive/Shareddrives/ESE_546_Final_Project/Processed_Tokens_Final/'
game_data = process_game_data(base_dir)

# Print a summary of the processed data
for game, data in game_data.items():
    print(f"{game}: {len(data)} token-loudness pairs processed.")
    print(f"First 5 pairs: {data[:5]}")

x = [y[:2160] for y in list(game_data.values()) if len(y) > 2000]
data = torch.tensor(x)

train_ratio = 0.8
sequence_length = 10
num_train_games = int(len(data) * train_ratio)

train_games = data[:num_train_games]
test_games = data[num_train_games:]
train_games.shape, test_games.shape

def generate_sequences(games, sequence_length):
    inputs, targets, change_rates = [], [], []
    for game_tokens in games:
        for i in range(len(game_tokens) - sequence_length):
            sequence = game_tokens[i:i + sequence_length]
            inputs.append(game_tokens[i:i + sequence_length])
            targets.append(game_tokens[i + sequence_length])
            change_rate = sum(abs(sequence[j] - sequence[j-1]) for j in range(1, sequence_length))
            change_rates.append(change_rate)
    return torch.stack(inputs), torch.tensor(targets), torch.tensor(change_rates)

# Training data
train_token_inputs, train_token_targets, train_token_change_rates = generate_sequences(train_games[:, :, 0], sequence_length)
train_audio_inputs, train_audio_targets, train_audio_change_rates = generate_sequences(train_games[:, :, 1], sequence_length)

# Testing data
test_token_inputs, test_token_targets, test_token_change_rates = generate_sequences(test_games[:, :, 0], sequence_length)
test_audio_inputs, test_audio_targets, test_audio_change_rates = generate_sequences(test_games[:, :, 1], sequence_length)

print(f"Train Inputs: {train_token_inputs.shape}, Train Targets: {train_token_targets.shape}")
print(f"Test Inputs: {test_token_inputs.shape}, Test Targets: {test_token_targets.shape}")
print(f"Train Change Rates: {train_token_change_rates.shape}, Test Change Rates: {test_token_change_rates.shape}")

def temporal_token_changes(tokens, window_size=5):
  changes = []
  for i in range(len(tokens) - window_size):
    curr_window = np.mean(tokens[i:1 + window_size], axis=0)
    next_window = np.mean(tokens[i + 1: i + 1 + window_size], axis=0)
    changes.append(next_window - curr_window)
  return np.array(changes)

print('Train token input sequence: ', train_token_inputs[0])
print('Train token target: ', train_token_targets[0])
print('Train audio input sequence: ', train_audio_inputs[0])
print('Train audio target: ', train_audio_targets[0])

train_token_inputs = train_token_inputs.long()
train_token_targets = train_token_targets.long()
train_audio_inputs = train_audio_inputs.float()
train_audio_targets = train_audio_targets.float()
test_token_inputs = test_token_inputs.long()
test_token_targets = test_token_targets.long()
test_audio_inputs = test_audio_inputs.float()
test_audio_targets = test_audio_targets.float()

class LSTM_one_hot(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers):
        super(LSTM_one_hot, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_size, num_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, loud):
        x = self.embedding(x)
        loud = loud.unsqueeze(2)
        x = torch.cat((x, loud), dim=2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def one_hot_encode(indices, vocab_size):
  batch_size, seq_length = indices.size()
  one_hot = torch.zeros(batch_size, seq_length, vocab_size)
  one_hot.scatter_(2, indices.unsqueeze(2), 1)
  return one_hot

vocab_size = 600
embedding_dim = 128
hidden_size = 256
output_size = vocab_size
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_labels = torch.cat((train_token_targets, test_token_targets), dim=0).cpu().numpy()
unique_labels = np.unique(all_labels)

class_weights = compute_class_weight(None, classes=unique_labels, y=all_labels)

class_weights_tensor = torch.zeros(600, dtype=torch.float).to(device)
class_weights_tensor[unique_labels] = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

model = LSTM_one_hot(vocab_size, embedding_dim, hidden_size, output_size, num_layers).to(device)

train_dataset = TensorDataset(train_token_inputs, train_token_targets, train_token_change_rates, train_audio_inputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_token_inputs, test_token_targets, test_token_change_rates, test_audio_inputs)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import matplotlib.pyplot as plt

epochs = 70
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

alpha = 0.5
beta = 1.0

training_losses = []
accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_inputs, batch_targets, batch_change_rates, batch_audio_inputs in train_loader:
        batch_inputs, batch_targets, batch_change_rates, batch_audio_inputs = batch_inputs.to(device), batch_targets.to(device), batch_change_rates.to(device), batch_audio_inputs.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs, batch_audio_inputs)
        individual_losses = nn.CrossEntropyLoss(reduction='none')(outputs, batch_targets)

        # Dynamic weights based on losses
        batch_weights = 1 + alpha * torch.pow(individual_losses.detach(), beta)
        batch_weights /= batch_weights.mean()

        # Compute weighted loss
        weighted_loss = (individual_losses * batch_weights).mean()
        weighted_loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += weighted_loss.item()

        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == batch_targets).sum().item()
        total_predictions += batch_targets.size(0)

    # Compute average loss and accuracy for the epoch
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    scheduler.step()

    # Append losses and accuracy for plotting
    training_losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), training_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid(True)
plt.show()

import copy
def generate_video(base_file_path, half_num, output_filename):
  def z_score_normalize(values):
        mean = np.mean(values)
        std = np.std(values)
        return [(x - mean) / std for x in values]

  def calculate_surprise(logits, actual_tokens):
    """
    Calculate surprise for each token based on the negative log-probability
    of the actual token under the predicted logits.

    Args:
        logits (np.ndarray): Array of shape (T, C), where T is the number of segments
                             and C is the number of classes (softmax probabilities).
        actual_tokens (np.ndarray): Array of shape (T,) with the true token indices.

    Returns:
        np.ndarray: Surprise scores of shape (T,).
    """
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax probabilities
    surprise = -np.log(probs[np.arange(len(actual_tokens)), actual_tokens])
    return surprise

  def calculate_significance(logits, class_weights):
    """
    Calculate significance for each token based on weighted probabilities
    from the predicted logits.

    Args:
        logits (np.ndarray): Array of shape (T, C), where T is the number of segments
                            and C is the number of classes (softmax probabilities).
        class_weights (dict): Dictionary mapping class indices to significance weights.

    Returns:
        np.ndarray: Significance scores of shape (T,).
    """
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax probabilities
    significance = np.sum(probs * np.array([class_weights.get(i, 0) for i in range(probs.shape[1])]), axis=1)
    return significance

  tokens = []
  file_path_1 = f'{base_file_path}/{half_num}_224p.mkv_1.json'
  with open(file_path_1, 'r') as json_file:
    data = json.load(json_file)
    tokens = data.get("tokens", [])

  file_path_2 = f'{base_file_path}/{half_num}_224p.mkv_1_2.json'
  with open(file_path_2, 'r') as json_file:
    data = json.load(json_file)
    tokens_2 = data.get("tokens", [])
  tokens.extend(tokens_2)

  segments_1 = []
  loudness_values_1 = []
  t = 'i' if half_num == 1 else 'ii'
  csv_path_1 = f'{base_file_path}/Copy of loudness{t}.csv'
  with open(csv_path_1, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        start_time = float(row['start_time'])
        loudness = float(row['loudness'])
        end_time = start_time + 5.0  # Calculate end time for segments
        segments_1.append((start_time, end_time))
        loudness_values_1.append(loudness)

  normalized_loudness = z_score_normalize(loudness_values_1)
  eval_data = np.array(list(zip(tokens, normalized_loudness)))

  sequence_length = 10
  eval_token_inputs, eval_audio_inputs, eval_targets = [], [], []
  for i in range(len(eval_data) - sequence_length):
    eval_token_inputs.append(eval_data[:, 0][i:i+sequence_length])
    eval_audio_inputs.append(eval_data[:, 1][i:i+sequence_length])
    eval_targets.append(eval_data[:, 0][i + sequence_length])

  token_inputs = torch.tensor(eval_token_inputs).to(device).long()
  audio_inputs = torch.tensor(eval_audio_inputs).to(device).float()
  token_targets = torch.tensor(eval_targets).to(device).long()
  predicted_logits = model(token_inputs, audio_inputs)
  predicted_logits = predicted_logits.cpu().detach().numpy()

  probs = np.exp(predicted_logits) / np.sum(np.exp(predicted_logits), axis=1, keepdims=True)
  true_tokens = copy.deepcopy(eval_data[10:, 0])
  true_tokens = [int(x) for x in true_tokens]
  predicted_tokens = np.argmax(probs, axis=1)

  class_frequencies = np.bincount(true_tokens, minlength=600)
  class_weights = {i: 1 / (class_frequencies[i] + 1e-6) for i in range(600)}

  def normalize(data):
    """Min-Max normalization."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

  def get_highlights(significance, surprise, w1=0.5, w2=0.5, threshold=0.7, window_size=3):
    """
    Identify highlight segments based on significance and surprise scores.

    Parameters:
        significance (numpy array): Significance scores for each segment.
        surprise (numpy array): Surprise scores for each segment.
        w1 (float): Weight for significance.
        w2 (float): Weight for surprise.
        threshold (float): Threshold for combined score to classify as a highlight.
        window_size (int): Window size for smoothing scores.

    Returns:
        List of tuples: Highlight windows as (start_index, end_index).
    """
    # Normalize scores
    norm_significance = normalize(significance)
    norm_surprise = normalize(surprise)

    # Compute combined score
    combined_score = w1 * norm_significance + w2 * norm_surprise

    # Smooth combined score
    smoothed_score = np.convolve(combined_score, np.ones(window_size)/window_size, mode='same')

    # Identify segments exceeding threshold
    highlight_indices = np.where(smoothed_score > threshold)[0]

    # Merge consecutive indices into highlight windows
    highlights = []
    start = highlight_indices[0]
    for i in range(1, len(highlight_indices)):
        if highlight_indices[i] != highlight_indices[i - 1] + 1:
            highlights.append((start, highlight_indices[i - 1]))
            start = highlight_indices[i]
    highlights.append((start, highlight_indices[-1]))

    return highlights

  # Example usage:
  significance_scores = calculate_significance(predicted_logits, class_weights)
  surprise_scores = calculate_surprise(predicted_logits, true_tokens)
  significance_weight = 0.5
  surprise_weight = 0.5
  threshold = 0.11
  window_size = 10

  highlights = get_highlights(significance_scores, surprise_scores, w1=significance_weight, w2=surprise_weight, threshold=threshold, window_size=window_size)
  print("Highlights:", highlights)
  if len(highlights) > 20:
    return

  context_size = 15 # in seconds
  intervals = []
  for highlight in highlights:
    start_token, end_token = highlight
    start_second = (start_token + 10) * 2.5 - context_size
    end_second = (end_token + 10) * 2.5 + context_size
    intervals.append((start_second, end_second))

  def merge_intervals(intervals):
    """
    Merge all overlapping intervals.

    Parameters:
        intervals (list of tuples): List of intervals as (start, end).

    Returns:
        list of tuples: List of merged intervals.
    """
    if not intervals:
        return []

    # Sort intervals by start point (and end point if starts are equal)
    intervals.sort(key=lambda x: (x[0], x[1]))

    # Initialize merged intervals with the first interval
    merged = [intervals[0]]

    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        current_start, current_end = current

        if current_start <= prev_end:  # Overlap condition
            # Merge intervals by updating the end of the last merged interval
            merged[-1] = (prev_start, max(prev_end, current_end))
        else:
            # No overlap, add the current interval
            merged.append(current)

    return merged

  merged_intervals = merge_intervals(intervals)

  from moviepy.editor import VideoFileClip, concatenate_videoclips
  video_file_path = f'{base_file_path}/{half_num}_224p.mkv'
  video_file_path = video_file_path.replace('Processed_Tokens_Final', 'SoccerNet_Files')
  video = VideoFileClip(video_file_path)
  all_clips = []
  for start, end in merged_intervals:
    clip = video.subclip(start, end)
    all_clips.append(clip)

  highlights = concatenate_videoclips(all_clips, method='compose')
  highlights.write_videofile(output_filename, codec="libx264", audio_codec='aac')

generate_video(base_file_path='/content/drive/Shareddrives/ESE_546_Final_Project/Processed_Tokens_Final/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool', half_num=2, output_filename='example_highlights_1.mp4')

generate_video(base_file_path='/content/drive/Shareddrives/ESE_546_Final_Project/Processed_Tokens_Final/england_epl/2016-2017/2016-12-10 - 20-30 Leicester 4 - 2 Manchester City', half_num=2, output_filename='example_highlights_2.mp4')

generate_video(base_file_path='/content/drive/Shareddrives/ESE_546_Final_Project/Processed_Tokens_Final/england_epl/2016-2017/2017-01-02 - 18-00 Sunderland 2 - 2 Liverpool', half_num=2, output_filename='example_highlights_3.mp4')