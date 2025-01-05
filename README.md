# Long Short-Term Memory Model-Enabled Highlight Detection in Soccer

## Overview

This study describes a Long Short-Term Memory-enabled approach to accomplish the task of detecting highlights
in soccer games. A deep neural network pre-trained on the Kinetics 600 dataset was used to generate token "predictions" for
each five-second segment of these game videos, which were downloaded from [SoccerNet](https://www.soccer-net.org/data). An LSTM model was subsequently trained on these tokenized predictions
along with audio data to establish sequential relationships for predicting the presence of highlights. Our work provides a promising approach to automated highlight detection and recap generation.

## Repository Structure

-   Report.pdf - a comprehensive report of our method and findings
-   data_download.py - code that was used to download the data onto our local machines and drive
-   process_and_predict.py - code that uses a TimeSformer model pretrained on the Kinetics-600 dataset and a VideoMAE model also pretrained on the Kinetics dataset to classify each 5-second segment with a token
-   sound_processing.py - code that processes the audio data and associates each segment with a loudness value in decibels
-   highlight_generator.py - contains the bulk of the code used to train the LSTM and generate video clips with the highlights

## Team Members

William Qi, Kyle Wu, Emmet Young
