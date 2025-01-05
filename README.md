# Long Short-Term Memory Model-Enabled Highlight Detection in Soccer

## Overview

This study describes a Long Short-Term Memory-enabled approach to accomplish the task of detecting highlights
in soccer games. A deep neural network pre-trained on the Kinetics 600 dataset was used to generate token "predictions" for
each five-second segment of these game videos, which were downloaded from [SoccerNet](https://www.soccer-net.org/data). An LSTM model was subsequently trained on these tokenized predictions
along with audio data to establish sequential relationships for predicting the presence of highlights. Our work provides a promising approach to automated highlight detection and recap generation.

## Repository Structure

-   Report.pdf - a concise and comprehensive report of our method and findings
-   data/ - a subdirectory containing all the data that was used for evaluating model performance along with the code that was used to generate the data
-   code/ - all code that was used for the evaluation scripts as well as code that was used for running the HuggingFace and OpenAI models
-   output/ - a subdirectory containing all models' raw outputs on the evaluation sets, along with the gold labels

## Team Members

John-Wesley Appleton, Tin Do, Edmund Doerksen, William Qi
