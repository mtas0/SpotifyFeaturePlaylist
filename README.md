# Spotify Feature Playlist Generator

## Overview

This Python project is designed to analyze Spotify stream history metadata, extract audio features using the Spotify Web API, and generate a personalized playlist based on a trained neural network model. The process involves:

1. Analyzing Spotify stream history metadata.
2. Retrieving additional track information using the Spotify API.
3. Performing exploratory data analysis (EDA) on top tracks and features.
4. Creating and running an ensemble neural network model with popularity as the target variable and audio features as input.
5. Generating a playlist of songs with similar features to the user's preferred tracks.

## Author

**Merve H. Tas Bangert**

## Date

**January 30, 2024**

## Usage

### Prerequisites

- Python 3.12
- Required Python packages:
  - spotipy
  - pandas
  - scikit-learn
  - seaborn
  - matplotlib

### Configuration

Create a `config.py` file in the project directory with your Spotify API client ID and secret:

```python
# config.py
API_CID = 'your_client_id'
API_SECRET = 'your_client_secret'
```

### Running the Playlist Generator

1. Ensure the necessary dependencies are installed.
2. Run the Jupyter Notebook **spotify_feat_playlist_nb.ipynb** to create a playlist using your data.

### Input Data

- Audio metadata JSON files (not included in the repository for privacy reasons).
- DataFrame (**df_with_audio_features.csv**) with extracted audio features. This is my final output dataframe before training the model.

### Output

- Trained model saved as **spotify_mlp_model.pk1**.
- Generated playlist as a DataFrame.

### Project Structure
- **SpotFeatPlaylist.py**: Object-oriented script containing classes for Spotify audio analysis, data visualization, classifier training, and playlist creation.

### Acknowledgments
Special thanks to the Spotipy library and the Spotify API for enabling access to Spotify data.
Special thanks to michimalek for their valuable contributions to their spotify-random repository. The implementation for the **'get_wildcard'** and **'get_random'** methods in
my script were influenced by their work.