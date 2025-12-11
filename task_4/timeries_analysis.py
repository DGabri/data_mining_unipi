from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.utils import to_time_series_dataset
import plotly.express as px
import pandas as pd
import numpy as np
import librosa
import os

# base path
mp3_base_path = "../fedez_fibra/"
tracks_list = os.listdir(mp3_base_path)

number_tracks = len(tracks_list)
print(f"Loaded: {number_tracks} mp3")

# check if tracks are in the origianl tracks dataset
tracks = pd.read_csv("../original_datasets/tracks.csv")

# parse fname -> artistID - trackID
tracks_info = []

for track in tracks_list:
    # :-4 take fname without .mp3
    # split on " - " to get left and right part
    splitted_fname = track[:-4].split(" - ")
    artist_id = splitted_fname[0]
    track_id = splitted_fname[1]
    
    track_info = {"id": track_id, "id_artist": artist_id}
    tracks_info.append(track_info)

tracks_info_df = pd.DataFrame(tracks_info)

# extract artists id and correlate it to name
unique_artists = tracks_info_df.id_artist.unique()
artists_names = list(tracks[tracks['id_artist'].isin(unique_artists)]['name_artist'].unique())

# check how many tracks are in the tracks df
track_matches = tracks_info_df['id'].isin(tracks['id']).sum()
missing_tracks_count = number_tracks - track_matches

print(f"Total mp3 tracks: {number_tracks} Matches found: {track_matches} Missing tracks: {missing_tracks_count}")

print(f"Artists names: {', '.join(artists_names)}")

#########################################################
# data exploration
# load all files, extract sampling rate and shape of data

tracks_stats = []

for i in range(number_tracks):
    fname = tracks_list[i]
    data, sr = librosa.load(mp3_base_path+fname)
    num_samples = data.shape[0]
    track_info = {"id": fname, "num_samples": num_samples, "sr": sr}
    tracks_stats.append(track_info)
    print(f"[{i}/{number_tracks}]Loaded: {fname}")


# test plot one timeseries
data, sr = librosa.load(mp3_base_path+tracks_list[2])

df = pd.DataFrame(data, columns=["value"])

fig = px.line(df, x=df.index, y=df['value'], title='data samples')
fig.show()
