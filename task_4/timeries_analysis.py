from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.utils import to_time_series_dataset
import plotly.express as px
import pandas as pd
import numpy as np
import librosa
import time
import os

# base path
mp3_base_path = "../fedez_fibra/"
dataset_output_dir = "../timeseries_datasets/"

df = pd.read_csv("../timeseries_datasets/tracks_timeseries.csv")

df['centroid_num_samples'] = df["centroid"].apply(lambda x: len(x))
df['rollof_num_samples'] = df["rolloff"].apply(lambda x: len(x))
df['flux_num_samples'] = df["flux"].apply(lambda x: len(x))
df['rms_num_samples'] = df["rms"].apply(lambda x: len(x))
df['zcr_num_samples'] = df["zcr"].apply(lambda x: len(x))
df['spectral_bw_num_samples'] = df["spectral_bw"].apply(lambda x: len(x))

df