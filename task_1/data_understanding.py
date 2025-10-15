import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# load datasets from CSV to pandas df
# artsists is delimited using ; so use sep=','
artists = pd.read_csv("../datasets/artists.csv", sep=';')
tracks = pd.read_csv("../datasets/tracks.csv")

## Artists
# Check how many NaNs are present and plot the number of NaN per each column
nan_pct_artists = (artists.isnull().sum() / len(artists)) * 100
non_nan_artists = (artists.notnull().sum() / len(artists)) * 100

plot_data = pd.DataFrame({
    'NaN Values %': nan_pct_artists,
    'Non-NaN Values %': non_nan_artists
})

# stacked bar to see the proportion of NaN vs non NaN in each column for 
ax = plot_data.plot(kind='bar', stacked=True, 
                     figsize=(12, 6),
                     color=['#e74c3c', '#2ecc71'])
ax.set_title('NaN Percentage Per Column (Artists Dataset)', fontsize=12, fontweight='bold')
ax.set_xlabel('Columns', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

## Tracks
# Check how many NaNs are present and plot the number of NaN per each column
nan_pct_tracks = (tracks.isnull().sum() / len(tracks)) * 100
non_nan_pct_tracks = (tracks.notnull().sum() / len(tracks)) * 100

plot_data = pd.DataFrame({
    'NaN Values %': nan_pct_tracks,
    'Non-NaN Values %': non_nan_pct_tracks
})

# stacked bar to see the proportion of NaN vs non NaN in each column for 
ax = plot_data.plot(kind='bar', stacked=True, 
                     figsize=(12, 6),
                     color=['#e74c3c', '#2ecc71'])
ax.set_title('NaN Percentage Per Column (Tracks Dataset)', fontsize=12, fontweight='bold')
ax.set_xlabel('Columns', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

###################

# 'DType' is the effective type used in the dataframe, we can see that many columns need casting to the expected data type
# let's start with artists
artists.info()
artists
# these columns are strings from what we can see from the dataset by using artists.head()
columns_to_string   = ["id_author", "name", "gender", "nationality", "description", "province", "region", "country"]
for column in columns_to_string:
    artists[column] = artists[column].astype('string')
    
# these columns need to be converted to datetime, the native pandas date type
columns_to_datetime = ["birth_date", "birth_place", "active_start", "active_end"]
for column in columns_to_datetime:
    artists[column] = pd.to_datetime(artists[column], errors='coerce')

# latitude and longitude are already float 64, so no casting is needed

###################
# show datasets type information
tracks.info()

columns_to_string   = ["id", "id_artist", "name_artist", "full_title", "title", "featured_artists", "primary_artist", "language", "album", "album_name", "album_type", "lyrics", "album_image", "id_album"]
for column in columns_to_string:
    tracks[column] = tracks[column].astype('string')
    
# these columns are array of strings
columns_to_array = ["swear_IT_words", "swear_EN_words"]

# to datetime
tracks['album_release_date'] = pd.to_datetime(tracks['album_release_date'], errors='coerce')
tracks['popularity'] = tracks['popularity'].apply(pd.to_numeric, errors='coerce')
tracks['popularity'] = tracks['popularity'].astype('Int64')

tracks['explicit'] = tracks['explicit'].astype('bool')

# different values, like NaN or 2021.0 so cast to int
tracks['year'] = tracks['year'].apply(pd.to_numeric, errors='coerce')
tracks["year"] = tracks['year'].astype('Int64')

# From the project description
# album: Album to which the track belongs.
# albumname: Name of the album
