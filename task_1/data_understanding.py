import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plotting import *

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# load datasets from CSV to pandas df
# artsists is delimited using ; so use sep=','
artists = pd.read_csv("../original_datasets/artists.csv", sep=';')
tracks = pd.read_csv("../original_datasets/tracks.csv")

## Datasets shape
artists_shape = artists.shape
tracks_shape = tracks.shape
print(f"Artists rows: {artists_shape[0]}, columns: {artists_shape[1]}")
print(f"Tracks  rows: {tracks_shape[0]}, columns: {tracks_shape[1]}")

## Let's see some row values by using df.head()
artists.head()
tracks.head()

############################
# Data quality
# - Missing values (NaN)
# - Duplicate records
# - Data types casting
# - Outliers and anomalie

## Artists
# Check how many NaNs are present and plot the number of NaN per each column
# from the distribution we can see that no active end date is present, and about half active start
plot_nans_stacked(artists, 'NaN Percentage Per Column (Artists Dataset)')

## Let's import augmented data from our search, we manually searched the result and saved the url in the last column for easy reference
artists_search = pd.read_csv("../original_datasets/artists_missing_vals.csv")
artists_search.head()

updated_ids = artists_search['id_author'].unique()

# remove rows from the original artists dataframe
artists_without_updates = artists[~artists['id_author'].isin(updated_ids)]

# concatenate the remaining original data with the updated data
artists_final = pd.concat([artists_search, artists_without_updates], ignore_index=True)

# save to csv
artists_final.to_csv("../enriched_datasets/artists.csv", index=False)

# Verify the result
print(f"Original artists: {len(artists)} rows")
print(f"Artists updated: {len(artists_search)} rows")
print(f"Artists without updated IDs: {len(artists_without_updates)} rows")
print(f"Final combined: {len(artists_final)} rows")

plot_nans_stacked(artists_final, 'NaN Percentage Per Column (Augmented Artists Dataset)')

## Tracks
# Check how many NaNs are present and plot the number of NaN per each column
plot_nans_stacked(tracks, 'NaN Percentage Per Column (Tracks Dataset)')

###################
## Data types, let's convert data types before doing duplicate analysis so that values in rows are in the correct datat type for comprare

# 'DType' is the effective type used in the dataframe, we can see that many columns need casting to the expected data type
# let's start with artists
artists.info()
artists.head()
artists.columns

# these columns are strings from what we can see from the dataset by using artists.head()
columns_to_string   = ["id_author", "name", "gender", "birth_place", "nationality", "description", "province", "region", "country"]
for column in columns_to_string:
    artists[column] = artists[column].astype('string')
    
# these columns need to be converted to datetime, the native pandas date type
columns_to_datetime = ["birth_date", "active_start", "active_end"]
for column in columns_to_datetime:
    artists[column] = pd.to_datetime(artists[column], errors='coerce')

# latitude and longitude are already float 64, so no casting is needed

###################
# show datasets type information
tracks.info()
tracks.head()

columns_to_string   = ["id", "id_artist", "name_artist", "full_title", "title", "featured_artists", "primary_artist", "language", "album", "album_name", "album_type", "lyrics", "album_image", "id_album"]
for column in columns_to_string:
    tracks[column] = tracks[column].astype('string')
    
# these columns are array of strings, let's leave them as objects
columns_to_array = ["swear_IT_words", "swear_EN_words"]

# to datetime
tracks['album_release_date'] = pd.to_datetime(tracks['album_release_date'], errors='coerce')
tracks['popularity'] = tracks['popularity'].apply(pd.to_numeric, errors='coerce')
tracks['popularity'] = tracks['popularity'].astype('Int64')

# from df.info we can see that this column is a boolean, so let's cast it to bool
tracks['explicit'] = tracks['explicit'].astype('bool')

# different values, like NaN or 2021.0 so cast to int
tracks['year'] = tracks['year'].apply(pd.to_numeric, errors='coerce')
tracks["year"] = tracks['year'].astype('Int64')

###################
## Duplicate analysis

# Tracks
tracks_duplicates = tracks.duplicated().sum()
print(f"Duplicates in tracks: {tracks_duplicates}")

# Check for duplicate track IDs
tracks_id_duplicates = tracks['id'].duplicated().sum()
print(f"Duplicate track ID: {tracks_id_duplicates}")

duplicate_tracks = tracks[tracks['id'].duplicated()]
dup_ids = tracks['id'][tracks['id'].duplicated(keep=False)].unique()

tracks[tracks["id"].isin(dup_ids)]

if 'title' in tracks.columns and 'primary_artist' in tracks.columns:
    tracks_content_duplicates = tracks.duplicated(subset=['title', 'primary_artist']).sum()
    
    print(f"Number of songs with same title and artist: {tracks_content_duplicates}")
    tracks[tracks.duplicated(subset=['title', 'primary_artist'])]

# Artists
artists_duplicates = artists.duplicated().sum()
print(f"Duplicate rows in artists: {artists_duplicates}")

artists_id_duplicates = artists['id_author'].duplicated().sum()
print(f"Duplicate artist IDs: {artists_id_duplicates}")

###################
# Variable distribution analysis
artists.columns

# Birth places distribution
plot_bar_chart_distribution(artists, 'birth_place', 'Birth Place', 'Occurrences', 'Artists Top 10 Birth Place')

# gender distribution
plot_bar_chart_distribution(artists, 'gender', 'Gender', 'Occurrences', 'Artists Gender Distribution')

# region distribution
plot_bar_chart_distribution(artists, 'region', 'Region', 'Occurrences', 'Artists Region Distribution')

# province distribution
plot_bar_chart_distribution(artists, 'province', 'Province', 'Occurrences', 'Artists Top 10 Provinces')

# countries distribution
plot_bar_chart_distribution(artists, 'country', 'Country', 'Occurrences', 'Artists Country Distribution')

# birth year
artists['birth_year'] = pd.to_datetime(artists['birth_date'], errors='coerce').dt.year
plot_histogram(artists, 'birth_year', 'Birth Year', 'Number of Authors', 'Distribution of Authors by Birth Year', nbins=50)

############
# Most used language
plot_bar_chart_distribution(tracks, 'language', 'Language', 'Occurrences', 'Artists Country Distribution')

# Swear words analysis
# let's first see some upper and lower bounds for popularity
print(f"Max popularity value: {tracks['popularity'].max()}")
print(f"Min popularity value: {tracks['popularity'].min()}")
print(f"Mean popularity: {tracks['popularity'].mean()}")
print(f"Median popularity: {tracks['popularity'].median()}")
print(f"Italian swear words [Max - Mean]: [{tracks['swear_IT'].max()} - {tracks['swear_IT'].mean():.2f}]")
print(f"English swear words [Max - Mean]: [{tracks['swear_EN'].max()} - {tracks['swear_EN'].mean():.2f}]")

# We can see that there are some odd values for popularity to explore, minimum value seems odd, same for max value which is very high, let's plot a distribution of popularity values
# we can observe that around 3x more italian swear words are used compared to english, this was expected as we are analyzing italian rap
# additionally we can see a low usage of swear words
plot_scatter(tracks, 'swear_IT', 'popularity', 'Swear Words', 'Popularity', 'Birth Year vs Career Start')

plt.figure(figsize=(12, 6))
plt.scatter(tracks['swear_IT'], tracks['popularity'], label='IT Swear Words')
plt.scatter(tracks['swear_EN'], tracks['popularity'], label='EN Swear Words')
plt.xlabel('Num Swear Words', fontsize=12)
plt.ylabel('Popularity', fontsize=12)
plt.title('Swear Words vs Track Popularity', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# We can see the map which resembles italy, partially
plot_scatter(artists, 'longitude', 'latitude', 'Longitude', 'Latitude', 'Geographic Distribution of Birth Places')

# We can see some expected correlation between birth year and activity start
plot_scatter(artists, 'birth_year', 'active_start', 'Birth Year', 'Carrer Start Year', 'Birth Year vs Career Start')

# let's see stats on starting carrer age
artists['active_start_year'] = pd.to_datetime(artists['active_start'], errors='coerce').dt.year
artists['age_at_start'] = artists['active_start_year'] - artists['birth_year']
plot_histogram(artists, 'age_at_start', 'Age At Carrer Start', 'Authors Count', 'Distribution of Age at Career Start', nbins=50)

print(f"Mean age at start: {artists['age_at_start'].mean():.2f}")
print(f"Median age at start: {artists['age_at_start'].median():.2f}")
print(f"Std dev: {artists['age_at_start'].std():.2f}")
