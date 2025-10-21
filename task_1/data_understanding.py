import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from plotting_functions import *

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# load datasets from CSV to pandas df
# artsists is delimited using ; so use sep=','
artists = pd.read_csv("../datasets/artists.csv", sep=';')
tracks = pd.read_csv("../datasets/tracks.csv")

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
nan_count_artists = artists.isnull().sum()
nan_pct_artists = (nan_count_artists / len(artists)) * 100
non_nan_artists = (artists.notnull().sum() / len(artists)) * 100

plot_data = pd.DataFrame({
    'NaN Values %': nan_pct_artists,
    'NaN Values': nan_count_artists,
    'Non-NaN Values %': non_nan_artists
})

plot_data = plot_data.sort_values(by=['NaN Values'], ascending=False)
plot_data

plot_data.drop("NaN Values", axis=1, inplace=True)
# from the distribution we can see that no active end date is present, and about half active start
# Region province and country are

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
nan_count_tracks = tracks.isnull().sum()
nan_pct_tracks = (nan_count_tracks / len(tracks)) * 100
non_nan_pct_tracks = (tracks.notnull().sum() / len(tracks)) * 100

plot_data = pd.DataFrame({
    'NaN Values %': nan_pct_tracks,
    'NaN Values': nan_count_tracks,
    'Non-NaN Values %': non_nan_pct_tracks
})

plot_data = plot_data.sort_values(by=['NaN Values'], ascending=False)
plot_data

# plot
plot_data.drop("NaN Values", axis=1, inplace=True)

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
birth_places = artists['birth_place'].value_counts().head(10)
plt.figure(figsize=(12, 6)) 
plt.bar(range(len(birth_places)), birth_places.values)
plt.xticks(range(len(birth_places)), birth_places.index, rotation=45, ha='right')
plt.xlabel('Birth Place')
plt.ylabel('Occurrences')
plt.title('Artists Top 10 Birth Place')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# gender distribution
gender = artists['gender'].value_counts()
plt.figure(figsize=(12, 6)) 
plt.bar(range(len(gender)), gender.values)
plt.xticks(range(len(gender)), gender.index, rotation=45, ha='right')
plt.xlabel('Gender')
plt.ylabel('Occurrences')
plt.title('Artists Gender Distribution')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# region distribution
regions = artists['region'].value_counts().head(10)
plt.figure(figsize=(12, 6)) 
plt.bar(range(len(regions)), regions.values)
plt.xticks(range(len(regions)), regions.index, rotation=45, ha='right')
plt.xlabel('Region')
plt.ylabel('Occurrences')
plt.title('Artists Region Distribution')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# province distribution
province = artists['province'].value_counts().head(10)
plt.figure(figsize=(12, 6)) 
plt.bar(range(len(province)), province.values)
plt.xticks(range(len(province)), province.index, rotation=45, ha='right')
plt.xlabel('Province')
plt.ylabel('Occurrences')
plt.title('Artists Top 10 Provinces')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# countries distribution
countries = artists['country'].value_counts()
plt.figure(figsize=(12, 6)) 
plt.bar(range(len(countries)), countries.values)
plt.xticks(range(len(countries)), countries.index, rotation=45, ha='right')
plt.xlabel('Country')
plt.ylabel('Occurrences')
plt.title('Artists Country Distribution')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Most used language
lang_counts = tracks['language'].value_counts().head(10)
plt.figure(figsize=(12, 6))
plt.bar(range(len(lang_counts)), lang_counts.values)
plt.xticks(range(len(lang_counts)), lang_counts.index, rotation=45, ha='right')
plt.xlabel('Language')
plt.ylabel('Occurrences')
plt.yscale('log') 
plt.title('Top 10 Languages Used')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


############
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
plt.figure(figsize=(12, 6))
plt.scatter(tracks['swear_IT'], tracks['popularity'])
plt.xlabel('Swear Words')
plt.ylabel('Popularity')
plt.title('Birth Year vs Career Start')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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

artists['birth_year'] = pd.to_datetime(artists['birth_date'], errors='coerce').dt.year
plt.figure(figsize=(12, 6))
plt.hist(artists['birth_year'].dropna(), bins=50)
plt.xlabel('Birth Year')
plt.ylabel('Number of Authors')
plt.title('Distribution of Authors by Birth Year')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# We can see the map which resembles italy, partially
plt.figure(figsize=(14, 8))
plt.scatter(artists['longitude'], artists['latitude'], s=10)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of Birth Places')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# We can see some expected correlation between birth year and activity start
plt.figure(figsize=(12, 6))
plt.scatter(artists['birth_year'], artists['active_start'])
plt.xlabel('Birth Year')
plt.ylabel('Carrer Start Year')
plt.title('Birth Year vs Career Start')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# let's see stats on starting carrer age
artists['active_start_year'] = pd.to_datetime(artists['active_start'], errors='coerce').dt.year
artists['age_at_start'] = artists['active_start_year'] - artists['birth_year']

plt.figure(figsize=(12, 6))
plt.hist(artists['age_at_start'].dropna(), bins=50)
plt.xlabel('Age At Carrer Start')
plt.ylabel('Authors Count')
plt.title('Distribution of Age at Career Start')
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean age at start: {artists['age_at_start'].mean():.2f}")
print(f"Median age at start: {artists['age_at_start'].median():.2f}")
print(f"Std dev: {artists['age_at_start'].std():.2f}")
