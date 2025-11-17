import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Importing datasets for data preparation.
"""
tracks = pd.read_csv("../enriched_datasets/tracks_enriched.csv")
artists = pd.read_csv("../enriched_datasets/artists.csv")

tracks = tracks.copy()
artists = artists.copy()

print(f"Initial dataset info:\n")
tracks.info()
artists.info()

# Print original number of tracks and artists
print(f"Tracks shape: {tracks.shape[0]} rows x {tracks.shape[1]} columns")
print(f"Artists shape: {artists.shape[0]} rows x {artists.shape[1]} columns")

print("\n \n \n")



"""
    Casting into correct types: tracks
"""
# Objects to strings
columns_to_string_tracks = ["id", "id_artist", "name_artist", "full_title", "title", "featured_artists", "primary_artist", "language", "album", "album_name", "album_type", "lyrics", "album_image", "id_album"]
for column in columns_to_string_tracks:
    tracks[column] = tracks[column].astype("string")

# Album release date: object -> datetime
tracks["album_release_date"] = pd.to_datetime(tracks["album_release_date"], errors="coerce")

# Floats/Objects to Ints
columns_to_ints_tracks = ["year", "month", "day", "popularity"]
for column in columns_to_ints_tracks:
    tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    tracks[column] = tracks[column].astype("Int64")

# Explicit: object -> boolean
print(tracks["explicit"].head())
tracks["explicit"] = tracks["explicit"].astype("bool")

print(f"Tracks after correct casting:\n")
tracks.info()

print("\n \n \n")



"""
    Casting into correct types: artists
"""
columns_to_string_artists = ["id_author", "name", "gender", "birth_place", "nationality", "description", "province", "region", "country", "source"]
for column in columns_to_string_artists:
    artists[column] = artists[column].astype("string")
    
columns_to_datetime_artists = ["birth_date", "active_start", "active_end"]
for column in columns_to_datetime_artists:
    artists[column] = pd.to_datetime(artists[column], errors='coerce')

print(f"Artists after correct casting:\n")
artists.info()

print("\n \n \n")



"""
    Removing duplicates:
    - track ID will not be useful as we don't merge datasets on it
    - checking if we have artists ID duplicates (on artist dataset)
    - checking if we have duplicates on the artists ID, full_title pair (tracks dataset)
    - dropping tracks with duplicated pair of features
"""
# IDs appearing more than once
duplicated_ids = artists["id_author"].value_counts()[artists["id_author"].value_counts() > 1]
print(f"Dupicated ids:\n", duplicated_ids)

duplicate_pairs = tracks[tracks.duplicated(subset=["id_artist", "full_title"], keep=False)]
print(f"Duplicated pairs:\n", duplicate_pairs)

# Keeping only first ID and full_title of the duplicates
tracks = tracks.drop_duplicates(subset=["id_artist", "full_title"], keep="first")

print(f"Tracks shape after removing duplicates: {tracks.shape[0]} rows x {tracks.shape[1]} columns")
print(f"Artists shape after removing duplicates: {artists.shape[0]} rows x {artists.shape[1]} columns")

print("\n \n \n")



"""
    Removing non-numeric values:
    - except artist/author IDs
    - except datetimes columns
    - except bools
    - except language
    - not useful for clustering
    - to be done before nans removal to avoid removing rows not for a good cause.
"""
# TRACKS
numeric_cols_t = tracks.select_dtypes(include=["number"]).columns
for col in numeric_cols_t:
    tracks[col] = pd.to_numeric(tracks[col], errors='coerce')

reduced_tracks = tracks[numeric_cols_t]

# Inserting artist id (string)
if "id_artist" in tracks.columns:
    reduced_tracks.insert(0, "id_artist", tracks["id_artist"])

# Inserting album_release (datetime)
if "album_release_date" in tracks.columns:
    reduced_tracks.insert(1, "album_release_date", tracks["album_release_date"])

# Inserting explicit, modified_popularity (bool)
if "explicit" in tracks.columns:
    reduced_tracks.insert(2, "explicit", tracks["explicit"])

if "modified_popularity" in tracks.columns:
    reduced_tracks.insert(3, "modified_popularity", tracks["modified_popularity"])

# Inserting language (string)
if "language" in tracks.columns:
    reduced_tracks.insert(4, "language", tracks["language"])
    

# ARTISTS
numeric_cols_a = artists.select_dtypes(include=["number"]).columns
for col in numeric_cols_a:
    artists[col] = pd.to_numeric(artists[col], errors='coerce')

reduced_artists = artists[numeric_cols_a]

# Inserting author id
if "id_author" in artists.columns:
    reduced_artists.insert(0, "id_author", artists["id_author"])

# Inserting datetimes
if "birth_date" in artists.columns:
    reduced_artists.insert(1, "birth_date", artists["birth_date"])

if "active_start" in artists.columns:
    reduced_artists.insert(2, "active_start", artists["active_start"])

if "active_end" in artists.columns:
    reduced_artists.insert(3, "active_end", artists["active_end"])

print(f"Tracks shape after removing non-numeric features: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")
print(f"Artists shape after removing non-numeric features: {reduced_artists.shape[0]} rows x {reduced_artists.shape[1]} columns")

print("\n \n \n")



"""
    Tracks nan values management:
    - from now on we work with the already reduced datasets
    - filling album release nans with track's year of release (valid for singles)
    - drop album_release_date
    - checking how many nans we have and in which columns and rows
    - removing columns with too many nans
"""
reduced_tracks["album_release_year"] = (
    reduced_tracks["album_release_date"].dt.year
    .fillna(reduced_tracks["year"])
)

reduced_tracks = reduced_tracks.drop(columns=["album_release_date"])

tracks_nan_per_feature = reduced_tracks.isna().sum()
print(f"Nan per feature (tracks):\n", tracks_nan_per_feature)

tracks_nan_per_row = reduced_tracks.isna().sum(axis=1)
print(f"Nan per row:\n", tracks_nan_per_row)

# Removing columns with more than 300 nans
cols_to_drop_tracks = tracks_nan_per_feature[tracks_nan_per_feature > 300].index
reduced_tracks = reduced_tracks.drop(columns=cols_to_drop_tracks)

print("Dropped columns for tracks:", list(cols_to_drop_tracks))

# Dropping rows with at least one nan
reduced_tracks = reduced_tracks.dropna()

# Checking remaining tracks
print(f"Tracks shape after removing nans: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")

print("\n \n \n")



"""
    Artists nan values management:
    - checking how many nans we have and in which columns and rows
    - if active_start_year is nan, we use the year of the first song instead
    - removing columns with too many nans
"""
# Active start year
first_year_by_artist = reduced_tracks.groupby("id_artist")["year"].min()
first_year_by_artist.name = "first_track_year"

print(first_year_by_artist.head())

reduced_artists = reduced_artists.merge(
    first_year_by_artist,
    left_on="id_author",
    right_on="id_artist",
    how="left"
)

print(reduced_artists.columns)

reduced_artists["active_start_year"] = (
    reduced_artists["active_start"].dt.year
    .fillna(reduced_artists["first_track_year"])
)

reduced_artists = reduced_artists.drop(columns=["active_start"])


artists_nan_per_feature = reduced_artists.isna().sum()
print(f"Nan per feature (artists):\n", artists_nan_per_feature)

artists_nan_per_row = reduced_artists.isna().sum(axis=1)
print(f"Nan per row (artists):\n", artists_nan_per_row)

# Removing columns with more than 20 nans
cols_to_drop_artists = artists_nan_per_feature[artists_nan_per_feature > 20].index
reduced_artists = reduced_artists.drop(columns=cols_to_drop_artists)

print("Dropped columns for artists:", list(cols_to_drop_artists))

# Dropping rows with at least one nan
reduced_artists = reduced_artists.dropna()

#Checking remaining artists
print(f"Artists shape after removing nans: {reduced_artists.shape[0]} rows x {reduced_artists.shape[1]} columns")

# Final check on all nans (should both be zero)
print(f"Making sure we dropped nans:\n")

tracks_nan_per_feature = reduced_tracks.isna().sum()
print(f"Nan per feature after nan removal(tracks):\n", tracks_nan_per_feature)

artists_nan_per_feature = reduced_artists.isna().sum()
print(f"Nan per feature after nan removal(artists):\n", artists_nan_per_feature)

print("\n \n \n")



"""
    INVALID VALUES REMOVAL for features with known domains:
    - popularity [0, 100]
    - label as explicit a song with swear words
    - year, album_release_year <=2025
    - zcr, flatness [0, 1]
    - Italian latitude [35, 47]
    - Italian longitude [4, 19]
"""
# popularity
reduced_tracks = reduced_tracks[
    (reduced_tracks["popularity"] >= 0) &
    (reduced_tracks["popularity"] <= 100)
]

print(f"Checking min popularity:", reduced_tracks["popularity"].min())
print(f"Checking max popularity:", reduced_tracks["popularity"].max())

print(f"Tracks shape after removing invalid popularity: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")

print("\n")

# swear_IT
count_invalid_swear_IT = (reduced_tracks["swear_IT"] < 0).sum()
print(f"Invalid swear_IT counter:", count_invalid_swear_IT)

print("\n")

#swear_EN
count_invalid_swear_EN = (reduced_tracks["swear_EN"] < 0).sum()
print(f"Invalid swear_EN counter:", count_invalid_swear_EN)

# explicit
count_swear = ((reduced_tracks["swear_IT"] > 0) | (reduced_tracks["swear_EN"] > 0)).sum()
print(f"Number of rows with swear words:", count_swear)

count_explicit = reduced_tracks["explicit"].sum()
print(f"Counter of explicit tracks:", count_explicit)

reduced_tracks["explicit"] = (
    (reduced_tracks["swear_IT"] > 0) |
    (reduced_tracks["swear_EN"] > 0)
).astype(int)

print(f"Counter of explicit tracks after correction:", reduced_tracks["explicit"].sum())

print("\n")

# year
reduced_tracks = reduced_tracks[reduced_tracks["year"] <= 2025]

print(f"Tracks shape after removing invalid year: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")

print("\n")

# album_release_year
reduced_tracks = reduced_tracks[reduced_tracks["album_release_year"] <= 2025]

print(f"Tracks shape after removing invalid album release year: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")

print("\n")

# zcr
count_invalid_zcr = ((reduced_tracks["zcr"] < 0) | (reduced_tracks["zcr"] > 1)).sum()
print(f"Invalid number of zcr:", count_invalid_zcr)

print("\n")

# flatness
count_invalid_flatness = ((reduced_tracks["flatness"] < 0) | (reduced_tracks["flatness"] > 1)).sum()
print(f"Invalid number of flatness:", count_invalid_flatness)

print("\n")

# disc_number
print(f"Disc number min value:", reduced_tracks["disc_number"].min())
print(f"Disc number max value:", reduced_tracks["disc_number"].max())

print("\n")

#track_number
print(f"Track number min value:", reduced_tracks["track_number"].min())
print(f"Track number max value:", reduced_tracks["track_number"].max())

print("\n")

#duration_ms
print(f"Duration min value:", reduced_tracks["duration_ms"].min())
print(f"Duration max value:", reduced_tracks["duration_ms"].max())

print("\n")

# modified_popularity
print(f"Modified popularity min value:", reduced_tracks["modified_popularity"].min())
print(f"Modified popularity max value:", reduced_tracks["modified_popularity"].max())
print("Not useful. Dropping column.")

reduced_tracks = reduced_tracks.drop(columns=["modified_popularity"])

print("\n \n \n")



# active_start_year
reduced_artists["birth_date_year"] = (
    reduced_artists["birth_date"].dt.year
)

reduced_artists = reduced_artists.drop(columns=["birth_date"])

count_invalid_active_start = (reduced_artists["active_start_year"] <  reduced_artists["birth_date_year"]).sum()
print(f"Invalid number of active start year:", count_invalid_active_start)

mask_invalid_active_start = reduced_artists["active_start_year"] < reduced_artists["birth_date_year"]

reduced_artists.loc[mask_invalid_active_start, "active_start_year"] = (
    reduced_artists["birth_date_year"] + 10
)

print("\n")

# latitude
count_invalid_latitude = ((reduced_artists["latitude"] < 35) | (reduced_artists["latitude"] > 47)).sum()
print(f"Invalid number of latitude:", count_invalid_latitude)

invalid_latitude = reduced_artists[
    (reduced_artists["latitude"] <= 35) |
    (reduced_artists["latitude"] >= 47)
]

print(f"Artists with invalid latitude:", invalid_latitude[["id_author", "latitude", "longitude"]])
print("Checked in dataset: born in Santo Domingo, italian.")

print("\n")

# longitude
count_invalid_longitude = ((reduced_artists["longitude"] < 4) | (reduced_artists["longitude"] > 19)).sum()
print(f"Invalid number of longitude:", count_invalid_longitude)

invalid_longitude = reduced_artists[
    (reduced_artists["longitude"] <= 4) |
    (reduced_artists["longitude"] >= 19)
]

print(f"Artists with invalid longitude:", invalid_longitude[["id_author", "latitude", "longitude"]])
print("Checked in dataset: born in Santo Domingo, italian.")

print("\n \n \n")



"""
    Remove outliers:
    1. bpm, centroid, rolloff, flux, rms, spectral_complexity, pitch >= 0
        - checking if they are in the expected range
        - removing outliers with the sigma method (more malleable)

    2. year, album_release_year
        - already removed > upper bound
        - removing outliers on lower bound w/ sigma method

    3. lexical features
        - making sure the lower bound isn't negative
        - choosing not to remove outliers (sometimes a single letter is counted as token)

    4. birth_date_year, active_start_year
"""
# === 1. ===
sound_outlier_mask_total = pd.Series(False, index=reduced_tracks.index)

sound_cols = ["bpm", "centroid", "rolloff", "flux", "rms", "spectral_complexity", "pitch", "loudness"]
for feature in sound_cols:
    col = reduced_tracks[feature]

    min = col.min()
    max = col.max()
    median = col.median()
    mu = col.mean()
    sigma = col.std()

    lower = mu - 4 * sigma
    upper = mu + 4 * sigma

    outlier_mask = (col < lower) | (col > upper)
    n_outliers = outlier_mask.sum()

    print("="*40)
    print(f"Feature: {feature}")
    print(f"Min value: {min}")
    print(f"Max value: {max}")
    print(f"Median value:{median}")
    print(f"Acceptable domain (4*sigma): [{lower:.3f}, {upper:.3f}]")
    print(f"Outliers found: {n_outliers}")

    sound_outlier_mask_total |= outlier_mask

# Counts total of outliers
print("Sound outliers rows:", sound_outlier_mask_total.sum())

# Removing outliers rows
reduced_tracks = reduced_tracks[~sound_outlier_mask_total]

print("\n")
print(f"Tracks shape after sound outliers removal: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")
print("\n")



# === 2. ===
tracks_year_outlier_mask_total = pd.Series(False, index=reduced_tracks.index)

tracks_datetime_cols = ["year", "album_release_year"]
for feature in tracks_datetime_cols:
    col = reduced_tracks[feature]

    min = col.min()
    max = col.max()
    median = col.median()
    mu = col.mean()
    sigma = col.std()

    lower = mu - 5 * sigma
    upper = 2025

    outlier_mask = (col < lower) | (col > upper)
    n_outliers = outlier_mask.sum()

    print("="*40)
    print(f"Feature: {feature}")
    print(f"Min value: {min}")
    print(f"Max value: {max}")
    print(f"Median value:{median}")
    print(f"Acceptable domain (5*sigma): [{lower:.3f}, {upper:.3f}]")
    print(f"Outliers found: {n_outliers}")

    tracks_year_outlier_mask_total |= outlier_mask

# Counts total of outliers
print("Tracks year outliers rows:", tracks_year_outlier_mask_total.sum())

# Removing outliers rows
reduced_tracks = reduced_tracks[~tracks_year_outlier_mask_total]

print("\n")
print(f"Tracks shape after year outliers removal: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")
print("\n")



# === 3. ===
lexical_cols = ["n_sentences", "n_tokens", "tokens_per_sent", "char_per_tok", "lexical_density", "avg_token_per_clause"]

# Choosing only some columns to clean
cols_to_clean = ["tokens_per_sent", "avg_token_per_clause"]

for feature in lexical_cols:
    col = reduced_tracks[feature]

    mu = col.mean()
    sigma = col.std()
    lower = mu - 5 * sigma
    upper = mu + 5 * sigma

    outlier_mask = (col < lower) | (col > upper)
    n_outliers = outlier_mask.sum()

    print("="*40)
    print(f"Feature: {feature}")
    print(f"Min value: {col.min()}")
    print(f"Max value: {col.max()}")
    print(f"Median value: {col.median()}")
    print(f"Acceptable domain (5*sigma): [{lower:.3f}, {upper:.3f}]")
    print(f"Outliers found: {n_outliers}")

    # Outlier removal only on chosen columns
    if feature in cols_to_clean:
        reduced_tracks = reduced_tracks[~outlier_mask]
        print(f"Removed {n_outliers} outliers from {feature}")
    else:
        print("Skipped outlier removal for this feature.")

print("\n")
print(f"Tracks shape after lexical outliers removal: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")
print("\n")



# === 4. ===
artists_datetime_cols = ["birth_date_year", "active_start_year"]
for feature in artists_datetime_cols:
    col = reduced_artists[feature]

    min = col.min()
    max = col.max()
    median = col.median()
    mu = col.mean()
    sigma = col.std()

    lower = mu - 3 * sigma
    upper = 2025

    outlier_mask = (col < lower) | (col > upper)
    n_outliers = outlier_mask.sum()

    print("="*40)
    print(f"Feature: {feature}")
    print(f"Min value: {min}")
    print(f"Max value: {max}")
    print(f"Median value:{median}")
    print(f"Acceptable domain (3*sigma): [{lower:.3f}, {upper:.3f}]")
    print(f"Outliers found: {n_outliers}")

print("\n")
print("No need to remove anything.")
print("\n")

print("\n \n \n")



print(f"Tracks shape after data preparation: {reduced_tracks.shape[0]} rows x {reduced_tracks.shape[1]} columns")
print(f"Artists shape after after data preparation: {reduced_artists.shape[0]} rows x {reduced_artists.shape[1]} columns")

print(f"Tracks sample:\n", reduced_tracks.head())
print(f"Artists sample:\n", reduced_artists.head())

print(f"Tracks types:", reduced_tracks.info())
print(f"Artists types:", reduced_artists.info())

"""
    Save new datasets
"""
reduced_tracks.to_csv('../prepared_datasets/tracks.csv', index=False)
reduced_artists.to_csv('../prepared_datasets/artists.csv', index=False)
