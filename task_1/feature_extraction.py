import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



"""
    Opening .csv file using Pandas df.
"""
tracks = pd.read_csv("../enriched_datasets/tracks_enriched.csv")
artists = pd.read_csv("../enriched_datasets/artists.csv")



"""
    Uncorrect value type temporary management.
    - Finds numeric columns in both datasets
    - Forces popularity as a numeric value
    - Fills non-number values with -1
    - Drop active_end column
    - Takes into account only the active_start year
    - Maps the artist gender into a boolean
    TODO Understand how we want to actually manage NaNs,
        missing values etc.
    TODO Understand why popularity is not recognised as numeric.
"""
def data_filling(tracks, artists):
    tracks = tracks.copy()
    artists = artists.copy()

    # Tracks
    tracks["popularity"] = pd.to_numeric(tracks["popularity"], errors='coerce')

    numeric_cols_t = tracks.select_dtypes(include=["number"]).columns
    
    for col in numeric_cols_t:
        tracks[col] = pd.to_numeric(tracks[col], errors='coerce').fillna(-1)
    
    numeric_tracks = tracks[numeric_cols_t]
    
    # Artists
    if "active_end" in artists.columns:
        artists = artists.drop(columns=["active_end"])

    if "active_start" in artists.columns:
        artists["active_start_year"] = pd.to_datetime(artists["active_start"], errors="coerce").dt.year

    if "gender" in artists.columns:
        artists["gender_numeric"] = artists["gender"].map({"M": 0, "F": 1})

    numeric_cols_a = artists.select_dtypes(include=["number"]).columns

    for col in numeric_cols_a:
        artists[col] = pd.to_numeric(artists[col], errors='coerce').fillna(-1)
    
    numeric_artists = artists[numeric_cols_a]

    # Verifies status
    print(f"Numeric tracks sample:\n", numeric_tracks.head())
    print(f"Numeric artists sample:\n", numeric_artists.head())
    return numeric_tracks, numeric_artists



"""
    Plotting a correlation heatmap over the numeric values for:
    1. tracks
    2. artist
    3. both
"""
def og_tracks_heatmap(numeric_tracks):
    og_tracks_corr = numeric_tracks.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(og_tracks_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations between original tracks features")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.show()
    return

def og_artists_heatmap(numeric_artists):
    og_artists_corr = numeric_artists.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(og_artists_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations between original artists features")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()
    return

def og_full_heatmap(numeric_tracks, numeric_artists):
    numeric_feats = pd.concat([numeric_tracks, numeric_artists], axis=1)

    og_corr = numeric_feats.corr()

    # Print couples of features with high correlation
    print("Couples of features with correlation > 0.30:\n")
    high_corr = []
    for i in range(len(og_corr.columns)):
        for j in range(i):
            corr_value = og_corr.iloc[i, j]
            if abs(corr_value) > 0.30:
                feat1 = og_corr.columns[i]
                feat2 = og_corr.columns[j]
                high_corr.append((feat1, feat2, corr_value))
                print(f"{feat1}, {feat2} = {corr_value:.2f}")

    plt.figure(figsize=(18, 14))
    sns.heatmap(og_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, linecolor='gray', annot_kws={"size": 8})
    plt.title("Heatmap of correlations between original features")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    return



"""
    LANGUAGE FEATURES
"""
def swear_ratio(tracks):
    tracks["swear_ratio"] = (tracks["swear_IT"] + tracks["swear_EN"])/tracks["n_tokens"]
    return tracks

# Re-evaluate
def syntactic_complexity(tracks):
    tracks["syntactic_complexity"] = tracks["tokens_per_sent"] * tracks["avg_token_per_clause"]
    return tracks

def text_density(tracks):
    tracks["text_density"] = tracks["n_tokens"] / tracks["n_sentences"]
    # check for Nans
    return tracks



"""
    SOUND FEATURES
    TODO Check all the math behind it
"""
def percussivness(tracks):
    tracks["percussivness"] = tracks["zcr"] * tracks["rolloff"]
    return tracks

def modulation_index(tracks):
    tracks["modulation_index"] = tracks["flux"] / tracks["pitch"]
    return tracks

def energy_index(tracks):
    tracks["energy_index"] = (tracks["rms"] + tracks["loudness"])/2
    return tracks

def norm_energy_index(tracks):
    energy_index = (tracks["rms"] + tracks["loudness"])

    energy_complexity = tracks["spectral_complexity"] * tracks["rms"]

    tracks["norm_energy_index"] = energy_index / energy_complexity
    return tracks

def timbre_brightness(tracks):
    tracks["timbre_brightness"] = (tracks["centroid"] + tracks["rolloff"])/2
    return tracks

def noise_ratio(tracks):
    tracks["noise_ratio"] = tracks["zcr"] * tracks["flatness"]
    return tracks

def rythmic_complexity(tracks):
    tracks["rythmic_complexity"] = tracks["bpm"] * tracks["flux"]
    return tracks



"""
    POPULARITY FEATURES
"""
def relative_popularity(tracks):
    # Mean popularity per artist
    artist_pop_mean = tracks.groupby("id_artist")["popularity"].transform("mean")

    # Relative popularity of the song w respect to the others from the same artist
    tracks["relative_popularity"] = tracks["popularity"] / artist_pop_mean
    return tracks





"""
    Creating a new dataframe with both old and new features.
"""
def create_df(tracks: pd.DataFrame) -> pd.DataFrame:
    tracks = swear_ratio(tracks)
    tracks = syntactic_complexity(tracks)
    tracks = energy_index(tracks)
    tracks = timbre_brightness(tracks)
    tracks = noise_ratio(tracks)
    tracks = rythmic_complexity(tracks)
    tracks = relative_popularity(tracks)
    return tracks



if __name__ == "__main__":
    numeric_tracks, numeric_artists = data_filling(tracks, artists)

    og_tracks_heatmap(numeric_tracks)

    og_artists_heatmap(numeric_artists)

    og_full_heatmap(numeric_tracks, numeric_artists)

    # Generates features enriched DataFrame
    tracks_new_features = create_df(tracks)

    ## Defines output path
    #output_path = "../enriched_datasets/tracks_features_enriched.csv"
#
    ## Saves new dataset (old + new features)
    #tracks_new_features.to_csv(output_path, index=False)
#
    #print(f"File saved in: {output_path}")



