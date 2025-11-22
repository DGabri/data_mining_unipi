import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



"""
    Plotting a correlation heatmap over the numeric values for:
    1. tracks
    2. artist
    3. both
"""
def og_tracks_heatmap(tracks):
    numeric_cols = tracks.select_dtypes(include=[np.number]).columns
    
    # Removing boolean
    if "explicit" in numeric_cols:
        numeric_cols = numeric_cols.drop("explicit")

    tracks_for_corr = tracks[numeric_cols]

    og_tracks_corr = tracks_for_corr.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(og_tracks_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations - original tracks features")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    return

def og_artists_heatmap(artists):
    numeric_cols = artists.select_dtypes(include=[np.number]).columns
    artists_for_corr = artists[numeric_cols]

    og_artists_corr = artists_for_corr.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(og_artists_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations - original artists features")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()
    return

def og_full_heatmap(merged_dataset):
    # Selecting only numeric columns
    numeric_cols = merged_dataset.select_dtypes(include=[np.number]).columns

    # Removing boolean
    if "explicit" in numeric_cols:
        numeric_cols = numeric_cols.drop("explicit")

    og_for_corr = merged_dataset[numeric_cols]

    og_corr = og_for_corr.corr()

    # Print couples of features with high correlation
    print("\nCouples of features with correlation > 0.70:")
    high_corr = []
    for i in range(len(og_corr.columns)):
        for j in range(i):
            corr_value = og_corr.iloc[i, j]
            if abs(corr_value) > 0.70:
                feat1 = og_corr.columns[i]
                feat2 = og_corr.columns[j]
                high_corr.append((feat1, feat2, corr_value))
                print(f"{feat1}, {feat2} = {corr_value:.2f}")

    plt.figure(figsize=(18, 14))
    sns.heatmap(og_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, linecolor='gray', annot_kws={"size": 8})
    plt.title("Heatmap of correlations - original features")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    return 



"""
    LEXICAL FEATURES
"""
def swear_ratio(merged_dataset):
    df = merged_dataset.copy()

    df["swear_ratio"] = (df["swear_IT"] + df["swear_EN"])/df["n_tokens"]
    return df

def syntactic_complexity(merged_dataset):
    df = merged_dataset.copy()

    df["syntactic_complexity"] = df["n_tokens"] * df["char_per_tok"] * df["lexical_density"]
    return df

def flow_complexity(merged_dataset):
    df = merged_dataset.copy()

    df["flow_complexity"] = df["tokens_per_sent"] * df["flux"]
    return df

def multilingual_index(merged_dataset):
    df = merged_dataset.copy()
    
    df["multilingual_index"] = 0

    df.loc[merged_dataset["language"] != "it", "multilingual_index"] = 1
    return df



"""
    SOUND AND STYLE FEATURES
"""
def percussiveness(merged_dataset):
    df = merged_dataset.copy()

    df["percussiveness"] = df["rolloff"] * df["flux"] * df["rms"]
    return df

def timbre_brightness(merged_dataset):
    df = merged_dataset.copy()

    df["timbre_brightness"] = df["centroid"] + df["rolloff"] / 2
    return df

def harmonic_richness(merged_dataset):
    df = merged_dataset.copy()

    tonality = 1 - df["flatness"]

    df["harmonic_richness"] = tonality * df["spectral_complexity"] * df["pitch"]
    return df

def trap_index(merged_dataset):
    df = merged_dataset.copy()

    """
        Identifies trap characteristics:
        - bpm -> 130-170 (or 65-85 half-time)
        - Sub-bass (low rolloff)
        - high rms
    """
    # bpm score
    bpm_trap = np.where(
        ((df["bpm"] >= 130) & (df["bpm"] <= 170)) | 
        ((df["bpm"] >= 65) & (df["bpm"] <= 85)),
        1.0, 0.3
    )
    
    subbass_score = df["timbre_brightness"].max() - df["timbre_brightness"]
    
    df["trap_index"] = bpm_trap * subbass_score * df["rms"]
    return df

def boombap_index(merged_dataset):
    df = merged_dataset.copy()

    """
        Identifies boom-bap/old school:
        - bpm -> 85-95
        - emphasizes mid range
    """
    # bpm score
    bpm_bb = np.where((df["bpm"] >= 85) & (df["bpm"] <= 95), 1.0, 0.3)
    
    spectral_mid = df["spectral_complexity"].max() -\
        np.abs(df["spectral_complexity"] - (df["spectral_complexity"]/2)) * 2
    
    df["boombap_index"] = bpm_bb * spectral_mid
    return df

def cloud_rap_index(merged_dataset):
    df = merged_dataset.copy()

    """
        Identifies cloud rap:
        - bpm -> 60-80
        - low flatness (harmonic)
        - low flux (sound changing slowly) 
    """
    # bpm score
    bpm_cloud = np.where((df["bpm"] >= 60) & (df["bpm"] <= 80), 1.0, 0.3)
    
    flatness_low = df["flatness"].max() - df["flatness"]

    flux_low = df["flux"].max() - df["flux"]
    
    df["cloud_rap_index"] = bpm_cloud * flatness_low * flux_low
    return df

def drill_index(merged_dataset):
    df = merged_dataset.copy()

    """
    Identifies UK Drill style:
    - bpm -> 135-150
    - low rolloff (dark sound)
    - Sliding bass
    """
    # BPM drill
    bpm_drill = np.where((df["bpm"] >= 135) & (df["bpm"] <= 150), 1.0, 0.3)

    subbass_score = df["timbre_brightness"].max() - df["timbre_brightness"]

    df["drill_index"] = bpm_drill * subbass_score * df["pitch"] * df["flatness"]
    return df



"""
    METADATA FEATURES
"""
def career_longevity(merged_dataset):
    df = merged_dataset.copy()

    current_year = datetime.now().year
    df["career_longevity"] = current_year - df["active_start_year"]
    return df



"""
    Final correlation matrices:
    1. all new features
    2. after non-useful columns removal
"""
def full_heatmap(final_merged_dataset):
    # Selecting only numeric columns
    numeric_cols = final_merged_dataset.select_dtypes(include=[np.number]).columns

    # Removing booleans
    if "explicit" in numeric_cols:
        numeric_cols = numeric_cols.drop("explicit")

    if "multilingual_index" in numeric_cols:
        numeric_cols = numeric_cols.drop("multilingual_index")

    final_for_corr = final_merged_dataset[numeric_cols]
    
    enriched_full_corr = final_for_corr.corr()
    
    # Print couples of features with high correlation
    print("\nCouples of features with correlation > 0.70")
    high_corr = []
    for i in range(len(enriched_full_corr.columns)):
        for j in range(i):
            corr_value = enriched_full_corr.iloc[i, j]
            if abs(corr_value) > 0.70:
                feat1 = enriched_full_corr.columns[i]
                feat2 = enriched_full_corr.columns[j]
                high_corr.append((feat1, feat2, corr_value))
                print(f"{feat1}, {feat2} = {corr_value:.2f}")
    
    print(f"\nTotal couples with correlation > 0.70: {len(high_corr)}")

    plt.figure(figsize=(18, 14))
    sns.heatmap(enriched_full_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, linecolor='gray', annot_kws={"size": 8})
    plt.title("Heatmap of correlations - enriched features")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    return



if __name__ == "__main__":
    # Opening files
    tracks = pd.read_csv("../prepared_datasets/tracks.csv")
    artists = pd.read_csv("../prepared_datasets/artists.csv")

    tracks = tracks.copy()
    artists = artists.copy()

    print(f"Tracks shape: {tracks.shape[0]} rows x {tracks.shape[1]} columns")
    print(f"Artists shape: {artists.shape[0]} rows x {artists.shape[1]} columns\n")

    print(f"Tracks sample:\n", tracks.head())
    print(f"Artists sample:\n", artists.head())

    print(f"Tracks types:", tracks.info())
    print(f"Artists types:", artists.info())

    print("\n \n \n")

    # Merging the two datasets
    merged_dataset = tracks.merge(
        artists,
        left_on='id_artist',
        right_on='id_author',
        how='left'
    )

    if "id_author" in merged_dataset.columns:
        merged_dataset = merged_dataset.drop(columns=["id_author"])

    merged_dataset = merged_dataset.dropna()
    print(f"Original merged dataset shape: {merged_dataset.shape[0]} rows x {merged_dataset.shape[1]} columns")
    print(f"Columns: {merged_dataset.columns.tolist()}")

    print("\n \n \n")

    # Original heatmaps
    #og_tracks_heatmap(tracks)

    #og_artists_heatmap(artists)

    og_full_heatmap(merged_dataset)

    # "loudness", "rms" = 1.00
    merged_dataset = merged_dataset.drop(columns=["loudness"])

    # "zcr" high corr w/ "rolloff"(0.97), "centroid"(0.87)
    merged_dataset = merged_dataset.drop(columns=["zcr"])

    # Tracks features
    merged_dataset = swear_ratio(merged_dataset)
    merged_dataset = syntactic_complexity(merged_dataset)
    merged_dataset = flow_complexity(merged_dataset)
    merged_dataset = multilingual_index(merged_dataset)
    merged_dataset = percussiveness(merged_dataset)
    merged_dataset = timbre_brightness(merged_dataset)
    merged_dataset = harmonic_richness(merged_dataset)
    merged_dataset = trap_index(merged_dataset)
    merged_dataset = boombap_index(merged_dataset)
    merged_dataset = cloud_rap_index(merged_dataset)
    merged_dataset = drill_index(merged_dataset)
    merged_dataset = career_longevity(merged_dataset)
    
    print(f"Merged dataset shape: {merged_dataset.shape[0]} rows x {merged_dataset.shape[1]} columns")
    print(f"Columns: {merged_dataset.columns.tolist()}")
    
    # Enriched dataset
    #full_heatmap(merged_dataset)

    # Removing redundant columns
    merged_dataset = merged_dataset.drop(columns=[
        "id_artist",
        "explicit",
        "language",
        "swear_IT",
        "swear_EN",
        "n_sentences",
        "n_tokens",
        "tokens_per_sent",
        "char_per_tok",
        "lexical_density",
        #"avg_token_per_clause",
        "bpm",
        "centroid",
        "rolloff",
        "flux",
        "rms",
        "flatness",
        "spectral_complexity",
        "pitch",
        "disc_number",
        "track_number",
        #"duration_ms",
        #"popularity",
        #"album_release_year",
        #"latitude",
        "longitude",
        "active_start_year",
        #"swear_ratio",
        #"syntactic_complexity",
        #"flow_complexity",
        #"multilingual_index",
        #"percussiveness",
        "timbre_brightness",
        #"harmonic_richness",
        #"trap_index",
        #"boombap_index",
        #"cloud_rap_index",
        #"drill_index",
        #"career_longevity"
    ])

    # Final dataset characterization
    print("\n \n \n")
    print(f"Final merged dataset shape: {merged_dataset.shape[0]} rows x {merged_dataset.shape[1]} columns")
    print(f"Columns: {merged_dataset.columns.tolist()}")

    # Final dataset heatmaps
    full_heatmap(merged_dataset)

    merged_dataset.to_csv('../clustering_dataset/merged_dataset.csv', index=False)

    
    
