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
    Defining constant for the denominator
"""
eps = 1e-6



"""
    LANGUAGE FEATURES
"""
def swear_density(merged_dataset):
    df = merged_dataset.copy()

    swear_ratio = (df["swear_IT"] + df["swear_EN"])/df["n_tokens"]

    # Percentage
    df["swear_density"] = swear_ratio * 100
    return df

def syntactic_complexity(merged_dataset):
    df = merged_dataset.copy()

    # Verbosity
    verbosity = df["n_tokens"] / df["n_tokens"].max()

    # Vocabulary complexity
    voc_complexity = df["lexical_density"] * (df["char_per_tok"]/df["char_per_tok"].max())

    df["syntactic_complexity"] = (verbosity + voc_complexity)/2
    return df

def flow_complexity(merged_dataset):
    df = merged_dataset.copy()
    # TODO Check math

    # Tokens per sentence normalized
    tps_norm = df["tokens_per_sent"] / df["tokens_per_sent"].max()
    
    # Flux normalized
    flux_norm = df["flux"] / (df["flux"].max() + eps)
    
    df["flow_complexity"] = (tps_norm + flux_norm) / 2
    return df

def lyrical_density(merged_dataset):
    df = merged_dataset.copy()

    clause_density = df["avg_token_per_clause"] / (df["avg_token_per_clause"].max() + eps)
    
    # Char per token (longer words -> more complex concepts)
    word_complexity = df["char_per_tok"] / (df["char_per_tok"].max() + eps)
    
    # TODO Check if lexical_density needs to be norm.
    df["lyrical_density"] = (clause_density + word_complexity + df["lexical_density"]) / 3
    return df

def multilingual_index(merged_dataset):
    df = merged_dataset.copy()
    
    df["multilingual_index"] = 0

    df.loc[merged_dataset["language"] != "it", "multilingual_index"] = 1
    return df



"""
    SOUND FEATURES
    TODO Check if 'pitch' actually needs log
"""
# TODO Keep because of high correlation
def percussiveness(merged_dataset):
    df = merged_dataset.copy()

    # Normalized zcr
    zcr_norm = df["zcr"] / (df["zcr"].max() + eps)

    # Normalized rolloff
    rolloff_norm = df["rolloff"] / (df["rolloff"].max() + eps)

    # Percussiveness
    df["percussiveness"] = zcr_norm * rolloff_norm
    return df

def timbre_brightness(merged_dataset):
    df = merged_dataset.copy()

    # Normalized centroid
    centroid_norm = df["centroid"] / (df["centroid"].max() + eps)

    # Normalized rolloff
    rolloff_norm = df["rolloff"] / (df["rolloff"].max() + eps)

    df["timbre_brightness"] = centroid_norm + rolloff_norm / 2
    return df

def energy_index(merged_dataset):
    df = merged_dataset.copy()

    # Normalized loudness
    loudness_norm = (df["loudness"] - df["loudness"].min()) / (df["loudness"].max() - df["loudness"].min() + eps)
    
    # Normalized flux
    flux_norm = df["flux"] / (df["flux"].max() + eps)
    
    # Normalized zcr
    zcr_norm = df["zcr"] / (df["zcr"].max() + eps)

    # Energy index
    df["energy_index"] = (loudness_norm + flux_norm + zcr_norm) / 3
    return df

def harmonic_richness(merged_dataset):
    df = merged_dataset.copy()

    # Tonality
    tonality = 1 - df["flatness"]
    
    # Complexity
    complexity_norm = df["spectral_complexity"] / (df["spectral_complexity"].max() + eps)
    
    # Pitch stability
    pitch_norm = np.log1p(df["pitch"]) / (np.log1p(df["pitch"].max()) + eps)
    
    df["harmonic_richness"] = (tonality * complexity_norm * pitch_norm) ** (1/3)
    return df

def trap_index(merged_dataset):
    df = merged_dataset.copy()

    """
        Identifies trap characteristics:
        - bpm -> 130-170 (or 65-85 half-time)
        - Sub-bass (low rolloff)
        - high loudness
    """
    # bpm score
    bpm_trap = np.where(
        ((df["bpm"] >= 130) & (df["bpm"] <= 170)) | 
        ((df["bpm"] >= 65) & (df["bpm"] <= 85)),
        1.0, 0.3
    )
    
    # Sub-bass intensity (low rolloff = more basses)
    rolloff_norm = df["rolloff"] / (df["rolloff"].max() + eps)
    subbass_score = 1 - rolloff_norm
    
    # Loudness
    loudness_norm = (df["loudness"] - df["loudness"].min()) / (df["loudness"].max() - df["loudness"].min() + eps)
    
    df["trap_index"] = (bpm_trap + subbass_score + loudness_norm) / 3
    return df

def boombap_index(merged_dataset):
    df = merged_dataset.copy()

    """
        Identifies boom-bap/old school:
        - bpm -> 80-95
        - high flux
        - Complexity media-alta (samples)
    """
    # bpm score
    bpm_bb = np.where((df["bpm"] >= 80) & (df["bpm"] <= 95), 1.0, 0.3)
    
    # flux
    flux_norm = df["flux"] / (df["flux"].max() + eps)
    
    # Complexity
    complexity_norm = df["spectral_complexity"] / (df["spectral_complexity"].max() + eps)
    
    df["boombap_index"] = (bpm_bb + flux_norm + complexity_norm) / 3
    return df

def cloud_rap_index(merged_dataset):
    df = merged_dataset.copy()

    """
        Identifies cloud/emo rap:
        - high pitch (melodic)
        - medium-high centroid
        - medium flatness
    """
    # Pitch presence
    pitch_norm = np.log1p(df["pitch"]) / (np.log1p(df["pitch"].max()) + eps)
    
    # Medium flatness (atmospheric)
    flatness_mid = 1 - np.abs(df["flatness"] - 0.5) * 2
    
    # Medium-high centroid
    centroid_norm = df["centroid"] / (df["centroid"].max() + eps)
    
    df["cloud_rap_index"] = (pitch_norm + flatness_mid + centroid_norm) / 3
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

    rolloff_norm = df["rolloff"] / (df["rolloff"].max() + 1e-6)
    subbass_score = 1 - rolloff_norm
    
    # Sliding bass (pitch + flatness combination)
    sliding_score = (np.log1p(df["pitch"]) / (np.log1p(df["pitch"].max()) + 1e-6) + 
                    df["flatness"]) / 2
    
    df["drill_index"] = (bpm_drill + subbass_score + sliding_score) / 3
    return df



"""
    POPULARITY FEATURES
"""
def production_modernity(merged_dataset):
    df = merged_dataset.copy()

    # Normalized year of production
    year_norm = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())
    
    # Normalized loudness (more recent = louder)
    loudness_norm = (df["loudness"] - df["loudness"].min()) / (df["loudness"].max() - df["loudness"].min() + eps)
    
    # Brightness increasing with modernity
    brightness = df["timbre_brightness"]
    
    df["production_modernity"] = (year_norm + loudness_norm + brightness) / 3
    return df

def career_maturity(merged_dataset):
    df = merged_dataset.copy()

    df["career_maturity"] = df["disc_number"] / (df["disc_number"].max() + eps)
    return df

def career_longevity(merged_dataset):
    df = merged_dataset.copy()

    current_year = datetime.now().year
    years_active = current_year - df["active_start_year"]

    df["career_longevity"] = years_active / (years_active.max() + eps)
    return df

def geographic_influence(merged_dataset):
    df = merged_dataset.copy()

    # Uses min latitude as the origin
    lat_norm = (df["latitude"] - df["latitude"].min()) / \
                  (df["latitude"].max() - df["latitude"].min() + 1e-6)
    
    df["geographic_north_south"] = lat_norm
    return df

"""
def summer_hit(merged_dataset):
    df = merged_dataset.copy()

    # Save convertion to integers
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype(int)

    # Vectorialized calculus
    conditions = [
        df["month"].isin([6, 7, 8]),   # summer
        df["month"].isin([4, 5])       # spring
    ]
    values = [1.0, 0.5]

    df["summer_hit_index"] = np.select(conditions, values, default=0.0)
    return df
"""



"""
    Final correlation matrices:
    1. all new features
    2. after non-useful columns removal
"""
def enriched_heatmap(final_merged_dataset):
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

    print("\n \n \n")

    # Original heatmaps
    #og_tracks_heatmap(tracks)
#
    #og_artists_heatmap(artists)
#
    #og_full_heatmap(merged_dataset)

    # Tracks features
    merged_dataset = swear_density(merged_dataset)
    merged_dataset = syntactic_complexity(merged_dataset)
    merged_dataset = flow_complexity(merged_dataset)
    merged_dataset = lyrical_density(merged_dataset)
    merged_dataset = multilingual_index(merged_dataset)
    merged_dataset = percussiveness(merged_dataset)
    merged_dataset = timbre_brightness(merged_dataset)
    merged_dataset = energy_index(merged_dataset)
    merged_dataset = harmonic_richness(merged_dataset)
    merged_dataset = trap_index(merged_dataset)
    merged_dataset = boombap_index(merged_dataset)
    merged_dataset = cloud_rap_index(merged_dataset)
    merged_dataset = drill_index(merged_dataset)
    merged_dataset = production_modernity(merged_dataset)
    merged_dataset = career_maturity(merged_dataset)
    #merged_dataset = summer_hit(merged_dataset)
    merged_dataset = career_longevity(merged_dataset)    
    merged_dataset = geographic_influence(merged_dataset)
    
    print(f"Merged dataset shape: {merged_dataset.shape[0]} rows x {merged_dataset.shape[1]} columns")
    print(f"Columns: {merged_dataset.columns.tolist()}")
    
    # Enriched dataset heatmaps
    enriched_heatmap(merged_dataset)

    # Dropping columns with high correlation
    merged_dataset = merged_dataset.drop(columns=[
            "zcr",
            "rolloff",
            "",
            "",
            "",
        ])

    enriched_heatmap(merged_dataset)
    
