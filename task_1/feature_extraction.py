import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



"""
    Opening .csv file using Pandas df.
"""
tracks = pd.read_csv("../enriched_datasets/tracks_enriched.csv")
artists = pd.read_csv("../enriched_datasets/artists.csv")



"""
    DATA PREPARATION
    Uncorrect value type (temporary) management.
    - Finds numeric columns in both datasets
    - Forces popularity as a numeric value
    - Forces year ad a numeric value
    - Drops non-number values
    - Drops active_end column
    - Takes into account only the active_start year
    TODO Understand why popularity is not recognised as numeric.
"""
def data_filling(tracks, artists):
    tracks = tracks.copy()
    artists = artists.copy()

    # Tracks
    tracks["popularity"] = pd.to_numeric(tracks["popularity"], errors='coerce')

    tracks["year"] = pd.to_numeric(tracks["year"], errors='coerce')

    numeric_cols_t = tracks.select_dtypes(include=["number"]).columns
    
    for col in numeric_cols_t:
        tracks[col] = pd.to_numeric(tracks[col], errors='coerce')
    
    numeric_tracks = tracks[numeric_cols_t]
    
    # Artists
    if "active_end" in artists.columns:
        artists = artists.drop(columns=["active_end"])

    if "active_start" in artists.columns:
        artists["active_start_year"] = pd.to_datetime(artists["active_start"], errors="coerce").dt.year

    numeric_cols_a = artists.select_dtypes(include=["number"]).columns

    for col in numeric_cols_a:
        artists[col] = pd.to_numeric(artists[col], errors='coerce')
    
    numeric_artists = artists[numeric_cols_a]

    # Verifies status
    print(f"Tracks shape: {numeric_tracks.shape[0]} rows x {numeric_tracks.shape[1]} columns")
    print(f"Artists shape: {numeric_artists.shape[0]} rows x {numeric_artists.shape[1]} columns\n")

    print(f"Numeric tracks sample:\n", numeric_tracks.head())
    print(f"Numeric artists sample:\n", numeric_artists.head())
    return numeric_tracks, numeric_artists



"""
    Reintroducing id and artist_id inside the modified datasets
"""
def datasets_completion(tracks, numeric_tracks, artists, numeric_artists):
    # Tracks
    if "id" in tracks.columns:
        numeric_tracks.insert(0, "id", tracks["id"])

    if "id_artist" in tracks.columns:
        numeric_tracks.insert(1, "id_artist", tracks["id_artist"])

    # Artists
    if "id_author" in artists.columns:
        numeric_artists.insert(0, "id_author", artists["id_author"])

    # Verifies status
    print(f"Numeric tracks sample:\n", numeric_tracks.head())
    print(f"Numeric artists sample:\n", numeric_artists.head(), "\n")
    return numeric_tracks, numeric_artists



"""
    Plotting a correlation heatmap over the numeric values for:
    1. tracks
    2. artist
    3. both
"""
def og_tracks_heatmap(numeric_tracks):
    numeric_cols = numeric_tracks.select_dtypes(include=[np.number]).columns
    tracks_for_corr = numeric_tracks[numeric_cols]

    og_tracks_corr = tracks_for_corr.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(og_tracks_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations - original tracks features")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    return

def og_artists_heatmap(numeric_artists):
    numeric_cols = numeric_artists.select_dtypes(include=[np.number]).columns
    artists_for_corr = numeric_artists[numeric_cols]

    og_artists_corr = artists_for_corr.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(og_artists_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations - original artists features")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()
    return

def og_full_heatmap(numeric_tracks, numeric_artists):
    # Merging the two datasets
    og_dataset = numeric_tracks.merge(
        numeric_artists,
        left_on='id_artist',
        right_on='id_author',
        how='left'
    )
    
    if 'id_author' in og_dataset.columns:
        og_dataset = og_dataset.drop(columns=['id_author'])
    
    # Selecting only numeric columns
    numeric_cols = og_dataset.select_dtypes(include=[np.number]).columns
    og_for_corr = og_dataset[numeric_cols]

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
    TODO
    DATA PREPARATION (still)
    - understand the correct values of each og feature and handle errors
"""



"""
    Defining constant for the denominator
"""
eps = 1e-6



"""
    LANGUAGE FEATURES
"""
def swear_density(numeric_tracks):
    df = numeric_tracks.copy()

    swear_ratio = (df["swear_IT"] + df["swear_EN"])/df["n_tokens"]

    # Percentage
    df["swear_density"] = swear_ratio * 100

    # TODO If it is too low try logarithm
    return df

def syntactic_complexity(numeric_tracks):
    df = numeric_tracks.copy()

    # Verbosity
    verbosity = df["n_tokens"] / df["n_tokens"].max()

    # Vocabulary complexity
    voc_complexity = df["lexical_density"] * (df["char_per_tok"]/df["char_per_tok"].max())

    df["syntactic_complexity"] = (verbosity + voc_complexity)/2
    return df

def flow_complexity(numeric_tracks):
    df = numeric_tracks.copy()
    # TODO Check math

    # Tokens per sentence normalized
    tps_norm = df["tokens_per_sent"] / df["tokens_per_sent"].max()
    
    # Flux normalized
    flux_norm = df["flux"] / (df["flux"].max() + eps)
    
    df["flow_complexity"] = (tps_norm + flux_norm) / 2
    return df

def lyrical_density(numeric_tracks):
    df = numeric_tracks.copy()

    clause_density = df["avg_token_per_clause"] / (df["avg_token_per_clause"].max() + eps)
    
    # Char per token (longer words -> more complex concepts)
    word_complexity = df["char_per_tok"] / (df["char_per_tok"].max() + eps)
    
    df["lyrical_density"] = (clause_density + word_complexity + df["lexical_density"]) / 3
    return df

def multilingual_index(numeric_tracks, tracks):
    df = numeric_tracks.copy()
    
    df["multilingual_index"] = 0

    df.loc[tracks["language"] != "it", "multilingual_index"] = 1
    return df



"""
    SOUND FEATURES
    TODO Check if 'pitch' actually needs log
"""
# TODO Keep because of high correlation
def percussivness(numeric_tracks):
    df = numeric_tracks.copy()

    # Normalized zcr
    zcr_norm = df["zcr"] / (df["zcr"].max() + eps)

    # Normalized rolloff
    rolloff_norm = df["rolloff"] / (df["rolloff"].max() + eps)

    # Percussivness
    df["percussivness"] = zcr_norm * rolloff_norm
    return df

def timbre_brightness(numeric_tracks):
    df = numeric_tracks.copy()

    # Normalized centroid
    centroid_norm = df["centroid"] / (df["centroid"].max() + eps)

    # Normalized rolloff
    rolloff_norm = df["rolloff"] / (df["rolloff"].max() + eps)

    # TODO Could be a weighted sum
    df["timbre_brightness"] = centroid_norm + rolloff_norm / 2
    return df

def energy_index(numeric_tracks):
    df = numeric_tracks.copy()

    # Normalized loudness
    loudness_norm = (df["loudness"] - df["loudness"].min()) / (df["loudness"].max() - df["loudness"].min() + eps)
    
    # Normalized flux
    flux_norm = df["flux"] / (df["flux"].max() + eps)
    
    # Normalized zcr
    zcr_norm = df["zcr"] / (df["zcr"].max() + eps)

    # Energy index
    df["energy_index"] = (loudness_norm + flux_norm + zcr_norm) / 3
    return df

def harmonic_richness(numeric_tracks):
    df = numeric_tracks.copy()

    # Tonality
    tonality = 1 - df["flatness"]
    
    # Complexity
    complexity_norm = df["spectral_complexity"] / (df["spectral_complexity"].max() + eps)
    
    # Pitch stability
    pitch_norm = np.log1p(df["pitch"]) / (np.log1p(df["pitch"].max()) + eps)
    
    df["harmonic_richness"] = (tonality * complexity_norm * pitch_norm) ** (1/3)
    return df

def trap_index(numeric_tracks):
    df = numeric_tracks.copy()

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

def boombap_index(numeric_tracks):
    df = numeric_tracks.copy()

    """
        Identifies boom-bap/old school:
        - bpm -> 85-95
        - high flux
        - Complexity media-alta (samples)
    """
    # bpm score
    bpm_bb = np.where((df["bpm"] >= 85) & (df["bpm"] <= 95), 1.0, 0.3)
    
    # flux
    flux_norm = df["flux"] / (df["flux"].max() + eps)
    
    # Complexity
    complexity_norm = df["spectral_complexity"] / (df["spectral_complexity"].max() + eps)
    
    df["boombap_index"] = (bpm_bb + flux_norm + complexity_norm) / 3
    return df

def cloud_rap_index(numeric_tracks):
    df = numeric_tracks.copy()

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

def drill_index(numeric_tracks):
    df = numeric_tracks.copy()

    """
    Identifies UK Drill style:
    - bpm -> 140-150
    - low rolloff (dark sound)
    - Sliding bass
    """
    # BPM drill
    bpm_drill = np.where((df["bpm"] >= 138) & (df["bpm"] <= 152), 1.0, 0.3)
    
    # Darkness
    rolloff_norm = df["rolloff"] / (df["rolloff"].max() + 1e-6)
    darkness = 1 - rolloff_norm
    
    # Sliding bass (pitch + flatness combination)
    sliding_score = (np.log1p(df["pitch"]) / (np.log1p(df["pitch"].max()) + 1e-6) + 
                    df["flatness"]) / 2
    
    df["drill_index"] = (bpm_drill + darkness + sliding_score) / 3
    return df



"""
    POPULARITY FEATURES
"""
def production_modernity(numeric_tracks):
    df = numeric_tracks.copy()

    # Normalized year of production
    year_norm = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())
    
    #TODO Check this
    #year_norm = np.clip(year_norm, 0, 1)
    
    # Normalized loudness (more recent = louder)
    loudness_norm = (df["loudness"] - df["loudness"].min()) / (df["loudness"].max() - df["loudness"].min() + eps)
    
    # Brightness increasing with modernity
    brightness = df["timbre_brightness"]
    
    df["production_modernity"] = (year_norm + loudness_norm + brightness) / 3
    return df

def career_maturity(numeric_tracks):
    df = numeric_tracks.copy()

    df["career_maturity"] = df["disc_number"] / (df["disc_number"].max() + eps)
    return df

def career_longevity(numeric_artists):
    df = numeric_artists.copy()

    current_year = datetime.now().year
    years_active = current_year - df["active_start_year"]

    df["career_longevity"] = years_active / (years_active.max() + eps)
    return df

def geographic_influence(numeric_artists):
    df = numeric_artists.copy()

    # Uses min latitude as the origin
    lat_norm = (df["latitude"] - df["latitude"].min()) / \
                  (df["latitude"].max() - df["latitude"].min() + 1e-6)
    
    df["geographic_north_south"] = lat_norm
    return df

def summer_hit(numeric_tracks):
    df = numeric_tracks.copy()

    # TODO can be simplified

    # Save convertion to integers
    # TODO Drop nans
    df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(-1).astype(int)

    # Vectorialized calculus
    conditions = [
        df["month"].isin([6, 7, 8]),   # summer
        df["month"].isin([4, 5])       # spring
    ]
    values = [1.0, 0.5]

    df["summer_hit_index"] = np.select(conditions, values, default=0.0)
    return df

def gender_encoding(numeric_artists, artists):
    df = numeric_artists.copy()
    
    # TODO Drop nans
    gender_map = {'M': 0, 'F': 1}
    df["gender_numeric"] = artists["gender"].map(gender_map).fillna(0.5)
    return df



"""
    Final correlation matrices:
    1. new tracks features
    2. new artists features
    3. all features
"""
def enriched_tracks_heatmap(numeric_tracks):
    numeric_cols = numeric_tracks.select_dtypes(include=[np.number]).columns
    tracks_for_corr = numeric_tracks[numeric_cols]
    
    enriched_tracks_corr = tracks_for_corr.corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(enriched_tracks_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations - enriched tracks features")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.show()
    return

def enriched_artists_heatmap(numeric_artists):
    numeric_cols = numeric_artists.select_dtypes(include=[np.number]).columns
    artists_for_corr = numeric_artists[numeric_cols]
    
    enriched_artists_corr = artists_for_corr.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(enriched_artists_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Heatmap of correlations - enriched artists features")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()
    return

def enriched_full_heatmap(numeric_tracks, numeric_artists):
    # Merging the two datasets
    final_dataset = numeric_tracks.merge(
        numeric_artists,
        left_on='id_artist',
        right_on='id_author',
        how='left'
    )
    
    if 'id_author' in final_dataset.columns:
        final_dataset = final_dataset.drop(columns=['id_author'])
    
    # Selecting only numeric columns
    numeric_cols = final_dataset.select_dtypes(include=[np.number]).columns
    final_for_corr = final_dataset[numeric_cols]
    
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



"""
    TODO Dropping redundant/useless features
"""



if __name__ == "__main__":
    numeric_tracks, numeric_artists = data_filling(tracks, artists)

    # Adding ids
    numeric_tracks, numeric_artists = datasets_completion(tracks, numeric_tracks, artists, numeric_artists)

    # Original heatmaps
    og_tracks_heatmap(numeric_tracks)

    og_artists_heatmap(numeric_artists)

    og_full_heatmap(numeric_tracks, numeric_artists)

    # Tracks features
    numeric_tracks = swear_density(numeric_tracks)
    numeric_tracks = syntactic_complexity(numeric_tracks)
    numeric_tracks = flow_complexity(numeric_tracks)
    numeric_tracks = lyrical_density(numeric_tracks)
    numeric_tracks = multilingual_index(numeric_tracks, tracks)
    numeric_tracks = percussivness(numeric_tracks)
    numeric_tracks = timbre_brightness(numeric_tracks)
    numeric_tracks = energy_index(numeric_tracks)
    numeric_tracks = harmonic_richness(numeric_tracks)
    numeric_tracks = trap_index(numeric_tracks)
    numeric_tracks = boombap_index(numeric_tracks)
    numeric_tracks = cloud_rap_index(numeric_tracks)
    numeric_tracks = drill_index(numeric_tracks)
    numeric_tracks = production_modernity(numeric_tracks)
    numeric_tracks = career_maturity(numeric_tracks)
    numeric_tracks = summer_hit(numeric_tracks)
    
    print(f"\nFinal numeric_tracks")
    print(f"Shape: {numeric_tracks.shape}")
    print(f"Columns: {numeric_tracks.columns.tolist()}")
    print(numeric_tracks.head())
    
    # Artist features
    numeric_artists = career_longevity(numeric_artists)    
    numeric_artists = geographic_influence(numeric_artists)
    numeric_artists = gender_encoding(numeric_artists, artists)
    
    print(f"\nFinal numeric_artists")
    print(f"Shape: {numeric_artists.shape}")
    print(f"Columns: {numeric_artists.columns.tolist()}")
    print(numeric_artists.head())

    # Enriched dataset heatmaps
    enriched_tracks_heatmap(numeric_tracks)
    enriched_artists_heatmap(numeric_artists)
    enriched_full_heatmap(numeric_tracks, numeric_artists)
