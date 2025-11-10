import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# --- Load datasets ---
artists = pd.read_csv("../enriched_datasets/artists.csv")
tracks = pd.read_csv("../enriched_datasets/tracks_enriched.csv")

# --- Ensure popularity is numeric ---
tracks['popularity'] = pd.to_numeric(tracks['popularity'], errors='coerce').fillna(0)

# --- Function to calculate and plot top N artists ---
def plot_top_artists(tracks_df, artists_df, top_n=40, title="Top Artists by Average Popularity"):
    """
    Calculate average popularity per artist and plot top N artists.
    
    Parameters:
    - tracks_df: DataFrame with tracks info (must contain 'id_artist' and 'popularity')
    - artists_df: DataFrame with artists info (must contain 'id_author' and 'name')
    - top_n: number of top artists to display
    - title: plot title
    """
    # Aggregate average popularity per artist
    artist_popularity = (
        tracks_df.groupby('id_artist')['popularity']
        .mean()
        .reset_index()
    )
    
    # Merge with artist metadata
    artists_merged = artists_df.merge(
        artist_popularity,
        left_on='id_author',
        right_on='id_artist',
        how='left'
    )
    
    # Sort by popularity
    top_artists = artists_merged.sort_values(by='popularity', ascending=False).head(top_n)
    
    # Plot horizontal bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='popularity',
        y='name',
        data=top_artists[::-1],  # reverse for top artist on top
        palette='viridis'
    )
    plt.title(title)
    plt.xlabel('Average Popularity')
    plt.ylabel('Artist')
    plt.tight_layout()
    plt.show()
    
    return top_artists

# --- Step 1: Plot raw popularity ---
print("📊 Plotting raw popularity (before clipping)")
top_artists_raw = plot_top_artists(tracks, artists, top_n=40, title="Top 40 Italian Artists by Raw Popularity")

# --- Step 2: Clip popularity to [0,100] for standardized data ---
tracks_standardized = tracks.copy()
tracks_standardized['popularity'] = tracks_standardized['popularity'].clip(0, 100)

# --- Step 3: Plot cleaned/standardized popularity ---
print("📊 Plotting cleaned popularity (after clipping 0–100)")
top_artists_cleaned = plot_top_artists(
    tracks_standardized,
    artists,
    top_n=40,
    title="Top 40 Italian Artists by Cleaned Popularity"
)

# --- Optional: save the standardized dataset ---
tracks_standardized.to_csv("../enriched_datasets/tracks_enriched.csv", index=False)
print("✅ Standardized tracks dataset saved as 'tracks_enriched.csv'")
