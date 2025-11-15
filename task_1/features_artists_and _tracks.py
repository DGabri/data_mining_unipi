import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAZIONE E CARICAMENTO DATI ---
print("--- Caricamento Dataset ---")
try:
    tracks = pd.read_csv("../prepared_datasets/tracks.csv", sep=',')
    artists = pd.read_csv("../prepared_datasets/artists.csv", sep=',')
    print("Dataset caricati con successo.")
except FileNotFoundError as e:
    print(f"Errore: File non trovato. Verifica il percorso. {e}")
    exit()

# Impostiamo uno stile globale per i grafici
sns.set_style("whitegrid")


# --- 2. DATA CLEANING & CORREZIONI (LOGICA "EXPLICIT") ---
print("\n--- Esecuzione Data Cleaning (Fix Explicit) ---")

# Conversione colonne numeriche
tracks['swear_IT'] = pd.to_numeric(tracks['swear_IT'], errors='coerce').fillna(0)
tracks['swear_EN'] = pd.to_numeric(tracks['swear_EN'], errors='coerce').fillna(0)

count_explicit_before = tracks['explicit'].sum()

# Maschera: Ha parolacce MA explicit è False
mask_to_fix = ((tracks['swear_IT'] > 0) | (tracks['swear_EN'] > 0)) & (tracks['explicit'] == False)
count_fixed = mask_to_fix.sum()

# Applicazione correzione
tracks.loc[mask_to_fix, 'explicit'] = True

print(f"Explicit prima: {count_explicit_before}")
print(f"Righe corrette: {count_fixed}")
print(f"Explicit dopo: {tracks['explicit'].sum()}")


# --- 3. PREPARAZIONE DATI AGGIUNTIVI ---

# A. Merge Tracks + Artists
full_data = pd.merge(
    tracks, 
    artists, 
    left_on='id_artist', 
    right_on='id_author', 
    how='inner'
)

# B. Preparazione Geo-Dati (Nord vs Sud)
geo_artists = artists.dropna(subset=['latitude', 'longitude']).copy()
LATITUDE_THRESHOLD = 41.9
geo_artists['zone'] = geo_artists['latitude'].apply(lambda x: 'North' if x >= LATITUDE_THRESHOLD else 'South')

# C. Preparazione Featuring (Explode)
feat_df = tracks[tracks['featured_artists'].notna() & (tracks['featured_artists'] != '')].copy()
feat_df['artists_list'] = feat_df['featured_artists'].astype(str).str.split(',')
exploded_feat = feat_df.explode('artists_list')
exploded_feat['artists_list'] = exploded_feat['artists_list'].str.strip()


# --- 4. VISUALIZZAZIONE ---

print("\n--- Generazione Grafici ---")

# === GRAFICO 1: Top 30 Artisti per Numero di Brani Pubblicati ===
top_publishers = tracks['name_artist'].value_counts().head(30)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_publishers.values, y=top_publishers.index, palette='magma')
plt.title('Top 30 Artists by Number of Published Tracks')
plt.xlabel('Number of Published Tracks')
plt.ylabel('Artist')
for index, value in enumerate(top_publishers.values):
    plt.text(value + 0.5, index, str(value), va='center', fontsize=9)
plt.tight_layout()
plt.show()


# === GRAFICO 2: Top 30 Artisti per Brani Explicit (Dati Corretti) ===
# Nota: Usiamo 'name' se è il nome dell'artista in full_data, altrimenti 'name_artist' da tracks
# Assumo che nel tuo codice originale 'name' dopo il merge si riferisse all'artista.
if 'name_artist' in full_data.columns:
    group_col = 'name_artist'
else:
    group_col = 'name' # Fallback al tuo codice originale

explicit_absolute = full_data.groupby(group_col)['explicit'].agg(['sum', 'count']).reset_index()
explicit_absolute.columns = ['Artist', 'Explicit_Tracks_Count', 'Total_Tracks']
top_absolute_explicit = explicit_absolute.sort_values(by='Explicit_Tracks_Count', ascending=False).head(30)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_absolute_explicit, x='Explicit_Tracks_Count', y='Artist', palette='Reds_r')
plt.title('Top 30 Artists by Total Number of Explicit Songs (Corrected Data)')
plt.xlabel('Number of Explicit Songs')
plt.ylabel('Artist')
for index, value in enumerate(top_absolute_explicit['Explicit_Tracks_Count']):
    plt.text(value, index, str(value), va='center')
plt.tight_layout()
plt.show()


# === GRAFICO 3: Artisti con più Featuring ===
top_feat_artists = exploded_feat['artists_list'].value_counts().head(30)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_feat_artists.values, y=top_feat_artists.index, palette='viridis')
plt.title('Artists with Most Featuring Appearances')
plt.xlabel('Number of Songs Featuring the Artist')
plt.ylabel('Artist')
plt.tight_layout()
plt.show()


# === GRAFICO 4: Distribuzione Geografica (Nord vs Sud) ===
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scatterplot
sns.scatterplot(
    data=geo_artists, 
    x='longitude', y='latitude', 
    hue='zone', 
    palette={'North': '#1f77b4', 'South': '#d62728'}, 
    alpha=0.6, ax=axes[0]
)
axes[0].set_title('Geographical Distribution')
axes[0].axhline(y=LATITUDE_THRESHOLD, color='green', linestyle='--', label='Threshold (Rome)')
axes[0].legend()

# Barplot Conteggio
zone_counts = geo_artists['zone'].value_counts().reset_index()
zone_counts.columns = ['zone', 'count']
sns.barplot(data=zone_counts, x='zone', y='count', palette={'North': '#1f77b4', 'South': '#d62728'}, ax=axes[1])
axes[1].set_title('Counting Artists by Zone')

plt.tight_layout()
plt.show()

print("Analisi completata.")