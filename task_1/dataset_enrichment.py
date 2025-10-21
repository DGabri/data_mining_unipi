import pandas as pd
import requests
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os


"""
    API config
"""
load_dotenv(dotenv_path="../API_keys/spotify_key.env")

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise SystemExit("❌ERROR: set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in the .env file")

INPUT_CSV = "../original_datasets/tracks.csv"
OUTPUT_CSV = "../enriched_datasets/tracks_enriched.csv"



""" 
    Gets an OAuth2 token of the Spotify API
"""
def get_spotify_token(client_id, client_secret):
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    r = requests.post(url, headers=headers, data=data, auth=(client_id, client_secret))
    r.raise_for_status()
    return r.json()["access_token"]

"""
    Searches a song on Spotify and returns:
    - release date
    - popularity -> range [0, 100]
"""
def search_track(title, artist, token):
    query = f"track:{title} artist:{artist}"
    url = f"https://api.spotify.com/v1/search?q={requests.utils.quote(query)}&type=track&limit=1"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None
    items = r.json().get("tracks", {}).get("items", [])
    if not items:
        return None
    track = items[0]
    return {
        "album_release_date": track["album"].get("release_date"),
        "popularity": track.get("popularity")
    }


def enrich_dataset(input_csv, output_csv, client_id, client_secret):
    df = pd.read_csv(input_csv)
    token = get_spotify_token(client_id, client_secret)

    # Creates new colums if missing
    for col in ["album_release_date", "popularity"]:
        if col not in df.columns:
            df[col] = None

    print(f"🔍 Enrichment of {len(df)} tracks loading...\n")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row.get("title", "")).strip()
        artist = str(row.get("primary_artist", "")).strip()
        if not title or not artist:
            continue

        result = search_track(title, artist, token)
        if result:
            df.loc[idx, "album_release_date"] = result["album_release_date"]
            df.loc[idx, "popularity"] = result["popularity"]

        time.sleep(0.2)  # API rate limit: 10 rps

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Enrichment file saved as: {output_csv}")

if __name__ == "__main__":
    enrich_dataset(INPUT_CSV, OUTPUT_CSV, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)


