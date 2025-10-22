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
    - year of release
    - month of release
    - day of release
"""
def search_track(title, artist, token):
    query = f"track:{title} artist:{artist}"
    url = f"https://api.spotify.com/v1/search?q={requests.utils.quote(query)}&type=track&limit=1"
    headers = {"Authorization": f"Bearer {token}"}
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 401:  # Token expired
                return "token_expired"
            if r.status_code != 200:
                print(f"\nWarning: API returned status {r.status_code} for track '{title}' by {artist}")
                return None
            items = r.json().get("tracks", {}).get("items", [])
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"\nError: Failed to fetch data for track '{title}' by {artist}: {str(e)}")
                return None
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    if not items:
        return None
    
    # Get album release date from the first matching track
    release_date = items[0].get("album", {}).get("release_date")
    if not release_date:
        return None
    
    # Parse the date parts - Spotify can return different date formats (YYYY-MM-DD, YYYY-MM, or YYYY)
    date_parts = release_date.split("-")
    year = int(date_parts[0]) if len(date_parts) > 0 else None
    month = int(date_parts[1]) if len(date_parts) > 1 else None
    day = int(date_parts[2]) if len(date_parts) > 2 else None
    
    return {
        "year": year,
        "month": month,
        "day": day
    }


def enrich_dataset(input_csv, output_csv, client_id, client_secret):
    df = pd.read_csv(input_csv)
    token = get_spotify_token(client_id, client_secret)
    
    print("🔍 Searching for track release dates...")
    progress_bar = tqdm(df.index, total=len(df))
    for idx in progress_bar:
        title = df.at[idx, 'title']
        artist = df.at[idx, 'name_artist']
        
        result = search_track(title, artist, token)
        if result == "token_expired":
            print("\nRefreshing Spotify API token...")
            token = get_spotify_token(client_id, client_secret)
            result = search_track(title, artist, token)
        
        if isinstance(result, dict):
            df.at[idx, 'year'] = result['year']
            df.at[idx, 'month'] = result['month']
            df.at[idx, 'day'] = result['day']
        
        progress_bar.set_description(f"Processing {title[:30]}...")
        time.sleep(0.2)  # API rate limit: 10 rps

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Enrichment file saved as: {output_csv}")

if __name__ == "__main__":
    enrich_dataset(INPUT_CSV, OUTPUT_CSV, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)


