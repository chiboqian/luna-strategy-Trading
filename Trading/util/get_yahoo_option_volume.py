#!/usr/bin/env python3
import pandas as pd
import requests
from io import StringIO
import time
import argparse

def download_yahoo_options(min_volume=20000):
    base_url = "https://query2.finance.yahoo.com/v7/finance/options/rankings"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    
    print("Fetching data from Yahoo Finance...")
    try:
        session = requests.Session()
        session.headers.update(headers)
        
        all_dfs = []
        offset = 0
        batch_size = 100  # Try to fetch 100 at a time
        max_records = 500 # Increased limit to ensure we catch enough volume candidates
        
        while offset < max_records:
            params = {"rankBy": "VOLUME", "sortOrder": "DESC", "offset": offset, "count": batch_size}
            print(f"Requesting items starting at {offset}...")
            
            # Retry logic for 429 errors
            for attempt in range(3):
                response = session.get(base_url, params=params)
                if response.status_code == 429:
                    wait = (attempt + 1) * 5
                    print(f"Rate limit hit (429). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                break
            
            print(f"Actual URL: {response.url}")
            response.raise_for_status()
            
            try:
                data = response.json()
                results = data.get("optionRankings", {}).get("result", [])
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                break
            
            if not results:
                break
                
            df = pd.DataFrame(results)
            if df.empty:
                break
                
            # Check for duplicates (using 'symbol' column from JSON)
            check_col = 'symbol' if 'symbol' in df.columns else df.columns[0]
            if all_dfs and not all_dfs[-1].empty and df.iloc[0][check_col] == all_dfs[-1].iloc[0][check_col]:
                print("Duplicate data detected (pagination limit reached).")
                break
                
            print(f"Found {len(df)} rows.")
            all_dfs.append(df)
            
            # If we got fewer than a standard small page (25), we are likely done
            if len(df) < 25:
                break
                
            offset += len(df)
            
            for i in range(30, 0, -1):
                print(f"\rFetched {offset} items. Waiting {i}s...", end="", flush=True)
                time.sleep(1)
            print()
        
        if not all_dfs:
            print("No data fetched.")
            return
            
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Deduplicate just in case
        dedup_col = 'symbol' if 'symbol' in final_df.columns else final_df.columns[0]
        final_df.drop_duplicates(subset=[dedup_col], inplace=True)
        
        # Filter by Volume
        if 'volume' in final_df.columns:
            # Ensure volume is numeric
            final_df['volume'] = pd.to_numeric(final_df['volume'], errors='coerce').fillna(0)
            original_count = len(final_df)
            final_df = final_df[final_df['volume'] >= min_volume]
            print(f"Filtered by Volume >= {min_volume}: {len(final_df)} / {original_count} items remaining.")
        
        # SAVE TO CSV
        filename = "yahoo_options_data.csv"
        final_df.to_csv(filename, index=False)
        print(f"Success! Data saved to {filename}")
        return final_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Most Active Options from Yahoo and optionally run Scanner")
    parser.add_argument("--min-volume", type=int, default=20000, help="Minimum volume to filter (default: 20000)")
    
    args = parser.parse_args()
    
    download_yahoo_options(min_volume=args.min_volume)