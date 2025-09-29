import os
import gspread
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from oauth2client.service_account import ServiceAccountCredentials
import json
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# --- Configuration ---
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
GOOGLE_SHEET_NAME = 'Sheet1'
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = 'call-insights'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- Initialize Clients ---
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Successfully connected to OpenAI and Pinecone.")
except Exception as e:
    print(f"ERROR: Could not initialize clients: {e}")
    exit(1)


def get_data_from_sheet():
    """Fetches data from Google Sheets and returns a pandas DataFrame."""
    print("Connecting to Google Sheets...")
    try:
        creds_json = json.loads(GOOGLE_CREDENTIALS_JSON)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        records = sheet.get_all_records()
        print(f"Fetched {len(records)} records from the sheet.")
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching from Google Sheets: {e}")
        return pd.DataFrame()


def find_new_rows_to_index(df):
    """Compares sheet data with Pinecone index to find new rows."""
    if df.empty or 'call_id' not in df.columns:
        return pd.DataFrame()

    all_sheet_ids = [str(id) for id in df['call_id'].unique()]
    
    print(f"Checking {len(all_sheet_ids)} IDs from the sheet against the Pinecone index...")
    
    # Fetch existing vectors from Pinecone in batches to check which IDs are already indexed
    existing_ids = set()
    for i in range(0, len(all_sheet_ids), 100):
        batch_ids = all_sheet_ids[i:i+100]
        fetch_response = index.fetch(ids=batch_ids)
        
        # --- FIX: Access the 'vectors' attribute directly ---
        existing_ids.update(fetch_response.vectors.keys())
        # ---------------------------------------------------

    new_ids = set(all_sheet_ids) - existing_ids
    
    print(f"Found {len(new_ids)} new rows to index.")
    
    # Filter the dataframe to only include the new rows
    return df[df['call_id'].astype(str).isin(new_ids)]


def create_embeddings_and_upsert(df):
    """Creates embeddings for new rows and upserts them into Pinecone."""
    if df.empty:
        print("No new data to index.")
        return

    print(f"Creating embeddings for {len(df)} new rows...")
    EMBEDDING_MODEL = "text-embedding-3-small"

    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        texts_to_embed = (
            "Call on " + batch['date'].astype(str) +
            " regarding '" + batch['primary_issue'].astype(str) +
            "'. Summary: " + batch['summary'].astype(str) +
            ". Action items: " + batch['action_items'].astype(str)
        ).tolist()

        response = openai_client.embeddings.create(input=texts_to_embed, model=EMBEDDING_MODEL)
        embeddings = [item.embedding for item in response.data]

        vectors_to_upsert = []
        for idx, row in batch.iterrows():
            vector = {
                "id": str(row['call_id']),
                "values": embeddings[batch.index.get_loc(idx)],
                "metadata": {
                    "summary": str(row['summary']),
                    "date": str(row['date']),
                    "sentiment": str(row['sentiment']),
                    "transcript_snippet": str(row.get('transcript_snippet', ''))
                }
            }
            vectors_to_upsert.append(vector)
        
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch {i//batch_size + 1}")

    print("Indexing complete!")


if __name__ == "__main__":
    full_dataframe = get_data_from_sheet()
    if not full_dataframe.empty:
        new_data_df = find_new_rows_to_index(full_dataframe)
        create_embeddings_and_upsert(new_data_df)

