import os
import json
import gspread
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# Load environment variables from .env file for local execution
load_dotenv()

# --- Configuration ---
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
GOOGLE_SHEET_NAME = 'Sheet1'
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON_CHAT')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = 'call-insights'

# --- Initialize Clients ---
try:
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY_CHAT'))
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Successfully connected to OpenAI and Pinecone.")
except Exception as e:
    print(f"Failed to initialize clients: {e}")
    exit()

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
    """Compares sheet data with Pinecone to find new, un-indexed rows."""
    if 'call_id' not in df.columns:
        print("ERROR: 'call_id' column not found.")
        return pd.DataFrame()

    sheet_ids = set(df['call_id'].astype(str).tolist())
    
    # Check which IDs already exist in Pinecone in batches
    existing_ids = set()
    ids_to_check = list(sheet_ids)
    for i in range(0, len(ids_to_check), 100):
        batch_ids = ids_to_check[i:i+100]
        # The 'fetch' operation is a fast way to check for existing IDs
        fetch_response = index.fetch(ids=batch_ids)
        existing_ids.update(fetch_response.get('vectors', {}).keys())
        
    new_ids = sheet_ids - existing_ids
    
    if not new_ids:
        print("No new rows to index. Pinecone is up to date.")
        return pd.DataFrame()
        
    print(f"Found {len(new_ids)} new rows to index.")
    # Filter the dataframe to only include the new rows
    return df[df['call_id'].astype(str).isin(new_ids)]


def create_embeddings_and_upsert(df_new):
    """Creates embeddings for new rows and upserts them into Pinecone."""
    print(f"Creating embeddings for {len(df_new)} new records...")
    embedding_model = "text-embedding-3-small"

    batch_size = 100
    for i in range(0, len(df_new), batch_size):
        batch_df = df_new.iloc[i:i+batch_size]
        
        texts_to_embed = (
            "Call regarding issue: '" + batch_df['primary_issue'].astype(str) +
            "'. Summary of the call: " + batch_df['summary'].astype(str) +
            ". Key topics discussed were: " + batch_df['key_topics'].astype(str) +
            ". The sentiment of the call was " + batch_df['sentiment'].astype(str) + "."
        ).tolist()

        try:
            response = openai_client.embeddings.create(input=texts_to_embed, model=embedding_model)
            embeddings = [item.embedding for item in response.data]

            vectors_to_upsert = []
            for j, row in batch_df.iterrows():
                vector = {
                    "id": str(row['call_id']),
                    "values": embeddings[j - i],
                    "metadata": {
                        "summary": str(row.get('summary', '')),
                        "date": str(row.get('date', '')),
                        "sentiment": str(row.get('sentiment', '')),
                        "primary_issue": str(row.get('primary_issue', '')),
                        "transcript_snippet": str(row.get('transcript_snippet', ''))
                    }
                }
                vectors_to_upsert.append(vector)
            
            index.upsert(vectors=vectors_to_upsert)
            print(f"Successfully upserted batch {i//batch_size + 1} with {len(vectors_to_upsert)} vectors.")

        except Exception as e:
            print(f"An error occurred during batch {i//batch_size + 1}: {e}")

    print("--- Indexing complete! ---")


if __name__ == "__main__":
    full_dataframe = get_data_from_sheet()
    if not full_dataframe.empty:
        new_data_df = find_new_rows_to_index(full_dataframe)
        if not new_data_df.empty:
            create_embeddings_and_upsert(new_data_df)

