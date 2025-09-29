import os
import gspread
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
from oauth2client.service_account import ServiceAccountCredentials
import json

# --- Configuration ---
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
GOOGLE_SHEET_NAME = 'Sheet1'
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = 'call-insights' # Or whatever you named your index

# --- Initialize Clients ---
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_data_from_sheet():
    """Fetches data from Google Sheets and returns a pandas DataFrame."""
    print("Connecting to Google Sheets...")
    creds_json = json.loads(GOOGLE_CREDENTIALS_JSON)
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    records = sheet.get_all_records()
    print(f"Fetched {len(records)} records from the sheet.")
    return pd.DataFrame(records)

def create_embeddings_and_upsert(df):
    """Creates embeddings for each row and upserts them into Pinecone."""
    print("Creating embeddings and indexing data...")
    # OpenAI's embedding model
    EMBEDDING_MODEL = "text-embedding-3-small"

    # We will batch our upserts to be more efficient
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Create a meaningful text chunk for embedding
        # This combines key fields into a single string
        texts_to_embed = (
            "Call on " + batch['date'].astype(str) +
            " regarding '" + batch['primary_issue'].astype(str) +
            "'. Summary: " + batch['summary'].astype(str) +
            ". Action items: " + batch['action_items'].astype(str)
        ).tolist()

        # Get embeddings from OpenAI
        response = openai_client.embeddings.create(input=texts_to_embed, model=EMBEDDING_MODEL)
        embeddings = [item.embedding for item in response.data]

        # Prepare vectors for Pinecone upsert
        vectors_to_upsert = []
        for idx, row in batch.iterrows():
            vector = {
                "id": str(row['call_id']),  # Unique ID for each call
                "values": embeddings[batch.index.get_loc(idx)],
                "metadata": {  # Store other useful data here
                    "summary": str(row['summary']),
                    "date": str(row['date']),
                    "sentiment": str(row['sentiment']),
                    "transcript_snippet": str(row.get('transcript_snippet', ''))
                }
            }
            vectors_to_upsert.append(vector)
        
        # Upsert batch to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch {i//batch_size + 1}")

    print("Indexing complete!")


if __name__ == "__main__":
    dataframe = get_data_from_sheet()
    if not dataframe.empty:
        create_embeddings_and_upsert(dataframe)
