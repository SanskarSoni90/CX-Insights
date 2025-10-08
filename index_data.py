import os
import gspread
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from oauth2client.service_account import ServiceAccountCredentials
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# --- Configuration ---
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
GOOGLE_SHEET_NAME = 'Sheet1'
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = 'call-insights'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Clients
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
    
    existing_ids = set()
    for i in range(0, len(all_sheet_ids), 100):
        batch_ids = all_sheet_ids[i:i+100]
        fetch_response = index.fetch(ids=batch_ids)
        existing_ids.update(fetch_response.vectors.keys())

    new_ids = set(all_sheet_ids) - existing_ids
    print(f"Found {len(new_ids)} new rows to index.")
    
    return df[df['call_id'].astype(str).isin(new_ids)]


def compute_daily_aggregates(df):
    """Compute daily aggregates from the full dataset."""
    print("Computing daily aggregates...")
    
    if df.empty:
        return []
    
    # Parse dates
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['date_only'] = df['date_parsed'].dt.date
    
    aggregates = []
    
    # Group by date
    for date, group in df.groupby('date_only'):
        if pd.isna(date):
            continue
            
        agg_data = {
            'date': str(date),
            'total_calls': len(group),
            'sentiment_counts': group['sentiment'].value_counts().to_dict() if 'sentiment' in group.columns else {},
            'issue_categories': group['issue_category'].value_counts().to_dict() if 'issue_category' in group.columns else {},
            'avg_satisfaction': float(pd.to_numeric(group['customer_satisfaction'], errors='coerce').mean()) if 'customer_satisfaction' in group.columns else 0,
            'urgency_distribution': group['urgency'].value_counts().to_dict() if 'urgency' in group.columns else {},
            'resolution_status': group['resolution_status'].value_counts().to_dict() if 'resolution_status' in group.columns else {},
            'top_issues': group['primary_issue'].value_counts().head(5).to_dict() if 'primary_issue' in group.columns else {}
        }
        
        aggregates.append(agg_data)
    
    return aggregates


def compute_overall_aggregates(df):
    """Compute overall statistics across all data."""
    print("Computing overall aggregates...")
    
    if df.empty:
        return {}
    
    return {
        'total_calls': len(df),
        'date_range': {
            'start': str(pd.to_datetime(df['date'], errors='coerce').min().date()) if 'date' in df.columns else 'N/A',
            'end': str(pd.to_datetime(df['date'], errors='coerce').max().date()) if 'date' in df.columns else 'N/A'
        },
        'sentiment_distribution': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
        'issue_category_distribution': df['issue_category'].value_counts().to_dict() if 'issue_category' in df.columns else {},
        'top_10_issues': df['primary_issue'].value_counts().head(10).to_dict() if 'primary_issue' in df.columns else {},
        'avg_satisfaction': float(pd.to_numeric(df['customer_satisfaction'], errors='coerce').mean()) if 'customer_satisfaction' in df.columns else 0,
        'urgency_breakdown': df['urgency'].value_counts().to_dict() if 'urgency' in df.columns else {},
        'resolution_breakdown': df['resolution_status'].value_counts().to_dict() if 'resolution_status' in df.columns else {},
        'avg_duration': float(pd.to_numeric(df['duration'], errors='coerce').mean()) if 'duration' in df.columns else 0
    }


def create_aggregate_vectors(overall_agg, daily_aggs):
    """Create vectors for aggregate data."""
    EMBEDDING_MODEL = "text-embedding-3-large"
    PINECONE_DIMENSION = 1024
    
    vectors = []
    
    # Overall aggregate vector
    overall_text = f"""Overall Call Statistics:
Total calls: {overall_agg['total_calls']}
Date range: {overall_agg['date_range']['start']} to {overall_agg['date_range']['end']}
Sentiment distribution: {', '.join([f"{k}: {v}" for k, v in overall_agg['sentiment_distribution'].items()])}
Top issues: {', '.join([f"{k} ({v} calls)" for k, v in list(overall_agg['top_10_issues'].items())[:5]])}
Average satisfaction: {overall_agg['avg_satisfaction']:.2f}/5
Resolution status: {', '.join([f"{k}: {v}" for k, v in overall_agg['resolution_breakdown'].items()])}
Average call duration: {overall_agg['avg_duration']:.0f} seconds"""

    response = openai_client.embeddings.create(
        input=[overall_text], 
        model=EMBEDDING_MODEL,
        dimensions=PINECONE_DIMENSION
    )
    
    vectors.append({
        "id": f"aggregate_overall_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "values": response.data[0].embedding,
        "metadata": {
            "type": "aggregate_overall",
            "summary": overall_text,
            "total_calls": overall_agg['total_calls'],
            "date_range_start": overall_agg['date_range']['start'],
            "date_range_end": overall_agg['date_range']['end'],
            "avg_satisfaction": overall_agg['avg_satisfaction'],
            "generated_at": datetime.now().isoformat()
        }
    })
    
    # Daily aggregate vectors
    for daily_agg in daily_aggs[-30:]:  # Last 30 days
        daily_text = f"""Daily Statistics for {daily_agg['date']}:
Total calls: {daily_agg['total_calls']}
Sentiments: {', '.join([f"{k}: {v}" for k, v in daily_agg['sentiment_counts'].items()])}
Top issues: {', '.join([f"{k} ({v})" for k, v in list(daily_agg['top_issues'].items())[:3]])}
Average satisfaction: {daily_agg['avg_satisfaction']:.2f}/5
Urgency: {', '.join([f"{k}: {v}" for k, v in daily_agg['urgency_distribution'].items()])}"""

        response = openai_client.embeddings.create(
            input=[daily_text], 
            model=EMBEDDING_MODEL,
            dimensions=PINECONE_DIMENSION
        )
        
        vectors.append({
            "id": f"aggregate_daily_{daily_agg['date']}",
            "values": response.data[0].embedding,
            "metadata": {
                "type": "aggregate_daily",
                "date": daily_agg['date'],
                "summary": daily_text,
                "total_calls": daily_agg['total_calls'],
                "avg_satisfaction": daily_agg['avg_satisfaction']
            }
        })
    
    return vectors


def create_embeddings_and_upsert(df):
    """Creates embeddings for new rows and upserts them into Pinecone."""
    if df.empty:
        print("No new data to index.")
        return

    print(f"Creating embeddings for {len(df)} new rows...")
    EMBEDDING_MODEL = "text-embedding-3-large"
    PINECONE_DIMENSION = 1024

    batch_size = 50  # Reduced batch size for better stability
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Create rich text for embedding (without full transcript to save tokens)
        texts_to_embed = []
        for idx, row in batch.iterrows():
            text = (
                f"Call on {row.get('date', 'N/A')} "
                f"regarding '{row.get('primary_issue', 'N/A')}' "
                f"in category '{row.get('issue_category', 'N/A')}'. "
                f"Sentiment: {row.get('sentiment', 'N/A')}. "
                f"Urgency: {row.get('urgency', 'N/A')}. "
                f"Status: {row.get('resolution_status', 'N/A')}. "
                f"Summary: {row.get('summary', 'N/A')} "
                f"Key topics: {row.get('key_topics', 'N/A')}. "
                f"Action items: {row.get('action_items', 'N/A')}"
            )
            texts_to_embed.append(text)

        response = openai_client.embeddings.create(
            input=texts_to_embed, 
            model=EMBEDDING_MODEL,
            dimensions=PINECONE_DIMENSION
        )
        embeddings = [item.embedding for item in response.data]

        vectors_to_upsert = []
        for idx, row in batch.iterrows():
            # Store comprehensive but compact metadata
            vector = {
                "id": str(row['call_id']),
                "values": embeddings[batch.index.get_loc(idx)],
                "metadata": {
                    "date": str(row.get('date', '')),
                    "duration": str(row.get('duration', '')),
                    "direction": str(row.get('direction', '')),
                    "primary_issue": str(row.get('primary_issue', ''))[:200],
                    "issue_category": str(row.get('issue_category', '')),
                    "sentiment": str(row.get('sentiment', '')),
                    "urgency": str(row.get('urgency', '')),
                    "resolution_status": str(row.get('resolution_status', '')),
                    "customer_satisfaction": str(row.get('customer_satisfaction', '')),
                    "summary": str(row.get('summary', ''))[:500],  # Limit summary length
                    "transcript_snippet": str(row.get('transcript_snippet', ''))[:300],
                    "key_topics": str(row.get('key_topics', ''))[:200],
                    "action_items": str(row.get('action_items', ''))[:200]
                }
            }
            vectors_to_upsert.append(vector)
        
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")

    print("Individual call indexing complete!")


def upsert_aggregate_vectors(vectors):
    """Upsert aggregate vectors to Pinecone."""
    if not vectors:
        return
    
    print(f"Upserting {len(vectors)} aggregate vectors...")
    
    # Delete old aggregate vectors first
    try:
        index.delete(delete_all=True, namespace="aggregates")
        print("Cleared old aggregate vectors")
    except:
        pass
    
    # Upsert new aggregates
    index.upsert(vectors=vectors, namespace="aggregates")
    print("Aggregate vectors upserted successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED INDEXING WITH AGGREGATES")
    print("=" * 60)
    
    # Step 1: Get all data from sheet
    full_dataframe = get_data_from_sheet()
    
    if full_dataframe.empty:
        print("No data found in sheet.")
        exit(0)
    
    # Step 2: Find and index new individual calls
    new_data_df = find_new_rows_to_index(full_dataframe)
    create_embeddings_and_upsert(new_data_df)
    
    # Step 3: Compute and store aggregates (using ALL data)
    print("\n" + "=" * 60)
    print("COMPUTING AGGREGATES")
    print("=" * 60)
    
    overall_agg = compute_overall_aggregates(full_dataframe)
    daily_aggs = compute_daily_aggregates(full_dataframe)
    
    # Step 4: Create and upsert aggregate vectors
    aggregate_vectors = create_aggregate_vectors(overall_agg, daily_aggs)
    upsert_aggregate_vectors(aggregate_vectors)
    
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE!")
    print(f"- Individual calls indexed: {len(new_data_df)}")
    print(f"- Aggregate vectors created: {len(aggregate_vectors)}")
    print(f"- Total calls in database: {len(full_dataframe)}")
    print("=" * 60)
