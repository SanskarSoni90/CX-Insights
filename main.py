import os
import requests
import time
import json
from datetime import datetime, timedelta, timezone
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai

# --- Configuration ---
# Load secrets from environment variables
EXOTEL_ACCOUNT_SID = os.getenv('EXOTEL_ACCOUNT_SID')
EXOTEL_API_KEY = os.getenv('EXOTEL_API_KEY')
EXOTEL_API_TOKEN = os.getenv('EXOTEL_API_TOKEN')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_SHEET_ID = '1LGeqJTaX6IfjHz2-H1YGzuRV3FM22P33Uqx2nwp5Zjs'
GOOGLE_SHEET_NAME = 'Sheet1'
# The GOOGLE_CREDENTIALS_JSON is a multi-line secret, it needs to be loaded carefully
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')

# File to store processed call SIDs
PROCESSED_CALLS_FILE = 'processed_calls.txt'

# --- Helper Functions ---

def get_processed_calls():
    """Reads the set of already processed call SIDs from a file."""
    if not os.path.exists(PROCESSED_CALLS_FILE):
        return set()
    with open(PROCESSED_CALLS_FILE, 'r') as f:
        return set(line.strip() for line in f)

def add_processed_call(sid):
    """Adds a new call SID to the processed calls file."""
    with open(PROCESSED_CALLS_FILE, 'a') as f:
        f.write(sid + '\n')

def fetch_recent_exotel_calls():
    """Fetches calls from Exotel from the last 15 minutes."""
    print("Fetching recent calls from Exotel...")
    try:
        # Exotel API uses UTC. We fetch calls from the last 15 minutes.
        start_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        # CORRECTED: Use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) for the date.
        # This is a more standard and reliable format for APIs.
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        url = f"https://api.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json"
        params = {
            'DateCreated': f'gte:{start_time_str}',
            'PageSize': 20 # Adjust as needed
        }
        response = requests.get(url, auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN), params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Found {len(data.get('Calls', []))} calls in the last 15 minutes.")
        return data.get('Calls', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching calls from Exotel: {e}")
        return []

def transcribe_audio(audio_url):
    """Submits audio for transcription to AssemblyAI and waits for the result."""
    print(f"Transcribing audio from: {audio_url}")
    try:
        headers = {'authorization': ASSEMBLYAI_API_KEY}
        json_data = {'audio_url': audio_url}
        
        # Submit transcription request
        upload_response = requests.post('https://api.assemblyai.com/v2/transcript', json=json_data, headers=headers)
        upload_response.raise_for_status()
        transcript_id = upload_response.json()['id']
        print(f"Transcription submitted. ID: {transcript_id}")

        # Poll for completion
        while True:
            poll_url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
            poll_response = requests.get(poll_url, headers=headers)
            poll_response.raise_for_status()
            transcript_data = poll_response.json()
            if transcript_data['status'] == 'completed':
                print("Transcription complete.")
                return transcript_data['text']
            elif transcript_data['status'] == 'failed':
                print(f"Transcription failed: {transcript_data.get('error')}")
                return None
            print("Transcription in progress, waiting...")
            time.sleep(5)
    except requests.exceptions.RequestException as e:
        print(f"Error during transcription: {e}")
        return None

def analyze_transcript_with_openai(transcript):
    """Analyzes the transcript using OpenAI to extract insights."""
    print("Analyzing transcript with OpenAI...")
    if not transcript:
        return {}
    
    openai.api_key = OPENAI_API_KEY
    
    system_prompt = """
    You are an expert call center analyst. Analyze the following call transcript and extract the specified information.
    Respond ONLY with a valid JSON object. Do not include any explanatory text before or after the JSON.
    The JSON object should have the following keys:
    "primary_issue", "issue_category", "sentiment", "urgency", "resolution_status", 
    "customer_satisfaction", "key_topics", "action_items", "agent_performance", 
    "summary", "transcript_snippet"
    
    - sentiment: "Positive", "Negative", or "Neutral".
    - urgency: "High", "Medium", or "Low".
    - resolution_status: "Resolved", "Unresolved", or "Needs Follow-up".
    - customer_satisfaction: A rating from 1 to 5.
    - key_topics: A comma-separated string of main topics.
    - action_items: A comma-separated string of action items.
    - transcript_snippet: A relevant 1-2 sentence snippet from the transcript.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            response_format={"type": "json_object"}
        )
        analysis = json.loads(response.choices[0].message.content)
        print("Analysis complete.")
        return analysis
    except Exception as e:
        print(f"Error analyzing transcript with OpenAI: {e}")
        return {}

def update_google_sheet(data_row):
    """Appends a new row of data to the Google Sheet."""
    print("Updating Google Sheet...")
    try:
        # Load credentials from the environment variable
        creds_json = json.loads(GOOGLE_CREDENTIALS_JSON)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        
        # The order must match the columns in the sheet
        headers = [
            'call_id', 'date', 'duration', 'direction', 'primary_issue', 'issue_category',
            'sentiment', 'urgency', 'resolution_status', 'customer_satisfaction',
            'key_topics', 'action_items', 'agent_performance', 'summary',
            'transcript_snippet', 'transcript'
        ]
        
        # Ensure all headers have a value, default to ""
        row_to_insert = [data_row.get(h, "") for h in headers]

        sheet.append_row(row_to_insert)
        print("Google Sheet updated successfully.")
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")

# --- Main Logic ---
def main():
    """Main function to orchestrate the process."""
    processed_sids = get_processed_calls()
    calls = fetch_recent_exotel_calls()
    
    if not calls:
        print("No new calls to process.")
        return

    for call in calls:
        call_sid = call.get('Sid')
        
        # Skip if already processed or not completed
        if call_sid in processed_sids:
            print(f"Skipping already processed call: {call_sid}")
            continue
        if call.get('Status') != 'completed':
            print(f"Skipping call not yet completed: {call_sid}")
            continue
        if not call.get('RecordingUrl'):
            print(f"Skipping call with no recording: {call_sid}")
            add_processed_call(call_sid) # Add to processed to avoid re-checking
            continue

        print(f"--- Processing new call: {call_sid} ---")
        
        # 1. Transcribe
        transcript = transcribe_audio(call.get('RecordingUrl'))
        if not transcript:
            print(f"Failed to get transcript for call {call_sid}. Skipping.")
            continue
        
        # 2. Analyze
        analysis = analyze_transcript_with_openai(transcript)
        
        # 3. Prepare data and update sheet
        call_data = {
            'call_id': call.get('Sid'),
            'date': call.get('DateCreated'),
            'duration': call.get('Duration'),
            'direction': call.get('Direction'),
            'transcript': transcript
        }
        
        # Combine call data with analysis results
        full_data_row = {**call_data, **analysis}
        
        update_google_sheet(full_data_row)
        
        # 4. Mark as processed
        add_processed_call(call_sid)
        print(f"--- Finished processing call: {call_sid} ---")


if __name__ == "__main__":
    main()

