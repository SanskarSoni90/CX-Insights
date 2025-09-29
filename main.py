import os
import requests
import time
import json
from datetime import datetime, timedelta, timezone
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai
import sys
import traceback

# --- Configuration ---
# Load secrets from environment variables
EXOTEL_ACCOUNT_SID = os.getenv('EXOTEL_ACCOUNT_SID')
EXOTEL_API_KEY = os.getenv('EXOTEL_API_KEY')
EXOTEL_API_TOKEN = os.getenv('EXOTEL_API_TOKEN')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_SHEET_ID = '1LGeqJTaX6IfjHz2-H1YGzuRV3FM22P33Uqx2nwp5Zjs'
GOOGLE_SHEET_NAME = 'Sheet1'
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')

# File to store processed call SIDs
PROCESSED_CALLS_FILE = 'processed_calls.txt'

# --- Helper Functions ---

def log(message):
    """Enhanced logging with timestamp and flush."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)
    sys.stdout.flush()

def check_environment_variables():
    """Verify all required environment variables are set."""
    log("Checking environment variables...")
    required_vars = {
        'EXOTEL_ACCOUNT_SID': EXOTEL_ACCOUNT_SID,
        'EXOTEL_API_KEY': EXOTEL_API_KEY,
        'EXOTEL_API_TOKEN': EXOTEL_API_TOKEN,
        'ASSEMBLYAI_API_KEY': ASSEMBLYAI_API_KEY,
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'GOOGLE_CREDENTIALS_JSON': GOOGLE_CREDENTIALS_JSON
    }
    
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing_vars.append(var_name)
            log(f"❌ {var_name} is NOT set")
        else:
            # Show partial value for debugging (first 10 chars)
            display_value = var_value[:10] + "..." if len(var_value) > 10 else var_value
            log(f"✅ {var_name} is set: {display_value}")
    
    if missing_vars:
        log(f"🚫 CRITICAL: Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    log("✅ All environment variables are set")
    return True

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
    log("=" * 50)
    log("Fetching recent calls from Exotel...")
    
    try:
        # Define a start and end time for the date range
        start_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        end_time = datetime.now(timezone.utc) 

        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

        # Combine gte (start) and lte (end) with a semicolon as required by the API
        date_range_str = f"gte:{start_time_str};lte:{end_time_str}"
        
        log(f"Fetching calls within range: {start_time_str} to {end_time_str}")

        # --- OPTIMIZATION: Target only the Singapore cluster endpoint ---
        endpoint_name = "Singapore Cluster"
        url = f"https://api.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json"
        # -------------------------------------------------------------
        
        params = {
            'DateCreated': date_range_str,
            'PageSize': 20
        }
        
        log(f"Request params: {params}")
        
        try:
            log(f"Trying {endpoint_name}: {url}")
            response = requests.get(
                url, 
                auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN), 
                params=params, 
                timeout=30
            )
            
            log(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                log(f"✅ SUCCESS! Connected to {endpoint_name}")
                data = response.json()
                calls = data.get('Calls', [])
                log(f"Found {len(calls)} calls in the last 15 minutes")
                
                if calls:
                    log("First call details:")
                    first_call = calls[0]
                    log(f"  - Call SID: {first_call.get('Sid')}")
                    log(f"  - Status: {first_call.get('Status')}")
                    log(f"  - Date Created: {first_call.get('DateCreated')}")
                
                return calls
            
            # Handle specific error codes
            elif response.status_code == 401:
                log(f"❌ Authentication failed. Please check your API Key and Token.")
            elif response.status_code == 404:
                log(f"❌ Account not found. Please check your Account SID.")
            else:
                log(f"❌ {endpoint_name} returned status {response.status_code}")
                log(f"Response: {response.text[:300]}") # Show more of the response
            
        except requests.exceptions.RequestException as e:
            log(f"❌ Network error while connecting to {endpoint_name}: {e}")
        
        # If we get here, the single endpoint call failed
        log("🚫 Failed to fetch calls from Exotel.")
        return []
        
    except Exception as e:
        log(f"EXCEPTION in fetch_recent_exotel_calls: {str(e)}")
        log(f"Full traceback: {traceback.format_exc()}")
        return []

def transcribe_audio(audio_url):
    """Submits audio for transcription to AssemblyAI and waits for the result."""
    log(f"Transcribing audio from: {audio_url}")
    try:
        headers = {'authorization': ASSEMBLYAI_API_KEY}
        json_data = {'audio_url': audio_url}
        
        # Submit transcription request
        upload_response = requests.post('https://api.assemblyai.com/v2/transcript', json=json_data, headers=headers)
        upload_response.raise_for_status()
        transcript_id = upload_response.json()['id']
        log(f"Transcription submitted. ID: {transcript_id}")

        # Poll for completion
        while True:
            poll_url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
            poll_response = requests.get(poll_url, headers=headers)
            poll_response.raise_for_status()
            transcript_data = poll_response.json()
            if transcript_data['status'] == 'completed':
                log("Transcription complete.")
                return transcript_data['text']
            elif transcript_data['status'] == 'failed':
                log(f"Transcription failed: {transcript_data.get('error')}")
                return None
            log("Transcription in progress, waiting...")
            time.sleep(5)
    except Exception as e:
        log(f"Error during transcription: {e}")
        log(f"Full traceback: {traceback.format_exc()}")
        return None

def analyze_transcript_with_openai(transcript):
    """Analyzes the transcript using OpenAI to extract insights."""
    log("Analyzing transcript with OpenAI...")
    if not transcript:
        return {}
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
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
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            response_format={"type": "json_object"}
        )
        analysis = json.loads(response.choices[0].message.content)
        log("Analysis complete.")
        return analysis
    except Exception as e:
        log(f"Error analyzing transcript with OpenAI: {e}")
        log(f"Full traceback: {traceback.format_exc()}")
        return {}

def update_google_sheet(data_row):
    """Appends a new row of data to the Google Sheet."""
    log("Updating Google Sheet...")
    try:
        creds_json = json.loads(GOOGLE_CREDENTIALS_JSON)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        
        headers = [
            'call_id', 'date', 'duration', 'direction', 'primary_issue', 'issue_category',
            'sentiment', 'urgency', 'resolution_status', 'customer_satisfaction',
            'key_topics', 'action_items', 'agent_performance', 'summary',
            'transcript_snippet', 'transcript'
        ]
        
        row_to_insert = [data_row.get(h, "") for h in headers]
        sheet.append_row(row_to_insert)
        log("✅ Google Sheet updated successfully.")
    except Exception as e:
        log(f"Error updating Google Sheet: {e}")
        log(f"Full traceback: {traceback.format_exc()}")

# --- Main Logic ---
def main():
    """Main function to orchestrate the process."""
    log("=" * 50)
    log("CX INSIGHTS - CALL PROCESSING STARTING")
    log("=" * 50)
    
    try:
        # Check environment variables first
        if not check_environment_variables():
            log("🚫 CRITICAL: Missing environment variables. Exiting.")
            sys.exit(1)
        
        # Ensure the processed_calls.txt file exists
        if not os.path.exists(PROCESSED_CALLS_FILE):
            with open(PROCESSED_CALLS_FILE, 'w') as f:
                pass
            log(f"Created {PROCESSED_CALLS_FILE}")

        # Set OpenAI API key for older versions
        if hasattr(openai, '__version__') and openai.__version__.startswith('0.'):
            openai.api_key = OPENAI_API_KEY

        processed_sids = get_processed_calls()
        log(f"Already processed {len(processed_sids)} calls")
        
        calls = fetch_recent_exotel_calls()
        
        if not calls:
            log("No new calls to process.")
            log("=" * 50)
            log("CALL PROCESSING COMPLETED")
            log("=" * 50)
            return

        log(f"Processing {len(calls)} calls...")
        
        for i, call in enumerate(calls, 1):
            call_sid = call.get('Sid')
            log(f"\n--- Processing call {i}/{len(calls)}: {call_sid} ---")
            
            # Skip if already processed
            if call_sid in processed_sids:
                log(f"⏭️  Skipping already processed call: {call_sid}")
                continue
                
            # Skip if not completed
            if call.get('Status') != 'completed':
                log(f"⏭️  Skipping call not yet completed: {call_sid} (Status: {call.get('Status')})")
                continue
                
            # Skip if no recording
            if not call.get('RecordingUrl'):
                log(f"⏭️  Skipping call with no recording: {call_sid}")
                add_processed_call(call_sid)
                continue

            # 1. Transcribe
            transcript = transcribe_audio(call.get('RecordingUrl'))
            if not transcript:
                log(f"❌ Failed to get transcript for call {call_sid}. Skipping.")
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
            
            full_data_row = {**call_data, **analysis}
            update_google_sheet(full_data_row)
            
            # 4. Mark as processed
            add_processed_call(call_sid)
            log(f"✅ Finished processing call: {call_sid}")
    
    except Exception as e:
        log(f"🚫 FATAL ERROR in main(): {str(e)}")
        log(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    log("=" * 50)
    log("CALL PROCESSING COMPLETED SUCCESSFULLY")
    log("=" * 50)


if __name__ == "__main__":
    main()
