#!/usr/bin/env python3

import requests
import os
import argparse
import json
import sys
from urllib.parse import urljoin
from pathlib import Path

# --- Configuration ---
# Use environment variables with defaults
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:7000")
PROCESS_MODEL_ID = "tools:hours"  # Hardcoded as requested
API_KEY_PATH = Path.home() / ".config" / "max" / "spiff_api_key"

# --- Helper Functions ---

def get_api_key(path: Path) -> str:
    """Reads the API key from the specified file."""
    try:
        key = path.read_text().strip()
        if not key:
            print(f"Error: API key file '{path}' is empty.", file=sys.stderr)
            sys.exit(1)
        return key
    except FileNotFoundError:
        print(f"Error: API key file not found at '{path}'.", file=sys.stderr)
        print("Please ensure the file exists and contains your SpiffWorkflow API key.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading API key file '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def make_api_request(method, endpoint, api_key, **kwargs):
    """Makes an authenticated request to the SpiffWorkflow backend using an API key."""
    url = urljoin(BACKEND_BASE_URL, endpoint)
    headers = {"SpiffWorkflow-Api-Key": api_key}
    try:
        response = requests.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        # Handle cases where response might be empty but successful (e.g., 204 No Content)
        if response.status_code == 204:
            return None
        # Check if content type is JSON before trying to parse
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            return response.text # Return raw text if not JSON
    except requests.exceptions.RequestException as e:
        print(f"API request error ({method} {url}): {e}", file=sys.stderr)
        if response is not None:
            print(f"Response status: {response.status_code}", file=sys.stderr)
            print(f"Response body: {response.text}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"API request error ({method} {url}): Failed to decode JSON response.", file=sys.stderr)
        print(f"Response status: {response.status_code}", file=sys.stderr)
        print(f"Response body: {response.text}", file=sys.stderr)
        sys.exit(1)


def check_result_for_error(result, operation_description):
    """Checks if the API result contains an error code."""
    if isinstance(result, dict) and result.get("error_code"):
        print(f"ERROR: {operation_description} failed. Received error: {result}", file=sys.stderr)
        sys.exit(1)

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description=f"Start and complete the '{PROCESS_MODEL_ID}' process.")
    parser.add_argument("client", help="The client associated with the task.")
    parser.add_argument("project", help="The project associated with the task.")
    parser.add_argument("summary", help="The summary to include in the task.")
    parser.add_argument("start_date_time", help="Start time in ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS).")
    parser.add_argument("end_time", help="End time in ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS).")

    args = parser.parse_args()

    # 1. Get API Key
    print(f"Reading API key from {API_KEY_PATH}...")
    api_key = get_api_key(API_KEY_PATH)
    print("API key loaded.")

    # 2. Start Process Instance
    print(f"Starting process instance for '{PROCESS_MODEL_ID}'...")
    # The process model identifier needs '/' replaced with ':' for the API endpoint
    modified_process_model_identifier = PROCESS_MODEL_ID.replace('/', ':')
    start_result = make_api_request("POST", f"/v1.0/process-instances/{modified_process_model_identifier}", api_key)
    check_result_for_error(start_result, "Start process instance")

    try:
        process_instance_id = start_result["id"]
        print(f"Process instance created with ID: {process_instance_id}")
    except (KeyError, TypeError):
        print(f"ERROR: Could not get process instance ID from response: {start_result}", file=sys.stderr)
        sys.exit(1)

    # 3. Run Process Instance (to get to the user task)
    print(f"Running process instance {process_instance_id}...")
    run_result = make_api_request("POST", f"/v1.0/process-instances/{modified_process_model_identifier}/{process_instance_id}/run", api_key)
    check_result_for_error(run_result, "Run process instance")

    # Check if process completed immediately (unlikely if there's a user task)
    if isinstance(run_result, dict) and run_result.get("status") == "complete":
        print("Process instance completed immediately.")
        sys.exit(0)

    # 4. Find the User Task
    # We use the progress endpoint as it often returns the next task directly
    print(f"Checking progress for process instance {process_instance_id} to find user task...")
    progress_result = make_api_request("GET", f"/v1.0/tasks/progress/{process_instance_id}", api_key)
    check_result_for_error(progress_result, "Check progress")

    try:
        task = progress_result.get("task")
        if not task:
            # Fallback: List tasks for the instance if progress doesn't return the task directly
            print("Progress endpoint did not return task, trying to list tasks...")
            tasks_list = make_api_request("GET", f"/v1.0/tasks?process_instance_id={process_instance_id}", api_key)
            check_result_for_error(tasks_list, "List tasks")
            # Find the first READY User Task
            ready_user_tasks = [t for t in tasks_list if t.get("type") == "UserTask" and t.get("state") == "READY"]
            if not ready_user_tasks:
                 print(f"ERROR: No READY User Task found for process instance {process_instance_id}.", file=sys.stderr)
                 print(f"Current tasks: {tasks_list}", file=sys.stderr)
                 sys.exit(1)
            task = ready_user_tasks[0] # Assume the first one

        task_id = task.get("id")
        task_type = task.get("type")
        task_state = task.get("state")

        if not task_id or task_type != "UserTask" or task_state != "READY":
            print(f"ERROR: Expected a READY User Task, but found: {task}", file=sys.stderr)
            sys.exit(1)

        print(f"Found READY User Task with ID: {task_id}")

    except (KeyError, TypeError, IndexError) as e:
        print(f"ERROR: Could not find or parse user task from progress/list response: {e}", file=sys.stderr)
        print(f"Progress response was: {progress_result}", file=sys.stderr)
        if 'tasks_list' in locals():
             print(f"List tasks response was: {tasks_list}", file=sys.stderr)
        sys.exit(1)


    # 5. Complete the User Task
    print(f"Completing User Task {task_id}...")
    task_data = {
        "client": args.client,
        "project": args.project,
        "summary": args.summary,
        "start_date_time": args.start_date_time,
        "end_time": args.end_time,
        # Add any other required form fields here with default or derived values if necessary
    }
    # The endpoint requires the process_instance_id AND task_id
    complete_result = make_api_request("PUT", f"/v1.0/tasks/{process_instance_id}/{task_id}", api_key, json=task_data)
    # Check the result of the PUT request itself for errors
    check_result_for_error(complete_result, f"Complete User Task {task_id}")

    # The PUT request might return the updated task or process instance info
    print(f"User Task {task_id} submitted successfully.")
    if complete_result:
        print(f"Completion response: {json.dumps(complete_result, indent=2)}")

    # Optional: Add loop here to check process status until completion if needed

    print(f"\nProcess '{PROCESS_MODEL_ID}' instance {process_instance_id} interaction finished.")
    print("Check SpiffWorkflow UI or logs for final status.")

if __name__ == "__main__":
    main()
