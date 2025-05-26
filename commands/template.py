import typer
from typing import Optional
import sqlite3
import os
import json
import csv
import difflib
import random
import string
import shutil
from datetime import datetime
import requests  # Add requests library for HTTP requests
import yaml

app = typer.Typer()

# -----------------------------------------------------
# Database helpers: create/connect/seed
# -----------------------------------------------------
DB_NAME = "app_data.db"


def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_NAME)


def create_db_if_not_exists():
    """Create tables if they do not exist and seed them with mock data."""
    conn = get_connection()
    cur = conn.cursor()

    # Create a sample 'users' table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    )

    # Create a sample 'tasks' table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name TEXT NOT NULL,
        priority INTEGER NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    )

    # Create a sample 'logs' table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        level TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    )

    # Check if 'users' table has data; if not, seed 25 rows
    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]
    if user_count == 0:
        roles = ["guest", "admin", "editor", "viewer"]
        for i in range(25):
            username = f"user_{i}"
            role = random.choice(roles)
            created_at = datetime.now().isoformat()
            cur.execute(
                "INSERT INTO users (username, role, created_at) VALUES (?, ?, ?)",
                (username, role, created_at),
            )

    # Seed 'tasks' table
    cur.execute("SELECT COUNT(*) FROM tasks")
    task_count = cur.fetchone()[0]
    if task_count == 0:
        statuses = ["pending", "in-progress", "complete"]
        for i in range(25):
            task_name = f"task_{i}"
            priority = random.randint(1, 5)
            status = random.choice(statuses)
            created_at = datetime.now().isoformat()
            cur.execute(
                "INSERT INTO tasks (task_name, priority, status, created_at) VALUES (?, ?, ?, ?)",
                (task_name, priority, status, created_at),
            )

    # Seed 'logs' table
    cur.execute("SELECT COUNT(*) FROM logs")
    logs_count = cur.fetchone()[0]
    if logs_count == 0:
        levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        for i in range(25):
            message = f"Log entry number {i}"
            level = random.choice(levels)
            created_at = datetime.now().isoformat()
            cur.execute(
                "INSERT INTO logs (message, level, created_at) VALUES (?, ?, ?)",
                (message, level, created_at),
            )

    conn.commit()
    conn.close()


# Ensure the database and tables exist before we do anything
create_db_if_not_exists()


# -----------------------------------------------------
# Simple Caesar cipher for “encryption/decryption” demo
# -----------------------------------------------------
def caesar_cipher_encrypt(plaintext: str, shift: int = 3) -> str:
    """A simple Caesar cipher encryption function."""
    result = []
    for ch in plaintext:
        if ch.isalpha():
            start = ord("A") if ch.isupper() else ord("a")
            offset = (ord(ch) - start + shift) % 26
            result.append(chr(start + offset))
        else:
            result.append(ch)
    return "".join(result)


def caesar_cipher_decrypt(ciphertext: str, shift: int = 3) -> str:
    """A simple Caesar cipher decryption function."""
    return caesar_cipher_encrypt(ciphertext, -shift)


# -----------------------------------------------------
# 2) show_config
# -----------------------------------------------------
@app.command()
def show_config(
    verbose: bool = typer.Option(False, "--verbose", help="Show config in detail?")
):
    """
    Shows the current configuration from modules/assistant_config.py.
    """
    try:

        config = ""

        with open("./assistant_config.yml", "r") as f:
            config = f.read()

        if verbose:
            result = f"Verbose config:\n{json.dumps(yaml.safe_load(config), indent=2)}"
        else:
            result = f"Config: {config}"
        typer.echo(result)
        return result
    except ImportError:
        result = "Error: Could not load assistant_config module"
        typer.echo(result)
        return result


# -----------------------------------------------------
# 3) list_files
# -----------------------------------------------------
@app.command()
def list_files(
    path: str = typer.Argument(..., help="Path to list files from"),
    all_files: bool = typer.Option(False, "--all", help="Include hidden files"),
):
    """
    Lists files in a directory. Optionally show hidden files.
    """
    if not os.path.isdir(path):
        msg = f"Path '{path}' is not a valid directory."
        typer.echo(msg)
        return msg

    entries = os.listdir(path)
    if not all_files:
        entries = [e for e in entries if not e.startswith(".")]

    result = f"Files in '{path}': {entries}"
    typer.echo(result)
    return result


# -----------------------------------------------------
# 7) backup_data
# -----------------------------------------------------
@app.command()
def backup_data(
    directory: str = typer.Argument(..., help="Directory to store backups"),
    full: bool = typer.Option(False, "--full", help="Perform a full backup"),
):
    """
    Back up data to a specified directory, optionally performing a full backup.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)

    backup_file = os.path.join(
        directory, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    )
    shutil.copy(DB_NAME, backup_file)

    result = (
        f"{'Full' if full else 'Partial'} backup completed. Saved to {backup_file}."
    )
    typer.echo(result)
    return result


# -----------------------------------------------------
# 8) restore_data
# -----------------------------------------------------
@app.command()
def restore_data(
    file_path: str = typer.Argument(..., help="File path of backup to restore"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing data"
    ),
):
    """
    Restores data from a backup file.
    """
    if not os.path.isfile(file_path):
        msg = f"Backup file {file_path} does not exist."
        typer.echo(msg)
        return msg

    if not overwrite:
        msg = "Overwrite not confirmed. Use --overwrite to proceed."
        typer.echo(msg)
        return msg

    shutil.copy(file_path, DB_NAME)
    msg = f"Data restored from {file_path} to {DB_NAME}."
    typer.echo(msg)
    return msg


@app.command()
def dirty_repos():
    """
    Gets list of dirty repos by shelling out to gdirtyrepos bash script
    """
    result = os.popen("gdirtyrepos").read()
    typer.echo(f"Git repos with local changes:\n{result}")
    return result

@app.command()
def change_dir(dir_full_path: str):
    """
    Changes to a directory and stores the current working directory
    """
    if not os.path.exists(dir_full_path):
        typer.echo(f"Directory path {dir_full_path} does not exist")
        return f"Directory path {dir_full_path} does not exist"
    
    # Store the new working directory
    config_dir = os.path.expanduser("~/.config/max")
    cwd_file = os.path.join(config_dir, "cwd")
    with open(cwd_file, "w") as f:
        f.write(dir_full_path)
    
    typer.echo(f"Changed to directory: {dir_full_path}")
    return f"Changed to directory: {dir_full_path}"

@app.command()
def update_spiff_process_model(
):
    """
    Updates the spiff process model
    """
    result = os.popen("./bin/update_spiff_process_model").read()

    typer.echo(f"Output from script: :\n{result}")
    return result

# -----------------------------------------------------
# 27) queue_task
# -----------------------------------------------------
@app.command()
def queue_task(
    task_name: str = typer.Argument(..., help="Name of the task to queue"),
    priority: int = typer.Option(1, "--priority", help="Priority of the task"),
    delay: int = typer.Option(
        0, "--delay", help="Delay in seconds before starting task"
    ),
):
    """
    Queues a task with a specified priority and optional delay.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO tasks (task_name, priority, status, created_at) VALUES (?, ?, ?, ?)",
        (task_name, priority, "pending", now),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()

    result = f"Task '{task_name}' queued with priority {priority}, delay {delay}s, assigned ID {task_id}."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 28) remove_task
# -----------------------------------------------------
@app.command()
def remove_task(
    task_id: str = typer.Argument(..., help="ID of the task to remove"),
    force: bool = typer.Option(False, "--force", help="Remove without confirmation"),
):
    """
    Removes a queued task by ID, optionally forcing removal without confirmation.
    """
    if not force:
        msg = f"Confirmation required to remove task {task_id}. Use --force."
        typer.echo(msg)
        return msg

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    removed = cur.rowcount
    conn.close()

    if removed:
        msg = f"Task {task_id} removed."
    else:
        msg = f"Task {task_id} not found."
    typer.echo(msg)
    return msg


# -----------------------------------------------------
# 29) list_tasks
# -----------------------------------------------------
@app.command()
def list_tasks(
    show_all: bool = typer.Option(
        False, "--all", help="Show all tasks, including completed"
    ),
    sort_by: str = typer.Option(
        "priority", "--sort-by", help="Sort tasks by this field"
    ),
):
    """
    Lists tasks, optionally including completed tasks or sorting by a different field.
    """
    valid_sort_fields = ["priority", "status", "created_at"]
    if sort_by not in valid_sort_fields:
        msg = f"Invalid sort field. Must be one of {valid_sort_fields}."
        typer.echo(msg)
        return msg

    conn = get_connection()
    cur = conn.cursor()
    if show_all:
        sql = f"SELECT id, task_name, priority, status, created_at FROM tasks ORDER BY {sort_by} ASC"
    else:
        sql = f"SELECT id, task_name, priority, status, created_at FROM tasks WHERE status != 'complete' ORDER BY {sort_by} ASC"

    cur.execute(sql)
    tasks = cur.fetchall()
    conn.close()

    result = "Tasks:\n"
    for t in tasks:
        result += (
            f"ID={t[0]}, Name={t[1]}, Priority={t[2]}, Status={t[3]}, Created={t[4]}\n"
        )

    typer.echo(result.strip())
    return result


# -----------------------------------------------------
# 30) inspect_task
# -----------------------------------------------------
@app.command()
def inspect_task(
    task_id: str = typer.Argument(..., help="ID of the task to inspect"),
    json_output: bool = typer.Option(
        False, "--json", help="Show output in JSON format"
    ),
):
    """
    Inspects a specific task by ID, optionally in JSON format.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, task_name, priority, status, created_at FROM tasks WHERE id = ?",
        (task_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        msg = f"No task found with ID {task_id}."
        typer.echo(msg)
        return msg

    task_dict = {
        "id": row[0],
        "task_name": row[1],
        "priority": row[2],
        "status": row[3],
        "created_at": row[4],
    }

    if json_output:
        result = json.dumps(task_dict, indent=2)
    else:
        result = f"Task ID={task_dict['id']}, Name={task_dict['task_name']}, Priority={task_dict['priority']}, Status={task_dict['status']}, Created={task_dict['created_at']}"
    typer.echo(result)
    return result


# -----------------------------------------------------
# 31) get_failed_spacelift_stacks
# -----------------------------------------------------
@app.command()
def get_failing_spacelift_stacks():
    """
    Gets the number of failing Spacelift stacks.
    """
    url = 'http://localhost:7000/v1.0/messages/chat?execution_mode=synchronous'
    spiff_token = os.getenv('SPIFF_TOKEN')
    headers = {
        'Authorization': f'Bearer {spiff_token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json={"command_type": "get_failing_spacelift_stacks"})
    if response.status_code == 200:
        response_data = response.json()
        message = response_data.get('task_data', {}).get('response_string', 'No response string found.')
        json_response_string = json.dumps({"verbatim_vocal_response": message})

        # this is what is actually used
        typer.echo(json_response_string)

        # this is ignored
        return json_response_string
    else:
        error_msg = f"Failed to get data: {response.status_code} {response.text}"
        typer.echo(error_msg)
        return error_msg


# -----------------------------------------------------
# 32) create_config
# -----------------------------------------------------
@app.command()
def create_config():
    """
    Creates a ~/.config/max directory if it doesn't exist.
    """
    config_dir = os.path.expanduser("~/.config/max")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        result = f"Directory '{config_dir}' created."
    else:
        result = f"Directory '{config_dir}' already exists."
    typer.echo(result)
    return result


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
def main():
    app()


if __name__ == "__main__":
    main()
