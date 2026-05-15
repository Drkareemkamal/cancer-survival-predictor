import os
import time
from collections import defaultdict
from pathlib import Path

# --- Configuration ---
# We hardcode your username here so 'sudo' doesn't redirect the script to /root
USER_NAME = "drkareemkamal"
TARGET_DIR = Path(f"/home/{USER_NAME}/.local/share/wandb/artifacts/staging")

# 1. SET TO True TO ENABLE DELETION
REAL_DELETE = True 

# 2. Check every 60 seconds
INTERVAL = 300 
# --- End Configuration ---

def clean_cycle():
    print(f"\n--- Scan Start: {time.strftime('%H:%M:%S')} ---")
    
    # Check if the path exists before proceeding
    if not TARGET_DIR.exists():
        print(f"Directory not found: {TARGET_DIR}")
        print("Please check if the USER_NAME in the script is correct.")
        return

    size_groups = defaultdict(list)
    
    try:
        # Focusing on the files seen in image_55aae4.png
        files = [f for f in TARGET_DIR.iterdir() if f.is_file()]
    except PermissionError:
        print(f"CRITICAL: Permission denied even with sudo at {TARGET_DIR}")
        return

    print(f"Analyzing {len(files)} files in staging...")

    for f in files:
        try:
            stat = f.stat()
            size = stat.st_size
            mtime = stat.st_mtime
            
            size_groups[size].append({
                'path': f,
                'name': f.name,
                'mtime': mtime,
                'size': size
            })
        except Exception as e:
            print(f" Warning: Could not read metadata for {f.name}: {e}")

    for size, group in size_groups.items():
        if len(group) > 1:
            # Sort by modification time: Newest (largest timestamp) first
            sorted_files = sorted(group, key=lambda x: x['mtime'], reverse=True)
            
            # The newest file is kept[cite: 1]
            keep = sorted_files[0]
            to_delete = sorted_files[3:]

            print(f"\n[!] Duplicate Detected ({size} bytes)")
            print(f"    KEEPING: {keep['name']} (Modified: {time.ctime(keep['mtime'])})")

            for item in to_delete:
                if REAL_DELETE:
                    try:
                        item['path'].unlink()
                        print(f"    DELETED: {item['name']}")
                    except Exception as e:
                        print(f"    FAILED to delete {item['name']}: {e}")
                else:
                    print(f"    WOULD DELETE: {item['name']} (Older version)")

if __name__ == "__main__":
    print(f"Monitor active. Targeting: {TARGET_DIR}")
    if not REAL_DELETE:
        print("--- RUNNING IN DRY-RUN MODE ---")
    
    try:
        while True:
            clean_cycle()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")