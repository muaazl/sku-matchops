#!/usr/bin/env python3
"""
SKU MatchOps — Automated Cloudflare Quick Tunnel Manager
Launches a Cloudflare Tunnel for the FastAPI backend and automatically syncs
the generated URL with Google Sheets (_config tab) so the Vercel frontend
and Google Sheets add-on connect seamlessly.
"""

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import json

from dotenv import load_dotenv

load_dotenv()

DEFAULT_APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL", "")
DEFAULT_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")

def find_or_download_cloudflared() -> str:
    """Finds existing cloudflared binary or downloads it automatically."""
    # 1. Check system PATH
    sys_binary = shutil.which("cloudflared")
    if sys_binary:
        return sys_binary

    # 2. Check local workspace directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    is_windows = platform.system().lower() == "windows"
    binary_name = "cloudflared.exe" if is_windows else "cloudflared"
    
    local_path = os.path.join(root_dir, binary_name)
    if os.path.exists(local_path) and os.access(local_path, os.X_OK if not is_windows else os.F_OK):
        return local_path

    # Check /usr/local/bin or /tmp
    for fallback in ["/usr/local/bin/cloudflared", "/tmp/cloudflared"]:
        if os.path.exists(fallback):
            return fallback

    # 3. Download cloudflared if missing
    print(f"[TUNNEL] 'cloudflared' not found. Downloading binary for {platform.system()} {platform.machine()}...", flush=True)
    
    if is_windows:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        dest = local_path
    elif platform.system().lower() == "linux":
        machine = platform.machine().lower()
        arch = "arm64" if "arm" in machine or "aarch64" in machine else "amd64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        dest = "/tmp/cloudflared" if not os.access("/usr/local/bin", os.W_OK) else "/usr/local/bin/cloudflared"
    elif platform.system().lower() == "darwin":
        machine = platform.machine().lower()
        arch = "arm64" if "arm" in machine else "amd64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-{arch}.tgz"
        dest = "/tmp/cloudflared"
    else:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")

    print(f"[TUNNEL] Downloading from {url} to {dest}...", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        if not is_windows:
            os.chmod(dest, 0o755)
        print(f"[TUNNEL] Successfully downloaded cloudflared to {dest}!", flush=True)
        return dest
    except Exception as e:
        raise RuntimeError(f"Failed to download cloudflared: {e}")

def update_google_sheets(apps_script_url: str, sheet_id: str, tunnel_url: str):
    """Sends the new tunnel URL to the Google Apps Script webhook."""
    print(f"[TUNNEL] Syncing Tunnel URL with Google Sheets (_config tab)...", flush=True)
    payload = {
        "type": "tunnel_update",
        "url": tunnel_url,
        "spreadsheet_id": sheet_id
    }
    
    req = urllib.request.Request(
        apps_script_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_data = resp.read().decode("utf-8")
            print(f"[TUNNEL] Google Sheets updated successfully! Response: {resp_data}", flush=True)
    except Exception as e:
        print(f"[TUNNEL] [WARN] Could not automatically update Google Sheets: {e}", flush=True)
        print(f"[TUNNEL] You can manually paste the URL into Google Sheets _config tab: {tunnel_url}", flush=True)

def run_tunnel(backend_url: str, apps_script_url: str, sheet_id: str, skip_sync: bool = False):
    binary = find_or_download_cloudflared()
    print("=" * 60, flush=True)
    print("         SKU MatchOps — Cloudflare Quick Tunnel          ", flush=True)
    print("=" * 60, flush=True)
    print(f"  Target Backend : {backend_url}", flush=True)
    print(f"  Binary Path    : {binary}", flush=True)
    print("=" * 60, flush=True)
    print("[TUNNEL] Starting Cloudflare Quick Tunnel...", flush=True)

    cmd = [binary, "tunnel", "--url", backend_url]
    
    # Launch subprocess capturing stderr where cloudflared logs the tunnel URL
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    tunnel_url = None
    url_pattern = re.compile(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)")

    def cleanup_and_exit(signum=None, frame=None):
        print("\n[TUNNEL] Stopping Cloudflare Tunnel...", flush=True)
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            process.kill()
        print("[TUNNEL] Stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    # Read stderr to extract the trycloudflare URL
    start_time = time.time()
    while process.poll() is None:
        line = process.stderr.readline()
        if not line:
            continue
        
        # Look for URL in output line
        if not tunnel_url:
            match = url_pattern.search(line)
            if match:
                tunnel_url = match.group(1)
                print("\n" + "=" * 60, flush=True)
                print(f"  TUNNEL LIVE AT : {tunnel_url}", flush=True)
                print(f"  FORWARDING TO  : {backend_url}", flush=True)
                print("=" * 60 + "\n", flush=True)
                
                if not skip_sync:
                    update_google_sheets(apps_script_url, sheet_id, tunnel_url)
                else:
                    print("[TUNNEL] Skipped Google Sheets update (--skip-sync)", flush=True)
        else:
            # Print periodic keep-alive or connection logs if desired
            if "error" in line.lower() or "warn" in line.lower():
                print(f"[cloudflared] {line.strip()}", flush=True)

    # If process died before finding URL
    if not tunnel_url:
        print("[TUNNEL] [ERROR] cloudflared exited without providing a tunnel URL.", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKU MatchOps Cloudflare Tunnel Manager")
    parser.add_argument("--url", default=os.getenv("BACKEND_URL", "http://localhost:8000"), help="Backend URL to expose")
    parser.add_argument("--apps-script-url", default=os.getenv("APPS_SCRIPT_URL", DEFAULT_APPS_SCRIPT_URL), help="Google Apps Script URL")
    parser.add_argument("--sheet-id", default=os.getenv("GOOGLE_SHEET_ID", DEFAULT_SHEET_ID), help="Google Sheet ID")
    parser.add_argument("--skip-sync", action="store_true", help="Skip pushing URL to Google Sheets")
    args = parser.parse_args()

    run_tunnel(
        backend_url=args.url,
        apps_script_url=args.apps_script_url,
        sheet_id=args.sheet_id,
        skip_sync=args.skip_sync
    )
