#!/usr/bin/env python3
"""
Utility script to send emails via Gmail SMTP or local mail command.
Can be used by other scripts to send notifications.

Usage:
    python common/send_email.py --subject "Subject" --body "Message"
    python common/send_email.py --subject "Subject" --body-file "path/to/log.txt"
"""
import os
import sys
import argparse
import smtplib
import ssl
import subprocess
import shutil
from email.message import EmailMessage
from pathlib import Path

def load_environment(project_dir):
    """Loads environment variables from .env file."""
    env_path = project_dir / ".env"
    if not env_path.exists():
        return

    # Try using python-dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass

    # Fallback: Manual parsing
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"Warning: Failed to parse .env file: {e}", file=sys.stderr)

def send_gmail(subject, body, to_email, user, password):
    """Sends email using Gmail SMTP."""
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = to_email

    context = ssl.create_default_context()
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(msg)
    print(f"Email sent to {to_email} via Gmail SMTP")

def send_local_mail(subject, body, to_email):
    """Sends email using local mail command."""
    if not shutil.which('mail'):
        print("Error: 'mail' command not found and Gmail credentials not provided.")
        return False
    
    try:
        process = subprocess.Popen(['mail', '-s', subject, to_email], stdin=subprocess.PIPE, text=True)
        process.communicate(input=body)
        if process.returncode == 0:
            print(f"Email sent to {to_email} via local mail")
            return True
        else:
            print(f"Failed to send email via local mail (exit code {process.returncode})")
            return False
    except Exception as e:
        print(f"Error sending local mail: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Send email notification")
    parser.add_argument("--subject", required=True, help="Email subject")
    parser.add_argument("--body", help="Email body text")
    parser.add_argument("--body-file", help="File containing email body")
    parser.add_argument("--to", help="Recipient email (defaults to ALERT_EMAIL env var)")
    
    args = parser.parse_args()
    
    # Resolve project root (assuming script is in common/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    load_environment(project_root)
    
    to_email = args.to or os.environ.get("ALERT_EMAIL")
    if not to_email:
        print("No recipient specified and ALERT_EMAIL not set. Skipping email.")
        sys.exit(0)
        
    body = args.body or ""
    if args.body_file:
        try:
            with open(args.body_file, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()
                if body:
                    body += "\n\n" + file_content
                else:
                    body = file_content
        except Exception as e:
            print(f"Error reading body file: {e}", file=sys.stderr)
            sys.exit(1)
            
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_password = os.environ.get("GMAIL_PASSWORD")
    
    success = False
    if gmail_user and gmail_password:
        try:
            send_gmail(args.subject, body, to_email, gmail_user, gmail_password)
            success = True
        except Exception as e:
            print(f"Error sending via Gmail: {e}", file=sys.stderr)
            print("Attempting fallback to local mail...", file=sys.stderr)
    
    if not success:
        send_local_mail(args.subject, body, to_email)

if __name__ == "__main__":
    main()