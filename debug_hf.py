import requests
from huggingface_hub import whoami
import sys

print("--- Network & Auth Diagnostic ---")

# 1. Check Basic Internet
try:
    print("1. Key Internet Check (google.com)... ", end="")
    requests.get("https://www.google.com", timeout=5)
    print("✅ OK")
except Exception as e:
    print(f"❌ FAILED: {e}")

# 2. Check Hugging Face Reachability
try:
    print("2. Hugging Face Reachability (huggingface.co)... ", end="")
    requests.get("https://huggingface.co", timeout=5)
    print("✅ OK")
except Exception as e:
    print(f"❌ FAILED: {e}")

# 3. Check Authentication State
print("3. Checking HF Auth (whoami)...")
try:
    user_info = whoami()
    print(f"   ✅ Logged in as: {user_info['name']}")
    print(f"   ✅ Org memberships: {[org['name'] for org in user_info.get('orgs', [])]}")
except Exception as e:
    print(f"   ❌ Auth Check Failed: {e}")
    print("   (This means Python cannot find the token saved by the CLI, or the token is invalid)")

# 4. Check Specific Model Access
model_id = "google/medgemma-1.5-4b-it"
print(f"4. Checking access to {model_id}...")
try:
    # Try to fetch info (lightweight)
    from huggingface_hub import model_info
    info = model_info(model_id)
    print("   ✅ Access Confirmed! Model info retrieved.")
except Exception as e:
    print(f"   ❌ Access Denied or Not Found: {e}")
