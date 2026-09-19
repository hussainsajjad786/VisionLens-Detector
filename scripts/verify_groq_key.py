"""Safely verify the Groq key stored in this project's .env file."""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
api_key = os.getenv("GROQ_API_KEY", "").strip()

if not api_key or api_key == "replace_with_your_new_groq_key":
    raise SystemExit("No usable GROQ_API_KEY was found in .env.")
if not api_key.startswith("gsk_"):
    raise SystemExit("The GROQ_API_KEY in .env does not have the expected gsk_ prefix.")

try:
    from groq import Groq

    Groq(api_key=api_key).models.list()
except Exception as error:
    raise SystemExit(f"Groq key check failed: {type(error).__name__}: {error}") from error

print("Groq key check passed. The key is valid and can access the Groq API.")
