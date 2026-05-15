import os
from pathlib import Path

# Config
CONFIG_FILE = Path('config.yml')
# Read API keys from environment to avoid committing secrets.
PERSPECTIVE_API_KEY = os.environ.get('PERSPECTIVE_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Perspective API

PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
