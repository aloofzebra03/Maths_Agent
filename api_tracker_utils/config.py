import os
from dotenv import load_dotenv

load_dotenv(dotenv_path = ".env", override=True)

AVAILABLE_MODELS = [
    # "gemini-2.5-flash",
    # "gemma-3-27b-it",
    "gemma-4-26b-a4b-it",
    
]

# Default model
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

# Rate limits per model (requests per minute and per day)
# RATE_LIMITS = {
#     "gemini-2.5-flash": {
#         "per_minute": 5,
#         "per_day": 20
#     },
#     "gemini-2.5-flash-lite": {
#         "per_minute": 10,
#         "per_day": 20
#     },  
#     "gemma-3-27b-it": {
#         "per_minute": 30,
#         "per_day": 14400
#     }
# }

RATE_LIMITS = {
    "gemini-2.5-flash": {
        "per_minute": 2,
        "per_day": 20
    },
    "gemini-2.5-flash-lite": {
        "per_minute": 2,
        "per_day": 20
    },  
    "gemma-3-27b-it": {
        "per_minute": 2,
        "per_day": 14400
    }
}


# API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
