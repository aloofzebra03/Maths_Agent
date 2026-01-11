"""
Configuration settings for the Math Tutoring Agent.

Contains thresholds, limits, and other configurable parameters.
"""

# ============================================================================
# Scoring Thresholds
# ============================================================================

# Ta/Tu thresholds for mode routing
TA_THRESHOLD_HIGH = 0.6  # Approach quality threshold for coach mode
TU_THRESHOLD_HIGH = 0.6  # Understanding threshold for coach mode

# Low thresholds for scaffold mode (both must be below this)
TA_THRESHOLD_LOW = 0.6
TU_THRESHOLD_LOW = 0.6


# ============================================================================
# Attempt Limits
# ============================================================================

# Maximum number of reflective nudges in coach mode before downgrading to guided
MAX_COACH_NUDGES = 3

# Maximum retries on a single scaffold step before giving the answer
MAX_SCAFFOLD_RETRIES = 2

# Maximum total steps expected (used for validation)
MAX_TOTAL_STEPS = 20

# Maximum times we can teach the same concept to a student
MAX_CONCEPT_VISITS_PER_CONCEPT = 2

# Maximum interactions allowed per concept teaching session
MAX_CONCEPT_INTERACTIONS = 3


# ============================================================================
# LLM Configuration
# ============================================================================

# Default model to use (can be overridden by tracker)
DEFAULT_MODEL = "gemini-2.0-flash-exp"

# Temperature for LLM calls
LLM_TEMPERATURE = 0.5


# ============================================================================
# File Paths
# ============================================================================

# Directory containing problem JSON files (relative to project root)
PROBLEMS_DIR = "problems_json"
