"""
API Tracker - Thread-safe model usage tracking using atomic list operations.

This module tracks which models have been called and how many times, using
Python lists which provide atomic append() operations (thread-safe without locks).

Key Design Points:
- list.append() is atomic (thread-safe) according to CONCURRENCY_AND_THREAD_SAFETY.md and official Python docs
- len(list) is atomic (safe to read)
- We use separate lists for each model to avoid complex operations
- No locks needed for tracking!

Trade-offs:
- Memory: Each call creates a new entry (but we can clean up periodically)
- Slightly more memory than a counter dict, but avoids race conditions
- Perfect for moderate traffic scenarios
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from api_tracker_utils.config import AVAILABLE_MODELS, DEFAULT_MODEL, RATE_LIMITS
from api_tracker_utils.error import MinuteLimitExhaustedError, DayLimitExhaustedError
import os
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

# def get_available_api_keys() -> List[str]:
#     """Get all 7 available Google API keys from environment."""
#     api_keys = []
#     for i in range(1, 9):  # GOOGLE_API_KEY_1 through GOOGLE_API_KEY_8
#         key = os.getenv(f"GOOGLE_API_KEY_{i}")
#         if key:
#             api_keys.append(key)
    
#     if not api_keys:
#         raise RuntimeError("No Google API keys found. Please set GOOGLE_API_KEY_1 through GOOGLE_API_KEY_7 in .env file")
    
#     return api_keys

import os
from typing import List

def get_available_api_keys() -> List[str]:
    """Get all available Google API keys with prefix GOOGLE_API_KEY_ from environment."""
    
    api_keys = [
        value for key, value in os.environ.items()
        if key.startswith("GOOGLE_API_KEY_") and value
    ]

    print(f"Found {len(api_keys)} Google API keys")

    if not api_keys:
        raise RuntimeError(
            "No Google API keys found. Please set environment variables with prefix GOOGLE_API_KEY_ in .env file"
        )

    return api_keys


# ============================================================================
# THREAD-SAFE TRACKER USING ATOMIC LIST OPERATIONS
# ============================================================================

class ModelUsageTracker:
    """
    Thread-safe tracker using atomic list.append() operations.
    
    Tracks usage per API key per model. Since list.append() is atomic,
    multiple threads can safely append to these lists simultaneously.
    
    Structure: {api_key_suffix: {model_name: [(timestamp, success), ...]}}
    """
    
    def __init__(self):
        # Track calls per API per Model - nested dict structure
        # Outer key: api_key_suffix, Inner key: model_name, Value: list of (timestamp, success) tuples
        self._api_model_calls: Dict[str, Dict[str, List[tuple]]] = {}
        
        # Rate limit tracking per API per Model
        self._api_model_failures: Dict[str, Dict[str, List[datetime]]] = {}
        
        # Per-minute tracking: {api_suffix: {model: [timestamp1, timestamp2, ...]}}
        # Only timestamps, filtered on read to last 60 seconds
        self._minute_calls: Dict[str, Dict[str, List[datetime]]] = {}
        
        # Per-day tracking: {api_suffix: {model: [timestamp1, timestamp2, ...]}}
        # Only timestamps, filtered on read to current day
        self._daily_calls: Dict[str, Dict[str, List[datetime]]] = {}
        
        # Email notification tracking
        self._last_day_limit_email_sent: Optional[datetime] = None
        self._email_cooldown_hours = 6  # Don't spam, wait 6 hours between emails
        
        print(f"[TRACKER] Initialized for models: {AVAILABLE_MODELS}")
        print(f"[TRACKER] Will track usage across 7 API keys")
        print(f"[TRACKER] Rate limits: {RATE_LIMITS}")
    
    def track_call(self, api_key: str, model_name: str):
        """
        Record an API call for a specific API key and model.
        
        This is thread-safe because list.append() is atomic!
        
        Args:
            api_key: Google API key used for the call
            model_name: Name of the model that was called
        """
        timestamp = datetime.now()
        api_key_suffix = api_key[-6:]  # Last 6 chars for identification
        
        # Initialize nested structure if first time seeing this API
        if api_key_suffix not in self._api_model_calls:
            self._api_model_calls[api_key_suffix] = {}
            self._api_model_failures[api_key_suffix] = {}
        
        # Initialize model list if first time seeing this model for this API
        if model_name not in self._api_model_calls[api_key_suffix]:
            self._api_model_calls[api_key_suffix][model_name] = []
            self._api_model_failures[api_key_suffix][model_name] = []
        
        # Initialize minute tracking
        if api_key_suffix not in self._minute_calls:
            self._minute_calls[api_key_suffix] = {}
        if model_name not in self._minute_calls[api_key_suffix]:
            self._minute_calls[api_key_suffix][model_name] = []
        
        # Initialize daily tracking
        if api_key_suffix not in self._daily_calls:
            self._daily_calls[api_key_suffix] = {}
        if model_name not in self._daily_calls[api_key_suffix]:
            self._daily_calls[api_key_suffix][model_name] = []
        
        # Atomic operations - thread-safe!
        self._api_model_calls[api_key_suffix][model_name].append(timestamp)
        self._minute_calls[api_key_suffix][model_name].append(timestamp)
        self._daily_calls[api_key_suffix][model_name].append(timestamp)
        
        # if not success:
        #     # Also track failures separately
        #     self._api_model_failures[api_key_suffix][model_name].append(timestamp)
        
        print(f"[TRACKER] Recorded API ...{api_key_suffix} + {model_name} call at {timestamp.isoformat()}")
    
    def get_call_count(self, api_key_suffix: str, model_name: str) -> int:
        """
        Get total number of calls for a specific API-model combination.
        
        len() is atomic - thread-safe to read!
        """
        if api_key_suffix not in self._api_model_calls:
            return 0
        return len(self._api_model_calls[api_key_suffix].get(model_name, []))
    
    def get_failure_count(self, api_key_suffix: str, model_name: str) -> int:
        """Get number of failed calls for a specific API-model combination."""
        if api_key_suffix not in self._api_model_failures:
            return 0
        return len(self._api_model_failures[api_key_suffix].get(model_name, []))
    
    def get_minute_usage(self, api_key_suffix: str, model_name: str) -> int:
        """
        Get number of calls in the last 60 seconds for a specific API-model pair.
        Uses filtering (no reset needed) - thread-safe!
        """
        if api_key_suffix not in self._minute_calls:
            return 0
        
        all_calls = self._minute_calls[api_key_suffix].get(model_name, [])
        if not all_calls:
            return 0
        
        # Filter to only recent calls (creates new list - thread-safe)
        cutoff = datetime.now() - timedelta(seconds=60)
        recent_calls = [ts for ts in all_calls if ts > cutoff]
        
        return len(recent_calls)
    
    def get_daily_usage(self, api_key_suffix: str, model_name: str) -> int:
        """
        Get number of calls today for a specific API-model pair.
        Uses filtering (no reset needed) - thread-safe!
        """
        if api_key_suffix not in self._daily_calls:
            return 0
        
        all_calls = self._daily_calls[api_key_suffix].get(model_name, [])
        if not all_calls:
            return 0
        
        # Filter to only today's calls (creates new list - thread-safe)
        today = datetime.now().date()
        today_calls = [ts for ts in all_calls if ts.date() == today]
        
        return len(today_calls)
    
    def is_within_limits(self, api_key_suffix: str, model_name: str) -> bool:
        """
        Check if an API-model pair is within both per-minute and per-day rate limits.
        Thread-safe as it only reads data.
        
        Returns:
            True if within limits, False otherwise
        """
        # Get rate limits for this model
        if model_name not in RATE_LIMITS:
            print(f"[TRACKER] Warning: No rate limits defined for {model_name}, allowing")
            return True
        
        limits = RATE_LIMITS[model_name]
        minute_limit = limits["per_minute"]
        day_limit = limits["per_day"]
        
        # Check current usage
        minute_usage = self.get_minute_usage(api_key_suffix, model_name)
        daily_usage = self.get_daily_usage(api_key_suffix, model_name)
        
        within_minute = minute_usage < minute_limit
        within_day = daily_usage < day_limit
        return within_minute and within_day
    
    def _send_day_limit_exhaustion_email(self):
        """Send email notification when all API-model day limits are exhausted."""
        
        # Check if we recently sent an email (avoid spam)
        if self._last_day_limit_email_sent:
            hours_since_last = (datetime.now() - self._last_day_limit_email_sent).total_seconds() / 3600
            if hours_since_last < self._email_cooldown_hours:
                print(f"[EMAIL] Skipping email - last sent {hours_since_last:.1f} hours ago (cooldown: {self._email_cooldown_hours}h)")
                return
        
        try:
            # SMTP Configuration
            SMTP_SERVER = "cp1.dnspark.in"
            SMTP_PORT = 587
            SENDER_EMAIL = os.getenv("SENDER_EMAIL")
            SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
            RECIPIENTS = [
                "aryansinghalaattss@gmail.com",
                "mail2anuragmn@gmail.com"
            ]
            # ------------------------------------------------------------
            # ✅ BUILD MODEL LIMIT SECTION DYNAMICALLY FROM CONFIG
            # ------------------------------------------------------------
            model_limit_lines = []
            for model in AVAILABLE_MODELS:
                limits = RATE_LIMITS.get(model, {})
                per_day = limits.get("per_day", "N/A")
                per_minute = limits.get("per_minute", "N/A")
                model_limit_lines.append(
                    f"- {model}: {per_day}/day, {per_minute}/minute"
                )

            model_limits_text = "\n".join(model_limit_lines)

            SUBJECT = "⚠️ API Tracker: All Day Limits Exhausted"

            BODY = f"""Hello,

            All API-model combinations have exhausted their DAILY rate limits.

            Summary:
            - Total API keys: {len(get_available_api_keys())}
            - Total models: {len(AVAILABLE_MODELS)}
            - Total combinations: {len(get_available_api_keys()) * len(AVAILABLE_MODELS)}

            Configured model limits:
            {model_limits_text}

            The tracker will resume normal operation after daily limits reset.

            Regards,
            API Tracker System

            ---
            Sent at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            # Build Email
            msg = MIMEMultipart()
            msg["From"] = SENDER_EMAIL
            msg["To"] = ", ".join(RECIPIENTS)
            msg["Subject"] = SUBJECT
            msg.attach(MIMEText(BODY, "plain"))
            
            # Send Email
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENTS, msg.as_string())
            server.quit()
            
            # Update timestamp
            self._last_day_limit_email_sent = datetime.now()
            print(f"[EMAIL] ✅ Day limit exhaustion email sent successfully to {len(RECIPIENTS)} recipients")
            
        except Exception as e:
            print(f"[EMAIL] ❌ Failed to send email: {e}")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Get statistics for all API-model combinations.
        
        Returns:
            Nested dict: {api_key_suffix: {model_name: {stats}}}
        """
        stats = {}
        for api_suffix, models in self._api_model_calls.items():
            stats[api_suffix] = {}
            for model, calls in models.items():
                # failures = len(self._api_model_failures.get(api_suffix, {}).get(model, []))
                total = len(calls)
                
                # Add rate limit usage info
                minute_usage = self.get_minute_usage(api_suffix, model)
                daily_usage = self.get_daily_usage(api_suffix, model)
                
                limits = RATE_LIMITS.get(model, {"per_minute": 0, "per_day": 0})
                
                stats[api_suffix][model] = {
                    "total_calls": total,
                    # "successful_calls": total - failures,
                    # "failed_calls": failures,
                    "minute_usage": minute_usage,
                    "minute_limit": limits["per_minute"],
                    "daily_usage": daily_usage,
                    "daily_limit": limits["per_day"],
                    "within_limits": self.is_within_limits(api_suffix, model),
                }
        return stats
    
    def get_least_used_api_model_pair(self) -> Tuple[str, str]:
        """
        Get the API-model combination with the least usage that is within rate limits.
        
        Selection algorithm:
        1. Filter to only pairs within both per-minute and per-day limits
        2. Sort by total usage (ascending)
        3. Randomly pick from top 3 least-used pairs to spread load across concurrent requests
        4. If no pairs are available (all hit limits), use least-used anyway
        
        Returns:
            Tuple of (api_key, model_name)
        """
        available_keys = get_available_api_keys()
        random.shuffle(available_keys)  # Shuffle to avoid bias
        
        # Collect all valid pairs (within limits) with their usage
        valid_pairs = []
        all_pairs = []  # Fallback if no valid pairs
        
        for api_key in available_keys:
            api_suffix = api_key[-6:]
            for model in AVAILABLE_MODELS:
                total_usage = self.get_call_count(api_suffix, model)
                within_limits = self.is_within_limits(api_suffix, model)
                
                pair_info = (api_key, model, total_usage)
                all_pairs.append(pair_info)
                
                if within_limits:
                    valid_pairs.append(pair_info)
        
        # Choose from valid pairs, or fallback to all pairs if none available
        # candidates = valid_pairs if valid_pairs else all_pairs
        candidates = valid_pairs
        if not valid_pairs:
            # Check if ALL pairs have exhausted DAY limits (not just minute limits)
            all_day_exhausted = True
            all_minute_exhausted = True
            for api_key in available_keys:
                api_suffix = api_key[-6:]
                for model in AVAILABLE_MODELS:
                    if all_day_exhausted:
                        daily_usage = self.get_daily_usage(api_suffix, model)
                        day_limit = RATE_LIMITS.get(model, {}).get("per_day", 0)
                        if daily_usage < day_limit:
                            all_day_exhausted = False
                    if all_minute_exhausted:
                        min_usage = self.get_minute_usage(api_suffix,model)
                        min_limit = RATE_LIMITS.get(model, {}).get("per_minute", 0)
                        if min_usage < min_limit:
                            all_minute_exhausted = False
                if not all_day_exhausted and not all_minute_exhausted:
                    break
            
            # Send email if all day limits are exhausted
            if all_day_exhausted:
                print("[TRACKER] ⚠️ ALL API-model combinations have exhausted their DAY limits!")
                self._send_day_limit_exhaustion_email()
                raise DayLimitExhaustedError()
            
            elif all_minute_exhausted:
                print("[TRACKER] ⚠️ ALL API-model combinations have exhausted their MINUTE limits!")
                raise MinuteLimitExhaustedError()
            
            else:
                raise RuntimeError("[TRACKER] No valid API-model pairs found, but not all limits are exhausted? Check code")
            
        else:
            print(f"[TRACKER] Found {len(valid_pairs)} valid API-model pairs within limits")
    
        
        # Sort by usage (ascending)
        candidates.sort(key=lambda x: x[2])  # Sort by total_usage
        
        # Pick randomly from top 3 to better distribute concurrent requests
        top_n = min(3, len(candidates))
        selected_api, selected_model, usage = random.choice(candidates[:top_n])
        
        if not valid_pairs:
            print(f"[TRACKER] Warning: All pairs at limits, using least-used anyway: ...{selected_api[-6:]} + {selected_model}")
        else:
            print(f"[TRACKER] Selected ...{selected_api[-6:]} + {selected_model} from top {top_n} (usage: {usage})")
        
        return selected_api, selected_model
    
    def is_api_model_available(self, api_key: str, model_name: str, failure_threshold: int = 3) -> bool:
        """
        Placeholder.Will implement later
        
        Args:
            api_key: API key to check
            model_name: Model to check
            failure_threshold: Number of recent failures to consider rate-limited
        """
        # api_suffix = api_key[-6:]
        # recent_failures = self.get_failure_count(api_suffix, model_name)
        # return recent_failures < failure_threshold
    
    def get_best_api_key_for_model(self, model_name: str) -> str:
        """
        Get the best API key for a specific model.
        
        Selection algorithm:
        1. Validate model exists in AVAILABLE_MODELS
        2. Filter to only API keys within rate limits for this specific model
        3. Sort by usage for this model (ascending)
        4. Randomly pick from top 3 least-used keys to spread load
        5. If no keys are available (all hit limits), raise appropriate error
        
        Args:
            model_name: Name of the model to get an API key for
            
        Returns:
            The best available API key for the specified model
            
        Raises:
            ValueError: If model_name is not in AVAILABLE_MODELS
            MinuteLimitExhaustedError: If all API keys exhausted minute limits for this model
            DayLimitExhaustedError: If all API keys exhausted day limits for this model
        """
        # Validate model
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not found in AVAILABLE_MODELS: {AVAILABLE_MODELS}")
        
        available_keys = get_available_api_keys()
        random.shuffle(available_keys)  # Shuffle to avoid bias
        
        # Collect valid API keys (within limits) with their usage for this specific model
        valid_keys = []
        all_keys = []  # Fallback if no valid keys
        
        for api_key in available_keys:
            api_suffix = api_key[-6:]
            total_usage = self.get_call_count(api_suffix, model_name)
            within_limits = self.is_within_limits(api_suffix, model_name)
            
            key_info = (api_key, total_usage)
            all_keys.append(key_info)
            
            if within_limits:
                valid_keys.append(key_info)
        
        # Check if we have valid keys
        if not valid_keys:
            # Check if ALL keys have exhausted DAY limits or MINUTE limits for this model
            all_day_exhausted = True
            all_minute_exhausted = True
            
            for api_key in available_keys:
                api_suffix = api_key[-6:]
                
                if all_day_exhausted:
                    daily_usage = self.get_daily_usage(api_suffix, model_name)
                    day_limit = RATE_LIMITS.get(model_name, {}).get("per_day", 0)
                    if daily_usage < day_limit:
                        all_day_exhausted = False
                
                if all_minute_exhausted:
                    min_usage = self.get_minute_usage(api_suffix, model_name)
                    min_limit = RATE_LIMITS.get(model_name, {}).get("per_minute", 0)
                    if min_usage < min_limit:
                        all_minute_exhausted = False
                
                if not all_day_exhausted and not all_minute_exhausted:
                    break
            
            # Raise appropriate error
            if all_day_exhausted:
                print(f"[TRACKER] ⚠️ ALL API keys have exhausted their DAY limits for model '{model_name}'!")
                raise DayLimitExhaustedError(f"All API keys exhausted day limits for model '{model_name}'")
            elif all_minute_exhausted:
                print(f"[TRACKER] ⚠️ ALL API keys have exhausted their MINUTE limits for model '{model_name}'!")
                raise MinuteLimitExhaustedError(f"All API keys exhausted minute limits for model '{model_name}'")
            else:
                raise RuntimeError(f"[TRACKER] No valid API keys found for '{model_name}', but not all limits are exhausted? Check code")
        else:
            print(f"[TRACKER] Found {len(valid_keys)} valid API keys within limits for model '{model_name}'")
        
        # Sort by usage (ascending)
        valid_keys.sort(key=lambda x: x[1])  # Sort by total_usage
        
        # Pick randomly from top 3 to better distribute concurrent requests
        top_n = min(3, len(valid_keys))
        selected_api, usage = random.choice(valid_keys[:top_n])
        
        print(f"[TRACKER] Selected ...{selected_api[-6:]} for {model_name} from top {top_n} (usage: {usage})")
        
        return selected_api
    
    def get_next_available_api_model_pair(self) -> Tuple[str, str]:
        """
        Get the next available API-model pair to use.
        
        For now, just returns the least used API-model pair.
        Can be extended later to filter out rate-limited combinations.
        
        Returns:
            Tuple of (api_key, model_name)
        """
        return self.get_least_used_api_model_pair()
    
    def reset_stats(self):
        """Reset all tracking data (useful for testing)."""
        self._api_model_calls = {}
        self._api_model_failures = {}
        self._minute_calls = {}
        self._daily_calls = {}
        print("[TRACKER] Reset all stats")


# ============================================================================
# GLOBAL TRACKER INSTANCE (Singleton)
# ============================================================================

_tracker = ModelUsageTracker()


def track_model_call(api_key: str, model_name: str):
    """Track an API call to a specific API key and model."""
    _tracker.track_call(api_key, model_name)


def get_next_available_api_model_pair() -> Tuple[str, str]:
    """Get the next available API-model pair to use."""
    return _tracker.get_next_available_api_model_pair()


def get_tracker_stats() -> Dict[str, Dict[str, Dict[str, int]]]:
    """Get statistics for all API-model combinations."""
    return _tracker.get_all_stats()


def reset_tracker():
    """Reset the tracker (for testing)."""
    _tracker.reset_stats()


def get_api_model_call_count(api_key_suffix: str, model_name: str) -> int:
    """Get call count for a specific API-model combination."""
    return _tracker.get_call_count(api_key_suffix, model_name)


def get_best_api_key_for_model(model_name: str) -> str:
    """
    Get the best API key for a specific model.
    
    This function selects the API key with the least usage for the given model
    while respecting rate limits (both per-minute and per-day).
    
    Args:
        model_name: Name of the model (must be in AVAILABLE_MODELS)
        
    Returns:
        The best available API key for the specified model
        
    Raises:
        ValueError: If model_name is not in AVAILABLE_MODELS
        MinuteLimitExhaustedError: If all API keys exhausted minute limits
        DayLimitExhaustedError: If all API keys exhausted day limits
        
    Example:
        >>> api_key = get_best_api_key_for_model("gemini-2.5-flash")
        >>> # Use api_key with the specified model
    """
    return _tracker.get_best_api_key_for_model(model_name)
