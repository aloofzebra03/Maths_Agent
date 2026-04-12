# `api_tracker_utils` — Integration Guide

A practical guide for integrating this package into any Python project that calls Google Gemini / Gemma models across multiple API keys.

---

## Table of Contents

1. [How It Works (Mental Model)](#1-how-it-works-mental-model)
2. [Setup](#2-setup)
3. [Public API — What to Import](#3-public-api--what-to-import)
4. [The Two Selection Functions](#4-the-two-selection-functions)
5. [The Correct Integration Pattern](#5-the-correct-integration-pattern)
6. [Error Types and What They Mean](#6-error-types-and-what-they-mean)
7. [The Critical Rule — Never Swallow Limit Errors](#7-the-critical-rule--never-swallow-limit-errors)
8. [Handling Errors at the Entry Point](#8-handling-errors-at-the-entry-point)
9. [Adding Models and API Keys](#9-adding-models-and-api-keys)
10. [Thread Safety](#10-thread-safety)
11. [Common Mistakes](#11-common-mistakes)

---

## 1. How It Works (Mental Model)

You have **N API keys** and **M models**. Every unique `(api_key, model)` pair has its own per-minute and per-day counter. The tracker:

1. Scans all `N × M` pairs and finds the ones still within limits.
2. Sorts them by total usage (least-used first) to spread load evenly.
3. Randomly picks one from the top 3 least-used to avoid concurrent requests all landing on the same pair.
4. Returns that pair to you — it is your job to use it and record it.

**The tracker does not retry and does not call the LLM itself.** It only selects keys and counts calls. When every pair is over limits, it raises a typed exception and stops.

---

## 2. Setup

### 2.1 Environment Variables

Name your API keys with the prefix `GOOGLE_API_KEY_` followed by any suffix (numbers work well):

```
GOOGLE_API_KEY_1=AIza...
GOOGLE_API_KEY_2=AIza...
GOOGLE_API_KEY_3=AIza...
```

The tracker auto-discovers every variable that starts with `GOOGLE_API_KEY_`. There is no fixed maximum — add as many keys as you have. Variables without that exact prefix are ignored.

### 2.2 Rate Limits (`config.py`)

Edit `api_tracker_utils/config.py` to match your actual Google AI Studio / Gemini API plan:

```python
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-3-27b-it",
]

RATE_LIMITS = {
    "gemini-2.5-flash": {
        "per_minute": 5,    # requests per minute per key
        "per_day": 20       # requests per day per key
    },
    "gemini-2.5-flash-lite": {
        "per_minute": 10,
        "per_day": 20
    },
    "gemma-3-27b-it": {
        "per_minute": 30,
        "per_day": 14400
    }
}
```

Limits are **per key per model**. Total daily capacity for a model = `per_day × number_of_keys`.

> If a model is in `AVAILABLE_MODELS` but not in `RATE_LIMITS`, the tracker allows all requests for it (with a warning). Always add an entry to both.

---

## 3. Public API — What to Import

```python
from api_tracker_utils.tracker import (
    get_next_available_api_model_pair,  # Returns (api_key, model_name) — tracker picks both
    get_best_api_key_for_model,         # Returns api_key — you specify the model
    track_model_call,                   # Records that a call was made
    get_tracker_stats,                  # Returns current usage statistics
    reset_tracker,                      # Resets all counters (testing only)
)

from api_tracker_utils.error import (
    MinuteLimitExhaustedError,          # All keys/models hit per-minute limit
    DayLimitExhaustedError,             # All keys/models hit per-day limit
    APITrackerError,                    # Base class for both above
)

from api_tracker_utils.config import (
    AVAILABLE_MODELS,                   # List of model name strings
    RATE_LIMITS,                        # Dict of limits per model
    DEFAULT_MODEL,                      # Fallback model name
)
```

---

## 4. The Two Selection Functions

Choose one based on whether your code cares which model is used.

### `get_next_available_api_model_pair() → (api_key, model_name)`

Use when **you don't care which model** is chosen. The tracker picks the best `(key, model)` combination across all models and all keys.

```python
api_key, model = get_next_available_api_model_pair()
```

### `get_best_api_key_for_model(model_name) → api_key`

Use when **your code requires a specific model** (e.g., your prompt relies on a long context window or a specific capability).

```python
api_key = get_best_api_key_for_model("gemini-2.5-flash")
```

Raises `ValueError` if the model name is not in `AVAILABLE_MODELS`.

---

## 5. The Correct Integration Pattern

Regardless of which selection function you use, the sequence is always:

```
1. Select  →  get_best_api_key_for_model() or get_next_available_api_model_pair()
2. Track   →  track_model_call(api_key, model)
3. Invoke  →  llm.invoke(...)
```

**Full example:**

```python
from api_tracker_utils.tracker import get_best_api_key_for_model, track_model_call
from api_tracker_utils.error import MinuteLimitExhaustedError, DayLimitExhaustedError
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-2.5-flash"

def call_llm(messages):
    # Step 1 — select (may raise MinuteLimitExhaustedError or DayLimitExhaustedError)
    api_key = get_best_api_key_for_model(MODEL)

    # Step 2 — record BEFORE invoking
    track_model_call(api_key, MODEL)

    # Step 3 — invoke
    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key)
    return llm.invoke(messages)
```

---

## 6. Error Types and What They Mean

### `MinuteLimitExhaustedError`
- **Meaning:** Every `(key, model)` combination has used its full per-minute quota.
- **Duration:** Temporary — resolves within 60 seconds as the sliding window advances.
- **Attribute:** `retry_after_seconds` (default `60`).

### `DayLimitExhaustedError`
- **Meaning:** Every `(key, model)` combination has used its full per-day quota.
- **Duration:** Permanent until midnight (quota reset).
- **Side effect:** Sends an email alert (if SMTP is configured).
- **Attribute:** `message`.

### Inheritance chain

```
Exception
  └── RuntimeError
        └── APITrackerError
              ├── MinuteLimitExhaustedError
              └── DayLimitExhaustedError
```

Because both extend `RuntimeError` and thus `Exception`, a plain `except Exception` will silently catch them. This is the most common integration bug — see [Section 11](#11-common-mistakes).

---

## 7. The Critical Rule — Never Swallow Limit Errors

> **`MinuteLimitExhaustedError` and `DayLimitExhaustedError` must always propagate to the outermost handler. Never catch them silently.**

Any helper function that wraps a tracker call must re-raise these before its generic catch:

```python
# ✅ Correct
def my_llm_helper(messages):
    try:
        api_key = get_best_api_key_for_model("gemini-2.5-flash")
        track_model_call(api_key, "gemini-2.5-flash")
        return llm.invoke(messages)
    except (MinuteLimitExhaustedError, DayLimitExhaustedError):
        raise                    # Let it reach the entry point handler
    except Exception as e:
        log(e)
        return fallback_value    # Only for non-limit errors


# ❌ Wrong — swallows the limit error
def my_llm_helper(messages):
    try:
        api_key = get_best_api_key_for_model("gemini-2.5-flash")
        track_model_call(api_key, "gemini-2.5-flash")
        return llm.invoke(messages)
    except Exception as e:
        log(e)
        return fallback_value    # MinuteLimitExhaustedError silently disappears here
```

The same rule applies to every layer of your call stack — node functions, helper utilities, translation wrappers, anything. If any layer has a bare `except Exception` that returns a default value, limit errors vanish there and your entry point never sees them.

### Placement tip

The cleanest way to ensure tracker errors are never caught is to call the selection function **outside** the `try` block entirely:

```python
def my_llm_helper(messages):
    # Outside try — limit errors naturally escape to the caller
    api_key = get_best_api_key_for_model("gemini-2.5-flash")
    track_model_call(api_key, "gemini-2.5-flash")

    try:
        return llm.invoke(messages)
    except Exception as e:
        log(e)
        raise
```

---

## 8. Handling Errors at the Entry Point

Your topmost handler (API endpoint, CLI main, task runner) is where you decide what to do with exhaustion. Do it explicitly — do not let them fall into a generic 500/crash handler.

### FastAPI example

```python
from api_tracker_utils.error import MinuteLimitExhaustedError, DayLimitExhaustedError

@app.post("/generate")
def generate(request: Request):
    try:
        return do_generation(request)

    except MinuteLimitExhaustedError as e:
        # Temporary — tell client to retry
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_minute", "retry_after": e.retry_after_seconds}
        )

    except DayLimitExhaustedError as e:
        # Permanent until quota resets
        raise HTTPException(
            status_code=503,
            detail={"error": "rate_limit_day", "message": str(e)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Plain Python / CLI example

```python
from api_tracker_utils.error import MinuteLimitExhaustedError, DayLimitExhaustedError
import time

try:
    result = call_llm(messages)
except MinuteLimitExhaustedError as e:
    print(f"Rate limited. Retry in {e.retry_after_seconds}s.")
    time.sleep(e.retry_after_seconds)
except DayLimitExhaustedError as e:
    print(f"Daily quota exhausted: {e}")
    sys.exit(1)
```

---

## 9. Adding Models and API Keys

### New model

1. Add the model name string to `AVAILABLE_MODELS` in `config.py`.
2. Add a matching entry in `RATE_LIMITS` with `per_minute` and `per_day`.
3. Use the **exact same string** everywhere: in `get_best_api_key_for_model()`, in `track_model_call()`, and in your LLM constructor. A mismatch raises `ValueError` from step one and silently skips rate limiting in steps two and three.

### New API key

Add an environment variable with the `GOOGLE_API_KEY_` prefix:

```
GOOGLE_API_KEY_9=AIza...
```

No code changes needed. The tracker re-discovers keys on each call to `get_available_api_keys()`.

---

## 10. Thread Safety

The tracker is safe to use from multiple threads without any external locks. It uses Python's `list.append()` which is atomic under the GIL, and `len()` which is also atomic. Each `(key, model)` pair has its own list so there is no shared mutable state between different pairs.

It is designed for the single-process, multi-thread deployment model (e.g., a single `uvicorn` worker with a thread pool). It is **not** safe across multiple processes — each process maintains its own in-memory counters with no cross-process synchronization.

---

## 11. Common Mistakes

### Mistake 1 — Falling back to a default key on limit errors

```python
# ❌ Wrong
try:
    api_key = get_best_api_key_for_model(MODEL)
except Exception as e:
    api_key = os.getenv("GOOGLE_API_KEY")  # Hits the same quota — causes a real 429 from Google
```

When `MinuteLimitExhaustedError` is raised, it means **all** your keys are exhausted. Falling back to a single key just causes Google to return a raw 429, which is a worse error with no clear retry signal.

This is a **real and factual failure mode**: if your selection call (for example `get_best_api_key_for_model`) is wrapped by a generic `except Exception` and that block assigns a default key, you effectively bypass tracker intent after exhaustion. The next LLM call then fails with provider-side `429 RESOURCE_EXHAUSTED`, and your entry-point-specific exhaustion handling never gets triggered.

```python
# ✅ Correct
try:
    api_key = get_best_api_key_for_model(MODEL)
except (MinuteLimitExhaustedError, DayLimitExhaustedError):
    raise   # Propagate — your entry point handles it
except Exception as e:
    # Only for unexpected non-limit errors (e.g., env var misconfiguration)
    api_key = os.getenv("GOOGLE_API_KEY")
```

---

### Mistake 2 — Tracking after the call

```python
# ❌ Wrong
response = llm.invoke(messages)
track_model_call(api_key, model)   # Never runs if invoke() raises
```

The tracker counts **attempts**, not just successes. If you track after the call, failed calls go unrecorded and the rate-limit window appears emptier than it is, causing more 429s from Google.

```python
# ✅ Correct
track_model_call(api_key, model)   # Always runs
response = llm.invoke(messages)
```

---

### Mistake 3 — Mismatched model strings

```python
# ❌ Wrong
api_key = get_best_api_key_for_model("gemini-2.5-flash")  # Selects key for flash
track_model_call(api_key, "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", ...)  # Different model!
```

The tracker thinks you used `gemini-2.5-flash` but you actually used `gemini-2.5-pro`. The pro model's counter is never incremented, so it appears to have infinite quota and will get hammered until Google returns 429.

```python
# ✅ Correct — use one constant for all three
MODEL = "gemini-2.5-flash"
api_key = get_best_api_key_for_model(MODEL)
track_model_call(api_key, MODEL)
llm = ChatGoogleGenerativeAI(model=MODEL, ...)
```

---

### Mistake 4 — Not importing the error classes before catching them

```python
# ❌ Wrong — NameError at runtime
except (MinuteLimitExhaustedError, DayLimitExhaustedError):
    raise
```

```python
# ✅ Correct
from api_tracker_utils.error import MinuteLimitExhaustedError, DayLimitExhaustedError

except (MinuteLimitExhaustedError, DayLimitExhaustedError):
    raise
```

---

### Mistake 5 — Catching `APITrackerError` thinking it covers only non-critical errors

`APITrackerError` is the **base class** for both `MinuteLimitExhaustedError` and `DayLimitExhaustedError`. Catching it and swallowing is just as bad as catching `Exception`:

```python
# ❌ Wrong — still swallows limit errors
except APITrackerError as e:
    print(f"tracker issue: {e}")
    return default   # Limit errors silently disappear

# ✅ Correct — re-raise if catching the base
except APITrackerError:
    raise
```

---

### Mistake 6 — Calling `track_model_call` with a model not in `AVAILABLE_MODELS`

`track_model_call` accepts any string and records it silently. If you pass a model name that is not in `RATE_LIMITS`, the tracker has no limits for it and will never raise `MinuteLimitExhaustedError` or `DayLimitExhaustedError` for that model — the counter runs forever. Always keep `AVAILABLE_MODELS`, `RATE_LIMITS`, and the model strings in your code in sync.

---

### Mistake 7 — Calling the selection function twice: once inside a helper, once outside to retrieve the key "for tracking"

This pattern appears when a `get_llm()` helper calls the tracker internally to pick a key, and then the caller makes a **second** call to `get_best_api_key_for_model()` to find out which key was used:

```python
# ❌ Wrong

def get_llm():
    api_key = get_best_api_key_for_model(MODEL)   # Call #1 — picks key A
    return ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key)

def teacher_node(state):
    llm = get_llm()                                # Uses key A internally

    # Trying to retrieve the key that was "used" — but this is a NEW selection call
    used_api_key = None
    try:
        used_api_key = get_best_api_key_for_model(MODEL)  # Call #2 — may pick key B!
    except:
        pass                                       # Bare except swallows limit errors too

    track_model_call(used_api_key, MODEL)          # Tracks key B, but LLM used key A!
    response = llm.invoke(messages)
```

**Two problems:**

1. **Wrong key tracked.** Between call #1 and call #2, the tracker's state may have changed (especially under concurrent load). Call #2 can return a different key than what `get_llm()` actually used. You end up tracking a key that did not make the request.

2. **Bare `except: pass` swallows limit errors.** If call #2 raises `MinuteLimitExhaustedError`, the `except: pass` silently discards it, `used_api_key` stays `None`, and tracking is skipped entirely — the quota counter never gets incremented for the call that just happened.

**Fix:** Select the key once and thread it through explicitly, so the same key is used for construction, tracking, and anything else:

```python
# ✅ Correct

from api_tracker_utils.error import MinuteLimitExhaustedError, DayLimitExhaustedError

def get_llm_with_key():
    """Return both the LLM instance and the key that was selected."""
    api_key = get_best_api_key_for_model(MODEL)   # Single selection
    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key)
    return llm, api_key

def teacher_node(state):
    llm, api_key = get_llm_with_key()             # One call, key flows through

    track_model_call(api_key, MODEL)               # Correct key, before invoke
    response = llm.invoke(messages)
```

If refactoring `get_llm()` is not immediately possible, the minimum fix is to stop using a second tracker call and instead accept the key as a parameter:

```python
# ✅ Also acceptable — pass the key in, don't re-select

def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key)

def teacher_node(state):
    api_key = get_best_api_key_for_model(MODEL)   # Single selection
    llm = get_llm(api_key)
    track_model_call(api_key, MODEL)
    response = llm.invoke(messages)
```
