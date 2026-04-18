#!/usr/bin/env python3
"""
Auto-labeling script for Amazon review PII annotations (v10.4 protocol).

Model  : gemma-4-31b-it with thinking/reasoning (Gemini API)
Keys   : Three free-tier API keys with automatic failover
Resume : Checkpointed after every row — safe to Ctrl+C and restart

Usage:
    Set GEMINI_KEY_1, GEMINI_KEY_2, GEMINI_KEY_3 as environment variables,
    then run:  python autolabel.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
INPUT_CSV       = BASE_DIR / "amazon_mined_shuffled.csv"
OUTPUT_CSV      = BASE_DIR / "amazon_labeled.csv"
CHECKPOINT_FILE = BASE_DIR / "autolabel_checkpoint.json"
LOG_FILE        = BASE_DIR / "autolabel.log"

# ── Model ─────────────────────────────────────────────────────────────────────

# Verify the exact ID in Google AI Studio if this returns a 404.
MODEL_NAME = "gemma-4-31b-it"

# ── API Keys ──────────────────────────────────────────────────────────────────

API_KEYS: list[str] = [
    os.getenv("GEMINI_KEY_1", ""),
    os.getenv("GEMINI_KEY_2", ""),
    os.getenv("GEMINI_KEY_3", ""),
]

RPM_LIMIT = 15                        # Gemma 4 free-tier cap on Google AI Studio
MIN_REQUEST_INTERVAL = 60 / RPM_LIMIT  # 4.0 s between calls on the same key

KEY_RETRY_PAUSE = 5    # seconds to wait before switching to next key
GENERAL_RETRY_PAUSE = 8  # seconds to wait on transient errors

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise PII annotation expert. Label privacy-sensitive spans in Amazon product reviews following PII Annotation Protocol v10.4.

## I. Minor Indication (Full Narrative Coreference)
Minors are high-risk. Label the identity anchor AND all gendered pronouns (he, she, him, her, his) that refer back to that specific child.

Anchor priority:
1. Tier 1 (Highest) — Numerical: age, clothes size, height (e.g. "5yo", "size 6T", "age 10")
2. Tier 2 — Educational: "kindergarten", "3rd grade"
3. Tier 3 (Fallback) — Behavioral: "stroller", "water balloons"
4. Infant Exception (Override): always label baby, infant, newborn

Coreference: after labeling an anchor, also label every gendered pronoun in the text referring to that child. Exclude non-gendered pronouns. Do NOT include "the" or "a" before a span.

## II. Medical Condition (Noun-Only)
Label condition/diagnosis spans only. Never label pronouns.
- P1 Diagnosis: always — e.g. ichthyosis, diabetes, UTI
- P2 Proxies: always — e.g. urologist, insulin, Rx
- P3 Symptoms: only if explicitly chronic — e.g. "chronic joint pain", "always shedding"

## III. Author Gender (Noun-Only)
Label anchors and proxies for the author's gender. Never label first-person pronouns (I, me, my, she, her).
- Explicit Noun: always — e.g. mom, wife, husband, father
- Bio/Physio: label only if author is the subject — e.g. breastfeeding, postpartum
- Clothing/Gear: label only if author is the user — e.g. bralette, beard oil
- Ownership Rule: skip Tier 2 & 3 if the item is a gift for someone else.

## Labels
Use exactly these strings:
- "minor_anchor"    — noun identifying a minor
- "minor_pronoun"   — gendered pronoun referring back to a labeled minor
- "medical"         — medical condition, diagnosis, or proxy
- "gender_noun"     — gender identity noun of the author
- "gender_bio"      — gendered bio/physio proxy for the author
- "gender_clothing" — gendered clothing/gear used by the author

## Output
Return ONLY a valid JSON array. Each element: {"span": "<exact text>", "label": "<label>"}
If no PII is found return: []
No markdown, no explanation, no text outside the JSON array."""


def make_user_prompt(text: str) -> str:
    return (
        f"Review:\n{text}\n\n"
        "Return JSON only."
    )


# ── Key manager ───────────────────────────────────────────────────────────────

class KeyManager:
    def __init__(self, keys: list[str], pre_exhausted: set[int] = None):
        self.keys = [k for k in keys if k]
        if not self.keys:
            raise RuntimeError("No API keys provided. Set GEMINI_KEY_1/2/3.")
        self.exhausted: set[int] = pre_exhausted or set()
        self._current = 0
        # Tracks the last time each key index was used, for rate-limiting.
        self._last_used: dict[int, float] = {}

    @property
    def active_key(self) -> str | None:
        for offset in range(len(self.keys)):
            idx = (self._current + offset) % len(self.keys)
            if idx not in self.exhausted:
                self._current = idx
                return self.keys[idx]
        return None

    def wait_for_key(self, key: str) -> None:
        """Block until the per-key rate limit (15 RPM) allows another request."""
        try:
            idx = self.keys.index(key)
        except ValueError:
            return
        elapsed = time.monotonic() - self._last_used.get(idx, 0.0)
        wait = MIN_REQUEST_INTERVAL - elapsed
        if wait > 0:
            log.debug(f"Rate-limit: sleeping {wait:.2f}s for key #{idx + 1}")
            time.sleep(wait)
        self._last_used[idx] = time.monotonic()

    def mark_exhausted(self, key: str) -> None:
        try:
            idx = self.keys.index(key)
        except ValueError:
            return
        self.exhausted.add(idx)
        remaining = len(self.keys) - len(self.exhausted)
        log.warning(f"Key #{idx + 1} quota exhausted — {remaining} key(s) remaining.")


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"Checkpoint loaded: {len(data['results'])} rows already labeled.")
        return data
    return {"results": {}, "exhausted_keys": []}


def save_checkpoint(results: dict, exhausted_keys: list[int]) -> None:
    payload = {"results": results, "exhausted_keys": exhausted_keys}
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(CHECKPOINT_FILE)  # atomic replace


# ── Labeling ──────────────────────────────────────────────────────────────────

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "resource exhausted" in msg


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def label_row(text: str, key_manager: KeyManager) -> list[dict] | None:
    """
    Returns a list of entity dicts, an empty list if none found, or None when
    all API keys are exhausted (caller should stop and save).
    """
    prompt = make_user_prompt(text)

    while True:
        key = key_manager.active_key
        if key is None:
            return None

        key_manager.wait_for_key(key)
        client = genai.Client(api_key=key)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            raw = _strip_fences(response.text.strip())
            return json.loads(raw)

        except json.JSONDecodeError as e:
            raw_preview = response.text[:200] if "response" in dir() else "N/A"
            log.warning(f"JSON parse error ({e}). Raw: {raw_preview!r} — skipping row.")
            return []

        except Exception as e:
            if _is_quota_error(e):
                key_manager.mark_exhausted(key)
                time.sleep(KEY_RETRY_PAUSE)
                # loop continues with next key
            else:
                log.warning(f"API error: {e}. Retrying in {GENERAL_RETRY_PAUSE}s…")
                time.sleep(GENERAL_RETRY_PAUSE)


# ── Entity splitting ──────────────────────────────────────────────────────────

_LABEL_TO_COLUMN = {
    "minor_anchor":    "minor_children",
    "minor_pronoun":   "minor_children",
    "medical":         "medical_condition",
    "gender_noun":     "gender_indication",
    "gender_bio":      "gender_indication",
    "gender_clothing": "gender_indication",
}

ENTITY_COLUMNS = ["minor_children", "medical_condition", "gender_indication"]


def split_entities(entities: list[dict] | None) -> dict[str, str]:
    """
    Partition a flat entity list into per-category JSON strings, one per column.
    Returns a dict mapping column name → JSON string (or None if unlabeled).
    """
    if entities is None:
        return {col: None for col in ENTITY_COLUMNS}
    buckets: dict[str, list[dict]] = {col: [] for col in ENTITY_COLUMNS}
    for ent in entities:
        col = _LABEL_TO_COLUMN.get(ent.get("label", ""))
        if col:
            buckets[col].append(ent)
    return {
        col: json.dumps(spans, ensure_ascii=False) if spans else "[]"
        for col, spans in buckets.items()
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    checkpoint = load_checkpoint()
    results: dict = checkpoint["results"]
    pre_exhausted = {int(i) for i in checkpoint.get("exhausted_keys", [])}

    key_manager = KeyManager(API_KEYS, pre_exhausted=pre_exhausted)

    total = len(df)
    done = len(results)
    log.info(f"Rows: {total} total | {done} done | {total - done} remaining")
    log.info(f"Model: {MODEL_NAME} | Rate limit: {RPM_LIMIT} RPM/key")

    def write_output() -> None:
        split = df["unique_id"].astype(str).map(lambda uid: split_entities(results.get(uid)))
        for col in ENTITY_COLUMNS:
            df[col] = split.map(lambda s, c=col: s[c])
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        log.info(f"Output saved → {OUTPUT_CSV}  ({done}/{total} rows labeled)")

    try:
        for _, row in df.iterrows():
            uid = str(row["unique_id"])
            if uid in results:
                continue

            if key_manager.active_key is None:
                log.error("All API keys exhausted. Save progress and exit.")
                break

            log.info(f"[{done + 1}/{total}] {uid}")
            entities = label_row(str(row["original_text"]), key_manager)

            if entities is None:
                log.error("Stopped: all keys exhausted mid-run.")
                break

            results[uid] = entities
            done += 1
            save_checkpoint(results, list(key_manager.exhausted))

            if done % 100 == 0:
                log.info(f"── Progress: {done}/{total} ({100 * done / total:.1f}%) ──")

    except KeyboardInterrupt:
        log.warning("Interrupted — saving partial results…")
    finally:
        write_output()


if __name__ == "__main__":
    main()
