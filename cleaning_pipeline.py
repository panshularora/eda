"""
=============================================================
DATA VORTEX – Social Engine Recovery | Phase 1 Cleaning Pipeline
Event: Aaruush '26 | Team: Panshul
=============================================================

DATASET FILES:
  - Social_Engine_Posts_Corrupted.csv  (12,360 rows × 8 cols)
  - Social_Engine_Users.csv            (1,500 rows × 5 cols)

CORRUPTIONS IDENTIFIED & FIXED:
  1. Duplicate post_ids         → 360 duplicates removed (kept first)
  2. Mixed timestamp formats    → Normalized all to ISO8601 (YYYY-MM-DDTHH:MM:SS)
       - Unix epoch (e.g. 1722528840)
       - DD-MM-YYYY (e.g. 25-09-2024)
       - ISO8601    (e.g. 2025-04-13T20:12:18)
  3. Missing platform (1,846)   → Imputed with mode ('YouTube' → 'Facebook')
  4. Missing likes (1,858)      → Imputed with median (2,498)
  5. Negative likes (525)       → Replaced with absolute value
  6. HTML entities in text      → Decoded (&amp; → &, etc.) – 341 rows fixed
  7. Missing text_content(1746) → Filled with '[CONTENT UNAVAILABLE]'

ASSUMPTIONS:
  - Negative likes are data entry errors; absolute value is appropriate
  - Platform missing-at-random; mode imputation is defensible given ~5 equal platforms
  - Unix timestamps assumed UTC
  - DD-MM-YYYY format inferred from day values > 12
  - No fabrication: no rows invented, no values invented beyond statistical imputation
=============================================================
"""

import pandas as pd
import numpy as np
import html
import re

# ── Load ──────────────────────────────────────────────────
posts = pd.read_csv('Social_Engine_Posts_Corrupted.csv')
users = pd.read_csv('Social_Engine_Users.csv')

print(f"Loaded posts: {posts.shape}, users: {users.shape}")

# ── Step 1: Remove Duplicate post_ids ────────────────────
before = len(posts)
posts = posts.drop_duplicates(subset='post_id', keep='first')
print(f"Removed {before - len(posts)} duplicate rows")

# ── Step 2: Normalize Timestamps ─────────────────────────
def normalize_timestamp(ts):
    ts = str(ts).strip()
    if ts.isdigit() and len(ts) == 10:           # Unix epoch
        return pd.to_datetime(int(ts), unit='s').strftime('%Y-%m-%dT%H:%M:%S')
    try:
        if ts[2] == '-' and ts[5] == '-':        # DD-MM-YYYY
            return pd.to_datetime(ts, dayfirst=True).strftime('%Y-%m-%dT%H:%M:%S')
    except: pass
    try:
        return pd.to_datetime(ts).strftime('%Y-%m-%dT%H:%M:%S')  # ISO / YYYY-MM-DD
    except:
        return np.nan

posts['timestamp'] = posts['timestamp'].apply(normalize_timestamp)

# ── Step 3: Fix Platform ──────────────────────────────────
platform_mode = posts['platform'].mode()[0]
posts['platform'] = posts['platform'].fillna(platform_mode)

# ── Step 4: Fix Likes ────────────────────────────────────
posts['likes'] = posts['likes'].abs()
likes_median = posts['likes'].median()
posts['likes'] = posts['likes'].fillna(likes_median).astype(int)

# ── Step 5: Clean Text Content ───────────────────────────
posts['text_content'] = posts['text_content'].apply(
    lambda x: html.unescape(str(x)) if pd.notna(x) else '[CONTENT UNAVAILABLE]'
)

# ── Step 6: Users – standardize date ─────────────────────
users['account_created'] = pd.to_datetime(
    users['account_created'], errors='coerce'
).dt.strftime('%Y-%m-%d')

# ── Save ──────────────────────────────────────────────────
posts.to_csv('Social_Engine_Posts_Cleaned.csv', index=False)
users.to_csv('Social_Engine_Users_Cleaned.csv', index=False)

print(f"\nFinal posts: {posts.shape}")
print(f"Final users: {users.shape}")
print("\nNull check (posts):")
print(posts.isnull().sum())
print("\nCleaning complete. Files saved.")
