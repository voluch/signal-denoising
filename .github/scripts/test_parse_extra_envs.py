#!/usr/bin/env python3
"""Tests for parse_extra_envs() in runpod_manage_template.py"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runpod_manage_template import parse_extra_envs

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label, result, expected):
    if result == expected:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        print(f"         expected: {expected}")
        print(f"         got:      {result}")
    return result == expected


def run_case(title, input_str, expected_kvs):
    """expected_kvs: list of (key, value) tuples"""
    print(f"\n{'─'*60}")
    print(f"Case: {title}")
    print(f"Input: {input_str!r}")
    result = parse_extra_envs(input_str)
    result_kvs = [(d["key"], d["value"]) for d in result]
    ok = check("parsed pairs match", result_kvs, expected_kvs)

    # For any value that looks like JSON, verify json.loads works
    for key, value in result_kvs:
        if value.startswith("{"):
            try:
                parsed = json.loads(value)
                print(f"  {PASS}  json.loads({key}) → {parsed}")
            except Exception as e:
                print(f"  {FAIL}  json.loads({key}) failed: {e}  (value={value!r})")
                ok = False
    return ok


failures = 0

# ── Case 1: simple space-separated ───────────────────────────────────────────
failures += not run_case(
    "simple space-separated",
    "KEY1=value1 KEY2=value2",
    [("KEY1", "value1"), ("KEY2", "value2")],
)

# ── Case 2: newline-separated ─────────────────────────────────────────────────
failures += not run_case(
    "newline-separated",
    "KEY1=value1\nKEY2=value2",
    [("KEY1", "value1"), ("KEY2", "value2")],
)

# ── Case 3: raw JSON object (no outer quotes) ─────────────────────────────────
failures += not run_case(
    "raw JSON object value",
    'MODEL_BATCH_SIZES={"transformer": 512, "unet": 2048, "vae": 16384}',
    [("MODEL_BATCH_SIZES", '{"transformer": 512, "unet": 2048, "vae": 16384}')],
)

# ── Case 4: quoted JSON with escaped inner quotes ─────────────────────────────
failures += not run_case(
    'double-quoted JSON with \\" escapes',
    r'MODEL_BATCH_SIZES="{\"transformer\": 512, \"unet\": 2048, \"vae\": 16384, \"resnet\": 8192, \"hybrid\": 16384, \"wavelet\": 512}"',
    [("MODEL_BATCH_SIZES", '{"transformer": 512, "unet": 2048, "vae": 16384, "resnet": 8192, "hybrid": 16384, "wavelet": 512}')],
)

# ── Case 5: JSON + extra simple var (newline-separated) ───────────────────────
# r-string preserves literal backslashes, matching what the GitHub Actions UI
# sends when the user types:  MODEL_BATCH_SIZES="{\"transformer\": 512, ...}"
failures += not run_case(
    'JSON + simple var on separate lines',
    r'MODEL_BATCH_SIZES="{\"transformer\": 512, \"unet\": 2048, \"vae\": 16384, \"resnet\": 8192, \"hybrid\": 16384, \"wavelet\": 512}"' + '\nEPOCHS=100',
    [
        ("MODEL_BATCH_SIZES", '{"transformer": 512, "unet": 2048, "vae": 16384, "resnet": 8192, "hybrid": 16384, "wavelet": 512}'),
        ("EPOCHS", "100"),
    ],
)

# ── Case 6: JSON + extra simple var (space-separated) ────────────────────────
failures += not run_case(
    'raw JSON + simple var space-separated',
    'MODEL_BATCH_SIZES={"transformer": 512, "unet": 2048} DEBUG=true',
    [
        ("MODEL_BATCH_SIZES", '{"transformer": 512, "unet": 2048}'),
        ("DEBUG", "true"),
    ],
)

# ── Case 7: empty string ──────────────────────────────────────────────────────
failures += not run_case(
    "empty string",
    "",
    [],
)

# ── Case 8: single-quoted value ───────────────────────────────────────────────
failures += not run_case(
    "single-quoted value",
    "KEY='hello world'",
    [("KEY", "hello world")],
)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
total = 8
passed = total - failures
if failures == 0:
    print(f"\033[32m✅ All {total} cases passed\033[0m")
else:
    print(f"\033[31m❌ {failures}/{total} cases failed\033[0m")

sys.exit(failures)
