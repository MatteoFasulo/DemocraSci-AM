#!/usr/bin/env python3
"""Run the DemocraSci-AM pipeline for every classified policy solution."""

import json
import subprocess
import sys
from pathlib import Path

import llm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "data" / "policy_solutions_classified.json"
MAIN_FILE = SCRIPT_DIR / "main.py"

STANCE_API_BASE_URL = "https://api.deepseek.com/v1"
STANCE_API_KEY = llm.get_key(explicit_key="deepseek-key")

# STANCE_API_BASE_URL = "https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1"
# STANCE_API_KEY = llm.get_key(explicit_key="sdsc-key")

TOPIC_ALIASES = {
    "Culture, Religion, Sports, Entertainment":
        "Culture, Arts, Religion, Sport, Entertainment",
    "Financial Sector": "Finances, Financial Sector",
    "National Security": "National Security, Armed Forces, Defense",
}


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {DATA_FILE}")
    if not MAIN_FILE.is_file():
        raise FileNotFoundError(f"Main program not found: {MAIN_FILE}")

    records = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    for index, record in enumerate(records, start=1):
        policy_solution = record["Policy Solution "]
        topic = TOPIC_ALIASES.get(record["Political Topic"], record["Political Topic"])
        party = record["Party"]

        print(f"\n[{index:03d}/{len(records):03d}] Running {party} ({topic})", flush=True)

        # If run exists, skip it. Otherwise, run the main program with the appropriate arguments.
        run_name = f"{index:03d}-{party}-{policy_solution}"
        run_dir = SCRIPT_DIR / "outputs" / run_name
        if run_dir.exists():
            print(f"Run {run_name} already exists. Skipping.", flush=True)
            continue

        subprocess.run(
            [
                sys.executable,
                str(MAIN_FILE),
                "--run-name",
                run_name,
                "--topic",
                topic,
                "--retrieval-query",
                policy_solution,
                "--stance-claim",
                policy_solution,
                "--stance-api-base-url",
                STANCE_API_BASE_URL,
                "--stance-api-key",
                STANCE_API_KEY,
            ],
            cwd=SCRIPT_DIR,
            check=True,
        )

    print("\nAll policy-solution runs completed.")


if __name__ == "__main__":
    main()
