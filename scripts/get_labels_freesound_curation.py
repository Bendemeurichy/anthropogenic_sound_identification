"""Script that loads in csv for freesound curation and uses api calls to retrieve labels for each sample.

Minimal changes:
- If <input>_with_labels.csv exists, load it and resume from the first row without labels.
- When a rate limit is hit (detected via exception text containing 429 / "rate limit" / "too many"),
  write the output file and exit so the script can be re-run to resume.

No CLI parameters or other refactors — behavior otherwise kept simple and close to original.
"""

import json
import os
import sys
import time

import freesound
import pandas as pd
import requests  # kept to avoid changing other code that may rely on this import
from dotenv import load_dotenv


def get_freesound_labels(csv_path: str):
    """Load freesound curation csv and retrieve labels for each sample using the freesound API.

    Args:
        csv_path: Path to the csv file containing the metadata for the freesound samples.

    Behavior:
        - Writes output to <csv_path>_with_labels.csv
        - If that file exists, resumes from it and skips rows that already have a labels value.
        - On rate-limit (429 / 'rate limit' / 'too many') the current file is written and the script exits (code 2).
    """
    # load api key from .env file
    load_dotenv()
    api_key = os.getenv("FREESOUND_KEY")
    assert api_key is not None, "FREESOUND_KEY must be set in the .env file"

    client = freesound.FreesoundClient()
    client.set_token(api_key, "token")

    out_path = csv_path.replace(".csv", "_with_labels.csv")

    if os.path.exists(out_path):
        print(f"Found existing output {out_path}, resuming from it.")
        df = pd.read_csv(out_path)
        # Ensure original columns exist (merge missing columns if any)
        original = pd.read_csv(csv_path)
        for col in original.columns:
            if col not in df.columns:
                df[col] = original[col]
    else:
        df = pd.read_csv(csv_path)
        # Create a new column for labels if not present
        if "labels" not in df.columns:
            df["labels"] = ""
        # write initial file so we have an output to resume from
        df.to_csv(out_path, index=False)

    # Iterate rows and fill labels for rows that don't have them yet
    for idx, row in df.iterrows():
        current = row.get("labels", "")
        # Normalize NaN to empty string
        if isinstance(current, float) and pd.isna(current):
            current = ""
        if str(current).strip() != "":
            # already processed
            continue

        freesound_id = row.get("search freesound")
        if pd.isna(freesound_id) or freesound_id == "":
            # Mark empty ids with empty list representation and save
            df.at[idx, "labels"] = "[]"
            df.to_csv(out_path, index=False)
            continue

        try:
            labels = request_labels_file(idx, freesound_id, client)
            # store as JSON string for reliable round-tripping
            df.at[idx, "labels"] = json.dumps(labels, ensure_ascii=False)
            # save after each successful retrieval so progress is persisted
            df.to_csv(out_path, index=False)
            # polite pause
            time.sleep(1)
        except Exception as e:
            msg = str(e).lower()
            # detect rate-limit and save & exit so user can rerun to resume
            if "429" in msg or "rate limit" in msg or "too many" in msg:
                print(f"Rate limit hit while requesting {freesound_id}: {e}")
                print(
                    f"Saving progress to {out_path} and exiting. Run the script again to resume."
                )
                df.to_csv(out_path, index=False)
                # exit with a distinct code
                sys.exit(2)
            else:
                # Non-rate-limit errors: record empty labels and continue
                print(f"Error retrieving labels for freesound ID {freesound_id}: {e}")
                df.at[idx, "labels"] = "[]"
                df.to_csv(out_path, index=False)
                # continue to next row

    # final save
    df.to_csv(out_path, index=False)
    print(f"Completed. Output written to {out_path}")


def request_labels_file(
    idx: int, freesound_id: int, client: freesound.FreesoundClient
) -> list[str]:
    """Make an API call to retrieve the labels for a given freesound sample.

    Args:
        freesound_id: The ID of the freesound sample.
        client: The FreesoundClient instance.
    Returns:
        A list of labels associated with the freesound sample.

    Raises:
        Exception: re-raises exceptions that look like rate-limit so the caller can handle/save/exit.
    """
    try:
        sound: freesound.FreesoundObject = client.get_sound(int(freesound_id))
        labels: list[str] = sound.tags
        category: str = sound.category

        print(
            f"Retrieved labels for freesound Index:{idx} ,ID {freesound_id}: {labels}, category: {category}"
        )
        return labels
    except Exception as e:
        # If this looks like a rate-limit, re-raise so the caller can detect and save
        msg = str(e).lower()
        if "429" in msg or "rate limit" in msg or "too many" in msg:
            raise
        # Otherwise just report and return empty list (caller will mark and continue)
        print(f"Error retrieving labels for freesound ID {freesound_id}: {e}")
        return []


def main():
    csv_path = "/home/bendm/Thesis/project/code/data/metadata/freesound_curation/source_freesound_field_recordings_links.csv"
    print("putting labels in freesound dataset")
    get_freesound_labels(csv_path)


if __name__ == "__main__":
    main()
