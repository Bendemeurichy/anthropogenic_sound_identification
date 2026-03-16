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
from pathlib import Path

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


def get_freesound_labels_batched(csv_path: str):
    """Refactored to use batch searching (100 IDs per request) to bypass daily limits."""
    load_dotenv()
    api_key = os.getenv("FREESOUND_KEY")
    assert api_key is not None, "FREESOUND_KEY must be set in the .env file"

    client = freesound.FreesoundClient()
    client.set_token(api_key, "token")

    out_path = csv_path.replace(".csv", "_with_labels.csv")

    # Load or initialize DataFrame
    if os.path.exists(out_path):
        print(f"Found existing output {out_path}, resuming...")
        df = pd.read_csv(out_path)
    else:
        df = pd.read_csv(csv_path)
        if "labels" not in df.columns:
            df["labels"] = ""
        df.to_csv(out_path, index=False)

    # 1. Identify rows that need labels
    # We look for rows where 'labels' is NaN or empty string
    to_process_mask = (df["labels"].isna()) | (
        df["labels"].astype(str).str.strip() == ""
    )

    # Get unique IDs to fetch to avoid redundant API calls
    ids_to_fetch = (
        df.loc[to_process_mask, "search freesound"].dropna().unique().tolist()
    )

    if not ids_to_fetch:
        print("No missing labels found. Everything is already processed.")
        return

    print(f"Total unique IDs to fetch: {len(ids_to_fetch)}")

    batch_size = 100  # Safe limit for URL length and API constraints

    for i in range(0, len(ids_to_fetch), batch_size):
        batch = ids_to_fetch[i : i + batch_size]

        # Format IDs for Solr filter: id:(123 456 789)
        id_filter = (
            "id:(" + " ".join(map(str, [int(float(fid)) for fid in batch])) + ")"
        )

        try:
            # ONE API call for 100 sounds
            results_pager = client.text_search(
                filter=id_filter, fields="id,tags,category", page_size=batch_size
            )

            # Create a mapping of what we found
            found_ids = []
            for sound in results_pager:
                label_str = json.dumps(sound.tags, ensure_ascii=False)
                # Update all rows in the DF that match this ID
                df.loc[
                    df["search freesound"].astype(str) == str(sound.id), "labels"
                ] = label_str
                found_ids.append(str(sound.id))

            # Handle IDs that weren't found (e.g., deleted or private sounds)
            # If we requested it but it didn't come back in search, mark as empty
            for requested_id in batch:
                rid_str = str(int(float(requested_id)))
                if rid_str not in found_ids:
                    df.loc[df["search freesound"].astype(str) == rid_str, "labels"] = (
                        "[]"
                    )

            # Save progress and report
            df.to_csv(out_path, index=False)
            print(
                f"Processed batch {i // batch_size + 1}: Fetched {len(found_ids)}/{len(batch)} sounds."
            )

            # Small pause to be a good citizen
            time.sleep(0.5)

        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                print(
                    f"Rate limit hit at batch starting with {batch[0]}. Saving and exiting."
                )
                df.to_csv(out_path, index=False)
                sys.exit(2)
            else:
                print(f"Error in batch starting with {batch[0]}: {e}")
                continue

    print(f"Completed! Final output written to {out_path}")


def main():
    csv_path = "/home/bendm/Thesis/project/code/data/metadata/freesound_curation/source_freesound_field_recordings_links.csv"
    print("putting labels in freesound dataset")
    get_freesound_labels_batched(csv_path)
    # remove_empty_labels(csv_path.replace(".csv", "_with_labels.csv"))
    # find_missing_indices(csv_path, csv_path.replace(".csv", "_with_labels.csv"))
    # remove_duplicate_previews(csv_path.replace(".csv", "_with_labels.csv"))
    # sync_indices_and_remove_no_filename(
    #     csv_path, csv_path.replace(".csv", "_with_labels.csv")
    # )
    # remove_padding(csv_path.replace(".csv", "_with_labels.csv"))


def remove_empty_labels(csv_path: str):
    """Utility function to remove rows with empty labels from the output csv."""
    df = pd.read_csv(csv_path)
    # Empty out labels column to None where array is empty or blank, don't drop rows
    df["labels"] = df["labels"].apply(
        lambda x: None if str(x).strip() in ["", "[]"] else x
    )
    df.to_csv(csv_path, index=False)
    print(
        f"Removed labels from rows with empty labels. Updated file saved to {csv_path}"
    )


def find_missing_indices(csv_original: str, csv_labels: str) -> None:
    """Sync rows from original to labels using index/background filename root.

    This function:
    1. Reads both the original CSV and the labels CSV
    2. Compares original 'index' against labels key column:
       - prefers 'background filename root'
       - falls back to 'index' if needed
    3. Adds rows present in original but missing in labels
    4. Ensures inserted rows set both 'background filename root' and 'index'
       so they are preserved during later cleanup

    Args:
        csv_original: Path to the original CSV file
        csv_labels: Path to the labels CSV file
    """
    df_original = pd.read_csv(csv_original)
    df_labels = pd.read_csv(csv_labels)

    if "index" not in df_original.columns:
        print("Original CSV has no 'index' column; skipping missing-row sync.")
        return

    # Decide which labels column to use as the key for presence checks
    labels_key_col = (
        "background filename root"
        if "background filename root" in df_labels.columns
        else "index"
    )
    if labels_key_col not in df_labels.columns:
        print(
            "Labels CSV has neither 'background filename root' nor 'index'; skipping."
        )
        return

    print(
        f"Comparing original 'index' against labels '{labels_key_col}' for missing rows."
    )

    # Normalize numeric IDs so 123 and 123.0 compare equal
    original_ids = set(pd.to_numeric(df_original["index"], errors="coerce").dropna())
    labels_ids = set(pd.to_numeric(df_labels[labels_key_col], errors="coerce").dropna())

    missing_ids = original_ids - labels_ids

    if not missing_ids:
        print("No missing rows found between original and labels CSV.")
        return

    print(f"Found {len(missing_ids)} missing IDs. Adding rows...")

    # Build rows shaped like labels CSV to avoid NaN in key columns after concat
    labels_cols = df_labels.columns.tolist()
    missing_rows = []

    for missing_id in sorted(missing_ids):
        match = df_original[
            pd.to_numeric(df_original["index"], errors="coerce") == missing_id
        ]
        if match.empty:
            continue
        src = match.iloc[0]

        # start with all labels columns as None
        row = {col: None for col in labels_cols}

        # map original first 4 columns where names match
        for col in df_original.columns[:4]:
            if col in row:
                row[col] = src[col]

        # always set both index-style columns when present in labels file
        if "background filename root" in row:
            row["background filename root"] = src["index"]
        if "index" in row:
            row["index"] = src["index"]

        # ensure preview/search fields are carried when present
        if "preview" in row and "preview" in src:
            row["preview"] = src["preview"]
        if "search freesound" in row and "search freesound" in src:
            row["search freesound"] = src["search freesound"]

        missing_rows.append(row)

    if not missing_rows:
        print("No rows could be constructed for missing IDs.")
        return

    new_rows_df = pd.DataFrame(missing_rows, columns=labels_cols)
    df_labels = pd.concat([df_labels, new_rows_df], ignore_index=True)

    df_labels.to_csv(csv_labels, index=False)
    print(f"Added {len(new_rows_df)} missing rows to {csv_labels}")


def remove_duplicate_previews(csv_path: str) -> None:
    """Remove duplicate preview URLs from the labels CSV.

    When there are duplicate previews, keep the row with a non-NaN 'background filename root'
    value and remove the ones without it.

    Args:
        csv_path: Path to the labels CSV file
    """
    df = pd.read_csv(csv_path)

    print(f"Starting with {len(df)} rows")

    # Check if there are duplicate previews
    duplicates = df[df.duplicated(subset=["preview"], keep=False)]

    if len(duplicates) == 0:
        print("No duplicate previews found.")
        return

    print(f"Found {len(duplicates)} rows with duplicate previews")
    print(f"Unique duplicate preview URLs: {duplicates['preview'].nunique()}")

    # For each duplicate preview, keep the one with non-NaN 'background filename root'
    # and remove the ones without it
    rows_to_keep = []
    seen_previews = set()

    for _, row in df.iterrows():
        preview = row["preview"]

        if preview in seen_previews:
            # This preview was already added, skip it
            continue

        seen_previews.add(preview)

        # Get all rows with this preview
        matching_rows = df[df["preview"] == preview]

        if len(matching_rows) > 1:
            # Multiple rows with same preview - keep the one with non-NaN background filename root
            rows_with_filename = matching_rows[
                matching_rows["background filename root"].notna()
            ]

            if len(rows_with_filename) > 0:
                # Keep the first row that has a background filename root
                rows_to_keep.append(rows_with_filename.iloc[0])
            else:
                # No rows with background filename root, keep the first one
                rows_to_keep.append(matching_rows.iloc[0])
        else:
            # Only one row with this preview, keep it
            rows_to_keep.append(row)

    # Create new dataframe from rows to keep
    df_cleaned = pd.DataFrame(rows_to_keep).reset_index(drop=True)

    print(f"After removing duplicates: {len(df_cleaned)} rows")
    print(f"Removed {len(df) - len(df_cleaned)} duplicate rows")

    # Write the cleaned CSV back
    df_cleaned.to_csv(csv_path, index=False)
    print(f"Updated file saved to {csv_path}")


def sync_indices_and_remove_no_filename(csv_original: str, csv_labels: str) -> None:
    """Remove rows without background filename root after index-based sync.

    Missing-row insertion is handled by `find_missing_indices`. This function only
    performs cleanup by removing rows that have no 'background filename root'.

    Args:
        csv_original: Path to the original CSV file (unused, kept for call compatibility)
        csv_labels: Path to the labels CSV file
    """
    _ = csv_original  # kept to preserve function signature
    df_labels = pd.read_csv(csv_labels)

    print(f"Starting cleanup with labels CSV: {len(df_labels)} rows")

    if "background filename root" not in df_labels.columns:
        print("No 'background filename root' column found; skipping cleanup.")
        return

    rows_before = len(df_labels)
    df_labels = df_labels[df_labels["background filename root"].notna()]
    rows_after = len(df_labels)
    rows_removed = rows_before - rows_after

    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with NaN 'background filename root'")
    else:
        print("No rows with NaN 'background filename root' to remove")

    df_labels.to_csv(csv_labels, index=False)
    print(f"Updated file saved to {csv_labels}")


def remove_padding(path: Path):
    """Utility function to remove padding from the labels column in the output csv, if any."""
    df = pd.read_csv(path)
    # Remove leading/trailing "/" from "search freesound" column
    if "search freesound" in df.columns:
        df["search freesound"] = df["search freesound"].apply(
            lambda x: str(x).split("/")[-1] if pd.notna(x) else x
        )
        df.to_csv(path, index=False)
        print(f"Removed padding from labels in {path}")
    else:
        print(
            f"No 'search freesound' column found in {path}, skipping padding removal."
        )


if __name__ == "__main__":
    main()
