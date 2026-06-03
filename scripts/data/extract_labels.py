#!/usr/bin/env python3
"""
Extract all unique labels from webdataset tar files.
Handles both simple string labels and complex numpy array string representations.
"""

import json
import tarfile
import re
from pathlib import Path
from collections import defaultdict

def parse_label(label):
    """
    Parse label which can be:
    - A simple string (e.g., "dog")
    - A string representation of numpy array with JSON list (e.g., "[array(['[\"water\", \"stream\"]'], dtype=object)]")
    - A list of strings
    """
    labels = set()
    
    if isinstance(label, str):
        # Check if it's a numpy array string representation
        if label.startswith("[array(") and "dtype=object)]" in label:
            # Extract the JSON string from the numpy array representation
            # Pattern: [array(['["tag1", "tag2", ...]'], dtype=object)]
            json_match = re.search(r'\[array\(\[\'(.*?)\'\]', label)
            if json_match:
                json_str = json_match.group(1)
                # Unescape the string
                json_str = json_str.replace('\\"', '"')
                try:
                    # Parse the JSON array
                    tag_list = json.loads(json_str)
                    if isinstance(tag_list, list):
                        for tag in tag_list:
                            if isinstance(tag, str):
                                labels.add(tag.lower())  # Normalize to lowercase
                except json.JSONDecodeError:
                    print(f"  Warning: Could not parse JSON from numpy array: {json_str}")
        else:
            # Simple string label
            labels.add(label)
    elif isinstance(label, list):
        # Direct list of labels
        for lbl in label:
            if isinstance(lbl, str):
                labels.add(lbl)
    
    return labels

def extract_labels_from_tar(tar_path):
    """Extract all labels from a tar file."""
    labels = set()
    label_format_counts = defaultdict(int)
    
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.json'):
                f = tar.extractfile(member)
                if f:
                    try:
                        data = json.load(f)
                        label = data.get('label')
                        
                        if label is not None:
                            # Determine format
                            if isinstance(label, str):
                                if label.startswith("[array("):
                                    label_format_counts['numpy_array_string'] += 1
                                else:
                                    label_format_counts['simple_string'] += 1
                            elif isinstance(label, list):
                                label_format_counts['list'] += 1
                            else:
                                label_format_counts['other'] += 1
                            
                            # Parse and collect labels
                            parsed_labels = parse_label(label)
                            labels.update(parsed_labels)
                            
                    except json.JSONDecodeError as e:
                        print(f"  Error decoding JSON in {tar_path.name}/{member.name}: {e}")
    
    return labels, label_format_counts

def main():
    data_dir = Path('../data/webdataset')
    
    # Get all tar files
    tar_files = sorted(data_dir.glob('*.tar'))
    
    print(f"Found {len(tar_files)} tar files")
    print("=" * 80)
    
    all_labels = set()
    all_format_counts = defaultdict(int)
    
    # Process each tar file
    for i, tar_path in enumerate(tar_files, 1):
        print(f"\n[{i}/{len(tar_files)}] Processing {tar_path.name}...")
        labels, format_counts = extract_labels_from_tar(tar_path)
        
        # Update global sets
        all_labels.update(labels)
        for format_type, count in format_counts.items():
            all_format_counts[format_type] += count
        
        print(f"  Found {len(labels)} unique labels in this tar")
        if format_counts:
            print(f"  Format distribution: {dict(format_counts)}")
    
    print("\n" + "=" * 80)
    print(f"\nTOTAL UNIQUE LABELS: {len(all_labels)}")
    print(f"\nLabel format distribution across all files:")
    for format_type, count in sorted(all_format_counts.items()):
        print(f"  {format_type}: {count:,} occurrences")
    
    print("\n" + "=" * 80)
    print("\nALL UNIQUE LABELS (sorted alphabetically):")
    print("=" * 80)
    sorted_labels = sorted(all_labels)
    for i, label in enumerate(sorted_labels, 1):
        print(f"{i:4d}. {label}")
    
    # Save to file
    output_file = Path('unique_labels_complete.txt')
    with open(output_file, 'w') as f:
        f.write("All unique labels found in webdataset\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total: {len(all_labels)} unique labels\n")
        f.write("=" * 80 + "\n\n")
        for label in sorted_labels:
            f.write(f"{label}\n")
    
    print(f"\n\nLabels saved to: {output_file.absolute()}")
    
    # Also save as a Python list for easy import
    output_py = Path('unique_labels.py')
    with open(output_py, 'w') as f:
        f.write('"""All unique labels extracted from webdataset tar files."""\n\n')
        f.write('UNIQUE_LABELS = [\n')
        for label in sorted_labels:
            f.write(f'    "{label}",\n')
        f.write(']\n')
    
    print(f"Labels also saved as Python list to: {output_py.absolute()}")

if __name__ == '__main__':
    main()
