#!/usr/bin/env python3
"""
Extract all unique labels from webdataset tar files.
"""

import json
import tarfile
from pathlib import Path
from collections import defaultdict

def extract_labels_from_tar(tar_path):
    """Extract all labels from a tar file."""
    labels = set()
    label_types = defaultdict(int)
    
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.json'):
                f = tar.extractfile(member)
                if f:
                    try:
                        data = json.load(f)
                        label = data.get('label')
                        
                        if label is not None:
                            # Track the type of label
                            label_type = type(label).__name__
                            label_types[label_type] += 1
                            
                            # Handle both string and list labels
                            if isinstance(label, str):
                                labels.add(label)
                            elif isinstance(label, list):
                                for lbl in label:
                                    if isinstance(lbl, str):
                                        labels.add(lbl)
                            else:
                                print(f"Warning: Unexpected label type {type(label)} in {tar_path.name}/{member.name}: {label}")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {tar_path.name}/{member.name}: {e}")
    
    return labels, label_types

def main():
    data_dir = Path('../data/webdataset')
    
    # Get all tar files
    tar_files = sorted(data_dir.glob('*.tar'))
    
    print(f"Found {len(tar_files)} tar files")
    print("=" * 80)
    
    all_labels = set()
    all_label_types = defaultdict(int)
    
    # Process each tar file
    for tar_path in tar_files:
        print(f"\nProcessing {tar_path.name}...")
        labels, label_types = extract_labels_from_tar(tar_path)
        
        # Update global sets
        all_labels.update(labels)
        for label_type, count in label_types.items():
            all_label_types[label_type] += count
        
        print(f"  Found {len(labels)} unique labels in this tar")
        print(f"  Label types: {dict(label_types)}")
    
    print("\n" + "=" * 80)
    print(f"\nTOTAL UNIQUE LABELS: {len(all_labels)}")
    print(f"\nLabel type distribution across all files:")
    for label_type, count in sorted(all_label_types.items()):
        print(f"  {label_type}: {count} occurrences")
    
    print("\n" + "=" * 80)
    print("\nALL UNIQUE LABELS (sorted alphabetically):")
    print("=" * 80)
    for label in sorted(all_labels):
        print(f"  - {label}")
    
    # Save to file
    output_file = Path('unique_labels.txt')
    with open(output_file, 'w') as f:
        f.write("All unique labels found in webdataset:\n")
        f.write("=" * 80 + "\n\n")
        for label in sorted(all_labels):
            f.write(f"{label}\n")
    
    print(f"\n\nLabels saved to: {output_file.absolute()}")

if __name__ == '__main__':
    main()
