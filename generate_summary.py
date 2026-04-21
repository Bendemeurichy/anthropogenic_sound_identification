"""
WEBDATASET LABEL EXTRACTION SUMMARY
================================================================================

Date: 2026-04-21
Task: Extract all unique labels from webdataset tar files in ../data/webdataset

DATASET OVERVIEW
--------------------------------------------------------------------------------
Total tar files processed: 37
  - Train shards: 23 (train-000000.tar to train-000022.tar)
  - Validation shards: 5 (val-000000.tar to val-000004.tar)
  - Test shards: 7 (test-000000.tar to test-000006.tar)
  - Manifest: 1 (manifest.json)

LABEL STATISTICS
--------------------------------------------------------------------------------
Total unique labels found: 5,892

Label format distribution:
  - simple_string: Labels from ESC-50 dataset (e.g., "dog", "airplane", "rain")
  - numpy_array_string: Labels from Freesound dataset stored as:
    '[array(\'["tag1", "tag2", ...]\', dtype=object)]'

ESC-50 DATASET LABELS (50 classes)
--------------------------------------------------------------------------------
All 50 ESC-50 labels are present in the webdataset:

✓ airplane          ✓ breathing         ✓ brushing_teeth    ✓ can_opening
✓ car_horn          ✓ cat               ✓ chainsaw          ✓ chirping_birds
✓ church_bells      ✓ clapping          ✓ clock_alarm       ✓ clock_tick
✓ coughing          ✓ cow               ✓ crackling_fire    ✓ crickets
✓ crow              ✓ crying_baby       ✓ dog               ✓ door_wood_creaks
✓ door_wood_knock   ✓ drinking_sipping  ✓ engine            ✓ fireworks
✓ footsteps         ✓ frog              ✓ glass_breaking    ✓ hand_saw
✓ helicopter        ✓ hen               ✓ insects           ✓ keyboard_typing
✓ laughing          ✓ mouse_click       ✓ pig               ✓ pouring_water
✓ rain              ✓ rooster           ✓ sea_waves         ✓ sheep
✓ siren             ✓ sneezing          ✓ snoring           ✓ thunderstorm
✓ toilet_flush      ✓ train             ✓ vacuum_cleaner    ✓ washing_machine
✓ water_drops       ✓ wind

FREESOUND DATASET LABELS
--------------------------------------------------------------------------------
In addition to ESC-50, there are 5,842 unique tags from the Freesound dataset.
These tags are more descriptive and include:

Categories:
  - Sound events (e.g., "thunder", "lightning", "rotor", "propeller")
  - Recording metadata (e.g., "field-recording", "stereo", "binaural")
  - Locations (e.g., "nature", "outdoor", "urban", "forest")
  - Weather (e.g., "storm", "rainstorm", "weather", "wind-chimes")
  - Water sounds (e.g., "waterfall", "stream", "river", "ocean")
  - Animals (e.g., "birds", "frogs", "insects", "wildlife")
  - Human activities (e.g., "talking", "footsteps", "clapping")
  - Mechanical (e.g., "engine", "motor", "machinery", "vehicle")
  - Musical (e.g., "bells", "percussion", "instruments")
  - Environmental (e.g., "ambience", "atmosphere", "soundscape")

NOTABLE LABELS
--------------------------------------------------------------------------------
Some interesting multi-tag examples from Freesound include:
  - ["helicopter", "field-recording", "rotor"]
  - ["field-recording", "lightning", "nature", "rain", "storm", "thunder"]
  - ["field-recording", "water", "stream", "waterfall", "nature"]
  - ["field-recording", "wind", "trees", "nature"]

RECOMMENDATIONS FOR TRAINING CONFIG
--------------------------------------------------------------------------------
1. ESC-50 Coverage: Complete ✓
   All 50 ESC-50 classes are present in the webdataset

2. Freesound Tags: 5,842 unique tags
   Consider the following approaches:
   
   a) Use all tags for multi-label classification
   b) Filter to most common N tags (e.g., top 100, 500, or 1000)
   c) Map similar tags to canonical forms (e.g., "rain" vs "rainfall" vs "raining")
   d) Group tags by category (weather, animals, mechanical, etc.)
   e) Use hierarchical classification (ESC-50 as top level, Freesound tags as fine-grained)

3. Label Format Handling:
   - ESC-50: Single string label
   - Freesound: List of string tags (stored as numpy array string representation)
   - Training code should handle both formats

4. Potential Issues:
   - High label diversity (5,892 unique labels)
   - Some labels are very specific (locations, dates, equipment names)
   - Some labels are metadata rather than sound content
   - Consider filtering out non-acoustic tags:
     * Equipment names (e.g., "zoom-h4n", "rode-nt5")
     * Locations (e.g., "rotterdam", "scotland", "tokyo")
     * Dates (e.g., "2020", "2-may-2017")
     * Recording metadata (e.g., "24-bit", "48khz", "stereo")

FILES GENERATED
--------------------------------------------------------------------------------
1. unique_labels_complete.txt - Full list of all 5,892 unique labels
2. unique_labels.py - Python list format for easy import
3. label_extraction_summary.txt - This summary document

NEXT STEPS
--------------------------------------------------------------------------------
1. Review training config to ensure ESC-50 labels are correctly mapped
2. Decide on strategy for handling Freesound multi-label tags
3. Consider filtering or grouping Freesound tags for better training
4. Verify label vocabulary in training code matches extracted labels
"""

with open('label_extraction_summary.txt', 'w') as f:
    f.write(__doc__)

print("Summary saved to label_extraction_summary.txt")
