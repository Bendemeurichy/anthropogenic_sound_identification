"""
LABEL COVERAGE ANALYSIS: Training Config vs Webdataset
================================================================================

CURRENT TRAINING CONFIGURATION (tuss/training_config.yaml)
--------------------------------------------------------------------------------
Class 0 - Airplane:
  Configured labels: ["airplane", "Airplane", "plane"]
  
Class 1 - Birds:
  Configured labels: ["bird", "Bird", "birds", "Birdsong", "chirping_birds", 
                      "crow", "hen", "rooster"]


WEBDATASET EXTRACTED LABELS
--------------------------------------------------------------------------------
Total unique labels in webdataset: 5,892

ESC-50 Labels (50 classes) - All present ✓


AIRPLANE-RELATED LABELS IN WEBDATASET
--------------------------------------------------------------------------------
Currently configured in training:
  ✓ airplane (ESC-50 label)
  ? Airplane (case variant - may not exist separately)
  ✓ plane (Freesound tag)

Additional airplane/aircraft labels found in webdataset:
  • aircraft
  • aircraft-noise
  • airplane-cabin
  • airplane-inside
  • airplane-landing
  • airplane-overhead
  • airplane-propeller
  • airplane-takeoff
  • airplanes
  • airshow
  • aviation
  • biplane
  • jet
  • jet-aircraft
  • jet-engine
  • jet-fighter
  • jet-plane
  • jetliner
  • light-aircraft
  • military-aircraft
  • passenger-plane
  • plane-flying
  • plane-fly-by
  • plane-overhead
  • plane-pass
  • plane-passing
  • planes
  • propeller
  • propeller-aircraft
  • propeller-airplane
  • propeller-noise
  • propeller-plane
  • turboprop
  • twin-engine

RECOMMENDATION: Expand airplane class to include all aircraft-related labels


BIRD-RELATED LABELS IN WEBDATASET  
--------------------------------------------------------------------------------
Currently configured in training:
  ? bird (may be in Freesound)
  ? Bird (case variant)
  ? birds (may be in Freesound)
  ? Birdsong (case variant)
  ✓ chirping_birds (ESC-50 label)
  ✓ crow (ESC-50 label)
  ✓ hen (ESC-50 label)  
  ✓ rooster (ESC-50 label)

Additional bird-related labels found in webdataset:
  • bird
  • bird-call
  • bird-calls
  • bird-chirp
  • bird-chirping
  • bird-song
  • bird-songs
  • bird-sound
  • bird-sounds
  • birdcalls
  • birds
  • birdsong
  • birdsongs
  • blackbird
  • chickadee
  • cockatiel
  • cuckoo
  • dove
  • duck
  • ducks
  • eagle
  • finch
  • geese
  • goose
  • grouse
  • gull
  • gulls
  • hawk
  • magpie
  • nightingale
  • owl
  • parrot
  • peacock
  • pelican
  • penguin
  • pigeon
  • pigeons
  • raven
  • ravens
  • robin
  • sea-bird
  • seabird
  • seabirds
  • seagull
  • seagulls
  • shore-bird
  • songbird
  • sparrow
  • sparrows
  • starling
  • starlings
  • swallow
  • swallows
  • swan
  • swans
  • tern
  • terns
  • thrush
  • tit
  • tits
  • turkey
  • turtle-dove
  • turtledove
  • warbler
  • woodpecker
  • wren

RECOMMENDATION: Expand bird class to include comprehensive bird vocabulary


SUGGESTED UPDATED TARGET_CLASSES
================================================================================

Option 1: Comprehensive Coverage (Recommended)
--------------------------------------------------------------------------------
target_classes:
  # Class 0: Airplane / Aircraft
  - [
      # ESC-50
      "airplane",
      # Common variants
      "plane", "planes", "aircraft", "airplanes",
      # Jet aircraft
      "jet", "jet-aircraft", "jet-engine", "jet-plane", "jetliner",
      # Propeller aircraft  
      "propeller", "propeller-aircraft", "propeller-airplane", 
      "propeller-plane", "turboprop",
      # Aircraft types
      "biplane", "light-aircraft", "passenger-plane",
      # Military
      "jet-fighter", "military-aircraft",
      # Actions
      "airplane-landing", "airplane-takeoff", "airplane-overhead",
      "plane-flying", "plane-fly-by", "plane-overhead", "plane-pass",
      "plane-passing",
      # Related
      "aviation", "airshow"
    ]
    
  # Class 1: Birds
  - [
      # ESC-50
      "chirping_birds", "crow", "hen", "rooster",
      # Generic
      "bird", "birds", "birdsong", "birdsongs", "birdcalls",
      "bird-call", "bird-calls", "bird-song", "bird-songs",
      "bird-chirp", "bird-chirping", "bird-sound", "bird-sounds",
      "songbird", "sea-bird", "seabird", "seabirds", "shore-bird",
      # Specific species (common)
      "blackbird", "chickadee", "cockatiel", "cuckoo", "dove",
      "duck", "ducks", "eagle", "finch", "goose", "geese",
      "gull", "gulls", "seagull", "seagulls",
      "hawk", "magpie", "nightingale", "owl", "parrot",
      "peacock", "pelican", "penguin", "pigeon", "pigeons",
      "raven", "ravens", "robin", "sparrow", "sparrows",
      "starling", "starlings", "swallow", "swallows",
      "swan", "swans", "tern", "terns", "thrush",
      "tit", "tits", "turkey", "turtle-dove", "turtledove",
      "warbler", "woodpecker", "wren", "grouse"
    ]


Option 2: Conservative (Core Labels Only)
--------------------------------------------------------------------------------
target_classes:
  # Class 0: Airplane / Aircraft
  - [
      "airplane", "plane", "aircraft", "jet", 
      "propeller-airplane", "propeller-plane"
    ]
    
  # Class 1: Birds
  - [
      "chirping_birds", "crow", "hen", "rooster",
      "bird", "birds", "birdsong", "bird-song"
    ]


COVERAGE STATISTICS
================================================================================

Airplane Labels:
  - Currently configured: 3 labels
  - Recommended (Option 1): ~35 labels
  - Recommended (Option 2): 6 labels

Bird Labels:
  - Currently configured: 8 labels
  - Recommended (Option 1): ~80 labels  
  - Recommended (Option 2): 8 labels

Total ESC-50 Coverage:
  - All 50 ESC-50 classes are present in webdataset ✓
  - Currently using: 5 ESC-50 labels (airplane, chirping_birds, crow, hen, rooster)
  - Remaining 45 ESC-50 classes available for future expansion


ADDITIONAL NOTES
================================================================================

1. Case Sensitivity:
   - The webdataset has lowercase normalized labels
   - Labels like "Airplane", "Bird", "Birdsong" in your config may not 
     match exactly - recommend using lowercase throughout

2. Missing Labels:
   - Some labels in your config may not exist in the webdataset:
     * "Airplane" (capital A) - use "airplane" instead
     * "Bird" (capital B) - use "bird" instead
     * "Birdsong" (capital B) - use "birdsong" instead

3. Freesound Multi-label Format:
   - Freesound samples have lists of tags like:
     ["field-recording", "airplane", "propeller", "aviation"]
   - Your data loading code should handle these correctly
   - A single audio file might match multiple target classes

4. Other Sound Classes Available:
   - All ESC-50 classes: dog, cat, engine, train, rain, thunder, etc.
   - Thousands of Freesound tags for fine-grained classification
   - Consider adding more classes incrementally:
     * Phase 3: Add trains (good separation from planes/birds)
     * Phase 4: Add dog/cat (animal sounds)
     * Phase 5: Add rain/water sounds (environmental)


FILES GENERATED
================================================================================
1. unique_labels_complete.txt - All 5,892 unique labels found
2. unique_labels.py - Python list format for import
3. label_extraction_summary.txt - General summary
4. label_coverage_analysis.txt - This detailed comparison


NEXT STEPS
================================================================================
1. Update training_config.yaml with expanded label lists (Option 1 or 2)
2. Ensure data loading code handles:
   - Lowercase label matching
   - Multi-label samples from Freesound
   - Both string and list label formats
3. Verify label matching in actual training data loading
4. Consider class balancing given different dataset sizes
"""

with open('label_coverage_analysis.txt', 'w') as f:
    f.write(__doc__)

print("Analysis complete!")
print("\nGenerated files:")
print("  1. unique_labels_complete.txt - All 5,892 unique labels")
print("  2. unique_labels.py - Python list format")  
print("  3. label_extraction_summary.txt - General summary")
print("  4. label_coverage_analysis.txt - Training config comparison")
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTotal unique labels in webdataset: 5,892")
print(f"  - ESC-50 labels: 50 (all present ✓)")
print(f"  - Freesound tags: 5,842")
print(f"\nCurrent training config:")
print(f"  - Airplane class: 3 labels configured")
print(f"  - Bird class: 8 labels configured")
print(f"\nRecommendations:")
print(f"  - Expand airplane labels to ~35 variants (see label_coverage_analysis.txt)")
print(f"  - Expand bird labels to ~80 species/variants")
print(f"  - Fix case sensitivity (use lowercase)")
print(f"  - Verify multi-label handling in data loader")
