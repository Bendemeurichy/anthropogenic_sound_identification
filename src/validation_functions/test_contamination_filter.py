#!/usr/bin/env python3
"""
Quick test script to verify contamination filtering is working correctly.
"""
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.validation_functions.test_pipeline import (
    COI_SYNONYMS,
    _is_coi_label,
    _filter_contaminated_backgrounds,
    _extract_label_atoms,
    _norm_label,
)


def test_synonym_expansion():
    """Test that COI_SYNONYMS includes all expected variants."""
    print("Testing COI_SYNONYMS expansion...")
    expected = {
        "plane",
        "planes",
        "airplane",
        "airplanes",
        "aeroplane",
        "aeroplanes",
        "aircraft",
        "fixed-wing aircraft, airplane",
        "fixed wing aircraft, airplane",
        "fixed-wing aircraft",
        "fixed wing aircraft",
        "aircraft engine",
        "jet engine",
        "propeller, airscrew",
    }
    assert COI_SYNONYMS == expected, f"COI_SYNONYMS mismatch: {COI_SYNONYMS}"
    print(f"✓ COI_SYNONYMS contains {len(COI_SYNONYMS)} terms as expected")


def test_label_normalization():
    """Test that label normalization works correctly."""
    print("\nTesting label normalization...")
    test_cases = [
        ("Airplane", "airplane"),
        ("  PLANE  ", "plane"),
        ("Fixed-Wing Aircraft, Airplane", "fixed-wing aircraft, airplane"),
        ("aircraft  engine", "aircraft engine"),
    ]
    for input_label, expected in test_cases:
        result = _norm_label(input_label)
        assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"✓ Label normalization working correctly ({len(test_cases)} test cases)")


def test_coi_detection():
    """Test that _is_coi_label correctly identifies COI variants."""
    print("\nTesting COI detection...")
    positive_cases = [
        "plane",
        "Plane",
        "AIRPLANE",
        "aircraft",
        "AeroPlane",
        "planes",
        "airplanes",
        "jet engine",
    ]
    negative_cases = [
        "train",
        "car",
        "background",
        "jet",  # Just "jet" alone, not "jet engine"
        "flying",  # Generic term
        "bird",
        "",
        None,
    ]

    for label in positive_cases:
        assert _is_coi_label(label), f"Should detect '{label}' as COI"
    print(f"✓ Detected {len(positive_cases)} positive cases correctly")

    for label in negative_cases:
        assert not _is_coi_label(label), f"Should NOT detect '{label}' as COI"
    print(f"✓ Rejected {len(negative_cases)} negative cases correctly")


def test_contamination_filter():
    """Test the contamination filter on mock data."""
    print("\nTesting contamination filter...")

    # Create mock background dataframe
    df_bg = pd.DataFrame(
        {
            "filename": [f"file_{i}.wav" for i in range(10)],
            "label": [0] * 10,
            "split": ["test"] * 5 + ["train"] * 5,
            "orig_label": [
                "background",  # 0: clean
                "wind",  # 1: clean
                "airplane",  # 2: CONTAMINATED
                "traffic",  # 3: clean
                "aircraft",  # 4: CONTAMINATED
                "rain",  # 5: clean
                "plane",  # 6: CONTAMINATED
                "birds",  # 7: clean
                "aeroplane",  # 8: CONTAMINATED
                "silence",  # 9: clean
            ],
        }
    )

    # Apply filter
    filtered_df, n_contaminated = _filter_contaminated_backgrounds(
        df_bg, verbose=False
    )

    # Verify results
    assert n_contaminated == 4, f"Expected 4 contaminated, got {n_contaminated}"
    assert len(filtered_df) == 6, f"Expected 6 clean samples, got {len(filtered_df)}"

    # Verify contaminated samples were removed
    remaining_labels = set(filtered_df["orig_label"].values)
    contaminated_labels = {"airplane", "aircraft", "plane", "aeroplane"}
    assert (
        len(remaining_labels & contaminated_labels) == 0
    ), "Contaminated labels still present"

    print(f"✓ Filter removed {n_contaminated} contaminated samples correctly")
    print(f"✓ {len(filtered_df)} clean samples retained")


def test_raw_label_expansion():
    """Test that raw label expansion splits composite labels."""
    print("\nTesting raw label expansion...")

    cases = [
        (["wind", "rain"], ["wind", "rain"]),
        ("['wind', 'rain']", ["wind", "rain"]),
        ("[array(['wind', 'rain'])]", ["wind", "rain"]),
        ("background", ["background"]),
        (None, []),
    ]

    for raw, expected in cases:
        assert _extract_label_atoms(raw) == expected, (
            f"Expected {_extract_label_atoms(raw)} to equal {expected}"
        )

    print(f"✓ Raw label expansion working correctly ({len(cases)} test cases)")


def test_filter_without_orig_label():
    """Test that filter gracefully handles missing orig_label column."""
    print("\nTesting filter without orig_label column...")

    df_bg = pd.DataFrame(
        {
            "filename": [f"file_{i}.wav" for i in range(5)],
            "label": [0] * 5,
            "split": ["test"] * 5,
        }
    )

    filtered_df, n_contaminated = _filter_contaminated_backgrounds(
        df_bg, verbose=False
    )

    assert n_contaminated == 0, "Should report 0 contaminated when no orig_label"
    assert len(filtered_df) == len(
        df_bg
    ), "Should return unmodified df when no orig_label"
    print("✓ Filter handles missing orig_label column correctly")


def main():
    print("=" * 60)
    print("CONTAMINATION FILTER TEST SUITE")
    print("=" * 60)

    try:
        test_synonym_expansion()
        test_label_normalization()
        test_coi_detection()
        test_contamination_filter()
        test_raw_label_expansion()
        test_filter_without_orig_label()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
