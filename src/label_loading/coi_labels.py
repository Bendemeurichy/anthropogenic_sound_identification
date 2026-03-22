"""
Centralized COI (Class of Interest) label definitions and utilities.
Ensures consistent label handling across dataset creation and evaluation.

This module serves as the single source of truth for:
- COI synonym definitions (plane/airplane/aircraft variants)
- Label normalization logic
- COI detection across different label formats

Used by:
- src/label_loading/sampler.py (dataset creation)
- src/validation_functions/test_pipeline.py (evaluation)
- Training scripts (configuration)
"""

import ast
import re
from typing import Any, List, Set

# ============================================================================
# COI SYNONYM DEFINITIONS
# ============================================================================

# Full set of synonyms for plane/aircraft COI
# These are normalized (lowercase, whitespace-collapsed) forms
COI_SYNONYMS: Set[str] = {
    "plane",
    "planes",
    "airplane",
    "airplanes",
    "aeroplane",
    "aeroplanes",
    "aircraft",
    # AudioSet standard labels
    "fixed-wing aircraft, airplane",
    "fixed wing aircraft, airplane",
    "fixed-wing aircraft",
    "fixed wing aircraft",
    # Engine/propulsion related (clearly aircraft-specific)
    "aircraft engine",
    "jet engine",
    "propeller, airscrew",
}


# ============================================================================
# LABEL NORMALIZATION & DETECTION
# ============================================================================


def normalize_label(label: Any) -> str:
    """Normalize raw labels to a comparable lowercase string.

    Normalization:
    - Convert to string
    - Strip whitespace
    - Lowercase
    - Collapse multiple spaces to single space

    Args:
        label: Raw label (can be str, int, None, etc.)

    Returns:
        Normalized string representation

    Examples:
        >>> normalize_label("Airplane")
        "airplane"
        >>> normalize_label("  PLANE  ")
        "plane"
        >>> normalize_label("Fixed-Wing Aircraft, Airplane")
        "fixed-wing aircraft, airplane"
    """
    if label is None:
        return ""
    s = str(label).strip().lower()
    return " ".join(s.split())


def _extract_label_atoms(label: Any) -> List[str]:
    """Extract atomic label strings from common string/list/array shapes."""
    if label is None:
        return []

    if isinstance(label, list):
        out: List[str] = []
        for item in label:
            out.extend(_extract_label_atoms(item))
        return list(dict.fromkeys(out))

    if isinstance(label, tuple) or isinstance(label, set):
        out: List[str] = []
        for item in label:
            out.extend(_extract_label_atoms(item))
        return list(dict.fromkeys(out))

    if hasattr(label, "tolist") and not isinstance(label, str):
        try:
            return _extract_label_atoms(label.tolist())
        except Exception:
            pass

    if isinstance(label, str):
        text = label.strip()
        if not text:
            return []

        candidates = [text]
        if text.startswith("array(") and text.endswith(")"):
            candidates.append(text[6:-1])
        if text.startswith("np.array(") and text.endswith(")"):
            candidates.append(text[9:-1])

        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                continue

            if isinstance(parsed, (list, tuple, set)) or hasattr(parsed, "tolist"):
                return _extract_label_atoms(parsed)
            if parsed is None:
                return []
            return [str(parsed)]

        quoted = [q.strip() for q in re.findall(r"['\"]([^'\"]+)['\"]", text)]
        if quoted:
            return list(dict.fromkeys([q for q in quoted if q]))

        return [text]

    return [str(label)]


def is_coi_label(label: Any, coi_synonyms: Set[str] = COI_SYNONYMS) -> bool:
    """Check if label matches any COI synonym.

    Handles multiple label formats:
    - Simple strings: "plane", "airplane"
    - Lists: ["airplane", "other_label"]
    - Complex nested structures from CSVs: '[array(["airplane", ...])]'

    Args:
        label: Label to check (can be str, list, or complex nested structure)
        coi_synonyms: Set of COI synonyms to check against (defaults to COI_SYNONYMS)

    Returns:
        True if label contains any COI synonym, False otherwise

    Examples:
        >>> is_coi_label("plane")
        True
        >>> is_coi_label("train")
        False
        >>> is_coi_label(["airplane", "background"])
        True
        >>> is_coi_label('[array(["aircraft", "traffic"])]')
        True
    """
    if label is None:
        return False

    for atom in _extract_label_atoms(label):
        normalized = normalize_label(atom)
        if normalized in coi_synonyms:
            return True

    # For complex nested structures / messy string blobs, check substrings too.
    s_lower = normalize_label(label).lower()
    for synonym in coi_synonyms:
        if synonym in s_lower:
            return True

    return False


def label_contains_coi(
    labels: Any, target_classes: List[str], use_synonyms: bool = True
) -> bool:
    """Check if labels contain any target class, optionally using synonym matching.

    This is the main function used by the sampler to decide if a recording should
    be excluded from the background pool.

    Args:
        labels: Label(s) to check (str, list, or complex structure)
        target_classes: List of target class names (e.g., ["airplane", "plane"])
        use_synonyms: If True, use COI_SYNONYMS for matching; if False, only
            check exact matches against target_classes

    Returns:
        True if labels contain any target class (or its synonym), False otherwise

    Examples:
        >>> label_contains_coi("aircraft", ["plane"], use_synonyms=True)
        True
        >>> label_contains_coi("aircraft", ["plane"], use_synonyms=False)
        False
        >>> label_contains_coi(["wind", "rain"], ["plane"], use_synonyms=True)
        False
    """
    if labels is None:
        return False

    if use_synonyms:
        # Use COI_SYNONYMS for comprehensive matching
        return is_coi_label(labels, COI_SYNONYMS)
    else:
        # Only check exact matches against target_classes
        # Normalize target classes for comparison
        normalized_targets = {normalize_label(tc) for tc in target_classes}

        # Handle list labels
        if isinstance(labels, list):
            return any(
                label_contains_coi(item, target_classes, use_synonyms=False)
                for item in labels
            )

        # Simple string check
        s = normalize_label(labels)
        if s in normalized_targets:
            return True

        # Check if any target appears in complex nested structures
        s_lower = s.lower()
        for target in normalized_targets:
            if target in s_lower:
                return True

        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_coi_synonyms_list() -> List[str]:
    """Return COI synonyms as a sorted list (for display/logging).

    Returns:
        Sorted list of COI synonyms
    """
    return sorted(COI_SYNONYMS)


def expand_target_classes_with_synonyms(target_classes: List[str]) -> Set[str]:
    """Expand a target class list to include all relevant synonyms.

    Given a list of target classes (e.g., ["plane", "airplane"]), returns
    the full set of synonyms from COI_SYNONYMS that match any of them.

    Args:
        target_classes: List of target class names

    Returns:
        Set of all matching synonyms from COI_SYNONYMS

    Examples:
        >>> expand_target_classes_with_synonyms(["plane"])
        {"plane", "planes", "airplane", "airplanes", ...}
    """
    expanded = set()
    for tc in target_classes:
        if is_coi_label(tc, COI_SYNONYMS):
            # If this target matches any synonym, include all aircraft synonyms
            expanded.update(COI_SYNONYMS)
            break
    return expanded


# ============================================================================
# VALIDATION
# ============================================================================


def validate_coi_config():
    """Validate that the COI configuration is internally consistent.

    Runs basic sanity checks on COI_SYNONYMS and utilities.
    Raises AssertionError if validation fails.
    """
    # Check that COI_SYNONYMS is not empty
    assert len(COI_SYNONYMS) > 0, "COI_SYNONYMS cannot be empty"

    # Check that all synonyms are normalized (lowercase, no extra whitespace)
    for synonym in COI_SYNONYMS:
        normalized = normalize_label(synonym)
        assert (
            synonym == normalized
        ), f"Synonym '{synonym}' is not normalized (should be '{normalized}')"

    # Check that is_coi_label works for all synonyms
    for synonym in COI_SYNONYMS:
        assert is_coi_label(
            synonym
        ), f"is_coi_label failed to detect synonym '{synonym}'"

    # Check that is_coi_label rejects non-COI labels
    non_coi_labels = ["train", "car", "background", "wind", "rain", "bird"]
    for label in non_coi_labels:
        assert not is_coi_label(
            label
        ), f"is_coi_label incorrectly detected '{label}' as COI"


# Run validation on import
validate_coi_config()
