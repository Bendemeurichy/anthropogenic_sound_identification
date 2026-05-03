"""
Centralized COI (Class of Interest) label definitions and utilities.
Ensures consistent label handling across dataset creation and evaluation.

This module serves as the single source of truth for:
- COI synonym definitions (airplane/aircraft, bird, or custom variants)
- Label normalization logic
- COI detection across different label formats

Used by:
- src/label_loading/sampler.py (dataset creation)
- src/validation_functions/test_pipeline.py (evaluation)
- Training scripts (configuration)

The module supports multiple COI types (airplane, bird, etc.) by providing
separate synonym sets. ValidationPipeline can be configured to use any set.
"""

import ast
import re
from typing import Any, List, Set

# ============================================================================
# COI SYNONYM DEFINITIONS
# ============================================================================

# Airplane/aircraft synonyms (normalized: lowercase, whitespace-collapsed)
AIRPLANE_SYNONYMS: Set[str] = {
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

# Bird synonyms for bird detection tasks
BIRD_SYNONYMS: Set[str] = {
    "bird",
    "birds",
    "avian",
    "birdsong",
    "bird song",
    "bird call",
    "bird calls",
    # AudioSet bird-related labels
    "bird vocalization, bird call, bird song",
    # Common categories (can be extended based on your dataset)
    "songbird",
    "songbirds",
    "waterfowl",
    "raptor",
    "raptors",
    # Risoux-specific label (biophony = biological sounds, primarily birds in this dataset)
    "biophony",
}

# Default COI synonyms (airplane for backwards compatibility)
# Can be overridden by passing custom synonym sets to functions
COI_SYNONYMS: Set[str] = AIRPLANE_SYNONYMS

# All known synonym sets, in priority order.  Used by _synonyms_for_targets()
# to auto-detect the appropriate set when target_classes are provided.
_ALL_SYNONYM_SETS: List[Set[str]] = [AIRPLANE_SYNONYMS, BIRD_SYNONYMS]


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

        # Strip common numpy array string representations with dtype suffix
        # e.g., "[array(['...'], dtype=object)]" -> "[array(['...'])]"
        if ", dtype=" in text:
            # Remove everything from ", dtype=" to the end of the dtype declaration
            import re as _re

            text_cleaned = _re.sub(r",\s*dtype\s*=\s*[^\])]+([\])])", r"\1", text)
            candidates.append(text_cleaned)

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

        # Try to extract JSON from within the string if it contains a JSON array
        # This handles cases like: '[array(['["tag1", "tag2"]'], dtype=object)]'
        json_match = re.search(r"\[\"[^\]]+\"\]", text)
        if json_match:
            try:
                import json as _json

                json_str = json_match.group(0)
                parsed = _json.loads(json_str)
                if isinstance(parsed, list):
                    return list(dict.fromkeys([str(x) for x in parsed if x]))
            except Exception:
                pass

        quoted = [q.strip() for q in re.findall(r"['\"]([^'\"]+)['\"]", text)]
        if quoted:
            return list(dict.fromkeys([q for q in quoted if q]))

        return [text]

    return [str(label)]


def _synonyms_for_targets(target_classes: List[str]) -> Set[str]:
    """Derive the appropriate synonym set from a list of target class names.

    For each class name in *target_classes*, checks every registered synonym
    set (``_ALL_SYNONYM_SETS``) in order.  The first set that contains a
    normalized form of the class name is used to expand **all** targets.
    Classes that match no known set are kept as-is (exact match only).

    This means:
    * ``["airplane"]``   → :data:`AIRPLANE_SYNONYMS`
    * ``["bird"]``       → :data:`BIRD_SYNONYMS`
    * ``["cat"]``        → ``{"cat"}`` (no expansion, unknown class)
    * ``["plane", "bird"]`` → ``AIRPLANE_SYNONYMS | BIRD_SYNONYMS``

    Args:
        target_classes: List of target class name strings.

    Returns:
        Set of synonyms to use for matching.
    """
    expanded: Set[str] = set()
    for tc in target_classes:
        norm_tc = normalize_label(tc)
        matched = False
        for known_set in _ALL_SYNONYM_SETS:
            if norm_tc in known_set:
                expanded.update(known_set)
                matched = True
                break
        if not matched:
            expanded.add(norm_tc)
    return expanded


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
        # Derive the correct synonym set from target_classes so that, e.g.,
        # label_contains_coi("avian", ["bird"], use_synonyms=True) → True.
        # Previously this hardcoded COI_SYNONYMS (= AIRPLANE_SYNONYMS), which
        # meant bird target classes were silently never matched.
        return is_coi_label(labels, _synonyms_for_targets(target_classes))
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


def get_coi_synonyms_for_classifier(classifier_type: str) -> Set[str]:
    """Get the appropriate COI synonym set for a given classifier type.

    This function maps classifier types to their corresponding COI synonym sets,
    enabling the validation pipeline to use the correct labels for different tasks.

    Args:
        classifier_type: Type of classifier being used. One of:
            - "plane": Custom plane classifier → AIRPLANE_SYNONYMS
            - "pann_finetuned": Fine-tuned PANN → AIRPLANE_SYNONYMS
            - "ast_finetuned": Fine-tuned AST → AIRPLANE_SYNONYMS
            - "bird_mae": Bird-MAE-Base → BIRD_SYNONYMS
            - "audioprotopnet": AudioProtoPNet-20-BirdSet-XCL → BIRD_SYNONYMS

    Returns:
        Set of COI synonyms appropriate for the classifier type

    Raises:
        ValueError: If classifier_type is not recognized

    Examples:
        >>> get_coi_synonyms_for_classifier("plane")
        AIRPLANE_SYNONYMS
        >>> get_coi_synonyms_for_classifier("birdnet")
        BIRD_SYNONYMS
    """
    # Map classifier types to their COI synonym sets
    classifier_to_synonyms = {
        "plane": AIRPLANE_SYNONYMS,
        "ast_finetuned": AIRPLANE_SYNONYMS,
        "pann_finetuned": AIRPLANE_SYNONYMS,
        "audioprotopnet": BIRD_SYNONYMS,
        "bird_mae": BIRD_SYNONYMS,
    }

    if classifier_type not in classifier_to_synonyms:
        raise ValueError(
            f"Unknown classifier_type: {classifier_type}. "
            f"Must be one of: {list(classifier_to_synonyms.keys())}"
        )

    return classifier_to_synonyms[classifier_type]


def get_coi_synonyms_list() -> List[str]:
    """Return COI synonyms as a sorted list (for display/logging).

    Returns:
        Sorted list of COI synonyms
    """
    return sorted(COI_SYNONYMS)


def expand_target_classes_with_synonyms(
    target_classes: List[str], coi_synonyms: Set[str] = None
) -> Set[str]:
    """Expand a target class list to include all relevant synonyms.

    Given a list of target classes (e.g., ["plane", "airplane"]), returns
    the full set of synonyms that cover those classes.

    When *coi_synonyms* is ``None`` (the default), the appropriate synonym set
    is auto-detected from *target_classes* via :func:`_synonyms_for_targets`:
    airplane targets expand to :data:`AIRPLANE_SYNONYMS`, bird targets expand
    to :data:`BIRD_SYNONYMS`, etc.

    When *coi_synonyms* is provided explicitly, only synonyms within that set
    that match a target class are returned (original behavior).

    Args:
        target_classes: List of target class names.
        coi_synonyms: Optional explicit synonym set to restrict expansion to.
            Defaults to ``None`` (auto-detect from target_classes).

    Returns:
        Set of all matching synonyms.

    Examples:
        >>> expand_target_classes_with_synonyms(["plane"])
        {"plane", "planes", "airplane", "airplanes", ...}
        >>> expand_target_classes_with_synonyms(["bird"])
        {"bird", "birds", "avian", ...}
        >>> expand_target_classes_with_synonyms(["bird"], BIRD_SYNONYMS)
        {"bird", "birds", "avian", ...}
    """
    if coi_synonyms is None:
        # Derive the synonym set from target_classes rather than defaulting to
        # the module-level COI_SYNONYMS (= AIRPLANE_SYNONYMS), which would
        # silently ignore bird targets.
        return _synonyms_for_targets(target_classes)

    expanded = set()
    for tc in target_classes:
        if is_coi_label(tc, coi_synonyms):
            # If this target matches any synonym, include all synonyms
            expanded.update(coi_synonyms)
            break
    return expanded


# ============================================================================
# VALIDATION
# ============================================================================


def validate_synonym_set(synonym_set: Set[str], set_name: str = "COI_SYNONYMS"):
    """Validate that a synonym set is internally consistent.

    Runs basic sanity checks on a synonym set.
    Raises AssertionError if validation fails.

    Args:
        synonym_set: Set of synonyms to validate
        set_name: Name of the set for error messages
    """
    # Check that set is not empty
    assert len(synonym_set) > 0, f"{set_name} cannot be empty"

    # Check that all synonyms are normalized (lowercase, no extra whitespace)
    for synonym in synonym_set:
        normalized = normalize_label(synonym)
        assert (
            synonym == normalized
        ), f"{set_name}: Synonym '{synonym}' is not normalized (should be '{normalized}')"

    # Check that is_coi_label works for all synonyms
    for synonym in synonym_set:
        assert is_coi_label(
            synonym, synonym_set
        ), f"{set_name}: is_coi_label failed to detect synonym '{synonym}'"


def validate_coi_config():
    """Validate that all COI configurations are internally consistent.

    Runs basic sanity checks on AIRPLANE_SYNONYMS, BIRD_SYNONYMS, and COI_SYNONYMS.
    Raises AssertionError if validation fails.
    """
    # Validate all predefined synonym sets
    validate_synonym_set(AIRPLANE_SYNONYMS, "AIRPLANE_SYNONYMS")
    validate_synonym_set(BIRD_SYNONYMS, "BIRD_SYNONYMS")
    validate_synonym_set(COI_SYNONYMS, "COI_SYNONYMS")

    # Check that airplane and bird sets are disjoint (don't overlap)
    overlap = AIRPLANE_SYNONYMS & BIRD_SYNONYMS
    assert len(overlap) == 0, (
        f"AIRPLANE_SYNONYMS and BIRD_SYNONYMS should be disjoint, "
        f"but found overlapping terms: {overlap}"
    )

    # Check that is_coi_label correctly rejects labels from other sets
    # When using airplane synonyms, bird labels should be rejected
    for bird_label in ["bird", "songbird", "avian"]:
        assert not is_coi_label(
            bird_label, AIRPLANE_SYNONYMS
        ), f"AIRPLANE_SYNONYMS incorrectly detected bird label '{bird_label}' as COI"

    # When using bird synonyms, airplane labels should be rejected
    for plane_label in ["plane", "airplane", "aircraft"]:
        assert not is_coi_label(
            plane_label, BIRD_SYNONYMS
        ), f"BIRD_SYNONYMS incorrectly detected airplane label '{plane_label}' as COI"

    # Check that is_coi_label rejects non-COI labels for both sets
    non_coi_labels = ["train", "car", "background", "wind", "rain", "traffic"]
    for label in non_coi_labels:
        assert not is_coi_label(
            label, AIRPLANE_SYNONYMS
        ), f"AIRPLANE_SYNONYMS incorrectly detected '{label}' as COI"
        assert not is_coi_label(
            label, BIRD_SYNONYMS
        ), f"BIRD_SYNONYMS incorrectly detected '{label}' as COI"

    # ---- label_contains_coi synonym-expansion tests ----

    # Airplane path: a synonym not literally in target_classes should match
    assert label_contains_coi("aircraft", ["plane"], use_synonyms=True), (
        "label_contains_coi: 'aircraft' should match target=['plane'] with use_synonyms=True"
    )
    assert not label_contains_coi("aircraft", ["plane"], use_synonyms=False), (
        "label_contains_coi: 'aircraft' should NOT match target=['plane'] with use_synonyms=False"
    )

    # Bird path: previously broken — was checking AIRPLANE_SYNONYMS instead of BIRD_SYNONYMS
    assert label_contains_coi("avian", ["bird"], use_synonyms=True), (
        "label_contains_coi: 'avian' should match target=['bird'] with use_synonyms=True"
    )
    assert label_contains_coi("birdsong", ["birds"], use_synonyms=True), (
        "label_contains_coi: 'birdsong' should match target=['birds'] with use_synonyms=True"
    )
    assert not label_contains_coi("airplane", ["bird"], use_synonyms=True), (
        "label_contains_coi: 'airplane' should NOT match target=['bird'] with use_synonyms=True"
    )
    assert not label_contains_coi("bird", ["plane"], use_synonyms=True), (
        "label_contains_coi: 'bird' should NOT match target=['plane'] with use_synonyms=True"
    )

    # expand_target_classes_with_synonyms auto-detection
    expanded_bird = expand_target_classes_with_synonyms(["bird"])
    assert "avian" in expanded_bird, (
        "expand_target_classes_with_synonyms(['bird']) should include 'avian'"
    )
    assert "airplane" not in expanded_bird, (
        "expand_target_classes_with_synonyms(['bird']) should not include 'airplane'"
    )
    expanded_plane = expand_target_classes_with_synonyms(["plane"])
    assert "aircraft" in expanded_plane, (
        "expand_target_classes_with_synonyms(['plane']) should include 'aircraft'"
    )


# Run validation on import
validate_coi_config()
