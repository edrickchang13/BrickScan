"""
Canonicalise LEGO part numbers to collapse mold variants + print variants.

The Part DB model has no `mold_parent_num` column (see backend/app/models/part.py),
so we can't do catalog-level mold collapsing. Fall back to regex pattern
matching on the part_num string — the same taxonomy Rebrickable uses
internally for its part grouping.

Rules, applied in order:
  1. Print variant:        "3001pr0001" -> "3001"
                           "3626cpx03b" -> "3626c"     (keep the base mold letter)
  2. Assembly suffix:      "3001old"    -> "3001"
                           "3001new"    -> "3001"
  3. Mold letter suffix:   "3001a"      -> "3001"      (a/b/c/.../z single letter)

Some part numbers legitimately end in a single lowercase letter and map to
DIFFERENT physical molds — e.g. 3001 vs 3001a are sometimes distinct.
Applied at the tail end of the cascade merge, so the highest-confidence
variant's full metadata is still preserved; this is just a deduplication
step that collapses "brick 2x4 variant A" and "brick 2x4 variant B"
into a single top-3 entry.
"""

from __future__ import annotations

import re

# Regex patterns — order matters (apply print-variant first so we don't
# accidentally strip part of a pr#### token).
_PRINT_VARIANT_RE = re.compile(r'pr\d+$', re.IGNORECASE)
_ASSEMBLY_SUFFIX_RE = re.compile(r'(old|new)$', re.IGNORECASE)
_MOLD_LETTER_RE = re.compile(r'[a-z]$')  # single trailing lowercase letter


def collapse_variant(part_num: str) -> str:
    """
    Return the canonical mold-parent part number.

    Safe to call on arbitrary strings — returns them unchanged if no
    pattern matches. Never raises.

    >>> collapse_variant("3001")
    '3001'
    >>> collapse_variant("3001pr0001")
    '3001'
    >>> collapse_variant("3001a")
    '3001'
    >>> collapse_variant("3001old")
    '3001'
    >>> collapse_variant("3626cpx3b")   # keep the 'c' (base mold), strip pr + trailing b
    '3626c'
    """
    if not part_num:
        return part_num

    pn = part_num.strip()

    # 1. Strip any print-variant suffix (pr####). Handle variants like
    #    3626cpx3 where 'pr' is represented as 'px' in some catalogues —
    #    we treat both as print variants. Track whether we stripped one
    #    because the letter PRECEDING a print suffix belongs to the
    #    base mold name (e.g. "3626c" in "3626cpx3"), NOT a mold variant.
    print_suffix_re = re.compile(r'(pr|px)\d+[a-z]?$', re.IGNORECASE)
    had_print_suffix = print_suffix_re.search(pn) is not None
    pn = print_suffix_re.sub('', pn)

    # 2. Strip old/new suffixes
    pn = _ASSEMBLY_SUFFIX_RE.sub('', pn)

    # 3. Strip trailing single lowercase letter — but ONLY when:
    #    - the remaining stem is all digits (so "3626c" stays "3626c"
    #      when the input was just "3626c" with no print variant), AND
    #    - we didn't just strip a print-variant suffix (so "3626cpx3"
    #      keeps the base 'c' after we drop "px3").
    if (not had_print_suffix
            and _MOLD_LETTER_RE.search(pn)
            and pn[:-1].isdigit()):
        pn = pn[:-1]

    return pn or part_num  # never return empty; fall back to original


def collapse_predictions(predictions: list) -> list:
    """
    Collapse duplicate candidates after variant-normalisation. Preserves
    the order + metadata of the highest-confidence variant and drops the
    lower-confidence duplicates.

    Input:  [{part_num: "3001",  confidence: 0.70, ...},
             {part_num: "3001a", confidence: 0.30, ...},
             {part_num: "3002",  confidence: 0.20, ...}]
    Output: [{part_num: "3001",  confidence: 0.70, ...},
             {part_num: "3002",  confidence: 0.20, ...}]
    """
    seen: set = set()
    out: list = []
    for p in predictions:
        pn = p.get("part_num", "")
        key = collapse_variant(pn).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
