"""
Phase B feature-flag tests — grounded Gemini, mold-collapse, color re-rank.

Each flag can be independently toggled via env var. Tests validate:
  1. `part_num_normalizer.collapse_variant` handles the regex cases correctly
  2. `part_num_normalizer.collapse_predictions` dedupes without losing order / metadata
  3. `color_extractor.extract_dominant_color` returns plausible RGB for a solid-color test image
  4. `color_extractor.rerank_predictions_by_color` downweights mismatched-color candidates
  5. `gemini_service._build_grounded_prompt` includes each candidate's part_num + name

Each service is unit-testable in isolation without spinning up the full
backend or mocking httpx (we don't test the live Gemini call here, just
that the prompt-building works).
"""

import io
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Phase B2 — part_num_normalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestCollapseVariant:
    def test_plain_part_num(self):
        from app.services.part_num_normalizer import collapse_variant
        assert collapse_variant("3001") == "3001"

    def test_mold_letter_suffix(self):
        from app.services.part_num_normalizer import collapse_variant
        assert collapse_variant("3001a") == "3001"
        assert collapse_variant("3001b") == "3001"

    def test_print_variant(self):
        from app.services.part_num_normalizer import collapse_variant
        assert collapse_variant("3001pr0001") == "3001"

    def test_assembly_suffix(self):
        from app.services.part_num_normalizer import collapse_variant
        assert collapse_variant("3001old") == "3001"
        assert collapse_variant("3001new") == "3001"

    def test_preserves_base_mold_letter(self):
        """3626c is a distinct base mold (not 3626a/b) — don't strip the 'c'."""
        from app.services.part_num_normalizer import collapse_variant
        # The part is "3626c" with a print variant — collapse the print, keep the 'c'
        assert collapse_variant("3626cpx3") == "3626c"

    def test_empty_input(self):
        from app.services.part_num_normalizer import collapse_variant
        assert collapse_variant("") == ""
        assert collapse_variant(None) is None


class TestCollapsePredictions:
    def test_dedupes_variants(self):
        from app.services.part_num_normalizer import collapse_predictions
        preds = [
            {"part_num": "3001",  "confidence": 0.70, "source": "brickognize"},
            {"part_num": "3001a", "confidence": 0.30, "source": "gemini"},
            {"part_num": "3002",  "confidence": 0.20, "source": "local_model"},
        ]
        result = collapse_predictions(preds)
        # 3001 keeps its higher-confidence metadata; 3001a removed; 3002 stays
        assert len(result) == 2
        assert result[0]["part_num"] == "3001"
        assert result[0]["confidence"] == 0.70
        assert result[0]["source"] == "brickognize"
        assert result[1]["part_num"] == "3002"

    def test_preserves_order(self):
        from app.services.part_num_normalizer import collapse_predictions
        preds = [
            {"part_num": "3002", "confidence": 0.50},
            {"part_num": "3001", "confidence": 0.90},
        ]
        # Input order is preserved — dedupe doesn't re-sort by confidence
        result = collapse_predictions(preds)
        assert result[0]["part_num"] == "3002"
        assert result[1]["part_num"] == "3001"


# ─────────────────────────────────────────────────────────────────────────────
# Phase B3 — color_extractor
# ─────────────────────────────────────────────────────────────────────────────

def _make_solid_color_jpeg(rgb: tuple) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (200, 200), color=rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class TestColorExtractor:
    def test_extracts_red(self):
        from app.services.color_extractor import extract_dominant_color
        bytes_ = _make_solid_color_jpeg((200, 20, 20))
        rgb = extract_dominant_color(bytes_)
        assert rgb is not None
        # Allow JPEG compression drift — R should be much larger than G and B
        assert rgb[0] > 150
        assert rgb[1] < 80
        assert rgb[2] < 80

    def test_extracts_yellow(self):
        from app.services.color_extractor import extract_dominant_color
        bytes_ = _make_solid_color_jpeg((240, 210, 30))
        rgb = extract_dominant_color(bytes_)
        assert rgb is not None
        assert rgb[0] > 150
        assert rgb[1] > 150
        assert rgb[2] < 100

    def test_empty_input_returns_none(self):
        from app.services.color_extractor import extract_dominant_color
        assert extract_dominant_color(b"") is None


class TestRerankByColor:
    def test_downweights_color_mismatch(self):
        from app.services.color_extractor import rerank_predictions_by_color
        # Scan is red. Candidate says the part is red → kept.
        # Candidate says the part is blue → downweighted.
        preds = [
            {"part_num": "3001", "confidence": 0.85, "color_hex": "#C91A09"},  # red
            {"part_num": "3002", "confidence": 0.80, "color_hex": "#0055BF"},  # blue
        ]
        result = rerank_predictions_by_color(preds, scan_rgb=(200, 20, 20))
        # Red kept at ~0.85, blue penalised to ~0.56 (0.80 * 0.7)
        assert result[0]["part_num"] == "3001"
        assert result[0]["confidence"] == pytest.approx(0.85)
        assert result[1]["part_num"] == "3002"
        assert result[1]["confidence"] == pytest.approx(0.80 * 0.7)

    def test_candidates_without_color_unchanged(self):
        from app.services.color_extractor import rerank_predictions_by_color
        preds = [
            {"part_num": "3001", "confidence": 0.85, "color_hex": None},
            {"part_num": "3002", "confidence": 0.80},
        ]
        result = rerank_predictions_by_color(preds, scan_rgb=(200, 20, 20))
        assert result[0]["confidence"] == pytest.approx(0.85)
        assert result[1]["confidence"] == pytest.approx(0.80)


# ─────────────────────────────────────────────────────────────────────────────
# Phase B1 — grounded Gemini prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestGroundedPrompt:
    def test_includes_every_candidate(self):
        from app.services.gemini_service import _build_grounded_prompt
        candidates = [
            {"part_num": "3001", "part_name": "Brick 2x4", "confidence": 0.82},
            {"part_num": "3003", "part_name": "Brick 2x2", "confidence": 0.41},
        ]
        prompt = _build_grounded_prompt(candidates)
        assert "3001" in prompt
        assert "Brick 2x4" in prompt
        assert "82%" in prompt
        assert "3003" in prompt
        assert "41%" in prompt

    def test_limits_to_5_candidates(self):
        """Prevents prompt bloat if upstream sends too many."""
        from app.services.gemini_service import _build_grounded_prompt
        candidates = [
            {"part_num": f"part_{i}", "part_name": f"Part {i}", "confidence": 0.1}
            for i in range(10)
        ]
        prompt = _build_grounded_prompt(candidates)
        assert "part_4" in prompt
        assert "part_5" not in prompt  # capped at first 5

    def test_handles_missing_fields_gracefully(self):
        from app.services.gemini_service import _build_grounded_prompt
        prompt = _build_grounded_prompt([{"part_num": "3001"}])  # no name, no conf
        assert "3001" in prompt
        assert "Unknown" in prompt or "part" in prompt.lower()
