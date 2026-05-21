"""Unit tests for the tunable text-quality classifier."""

import unittest

from app.domain.datasources.text_quality import (
    classify_text_line,
    TextQualityConfig,
    DecisionKind,
)


class TextQualityConfigTests(unittest.TestCase):
    def test_default_config_matches_original_behaviour(self):
        """Without a config the classifier must behave exactly as before for non-structured lines."""
        line = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"
        result = classify_text_line(line)
        self.assertEqual(result.kind, DecisionKind.DROP)
        self.assertIn("numeric_array", result.reason)

    def test_default_config_keeps_structured_hash_prefix(self):
        """The regex fix means ##.DELAY= … now gets structured_text bonus by default."""
        line = "##.DELAY= (6.5, 6.5)"
        result = classify_text_line(line)
        self.assertEqual(result.kind, DecisionKind.KEEP)
        self.assertIn("structured_text", result.reason)
        self.assertIn("too_many_symbols", result.reason)

    def test_tuned_symbol_threshold_keeps_metadata_line(self):
        """Raising symbol_ratio_threshold to 0.50 keeps ##.DELAY= (6.5, 6.5)."""
        line = "##.DELAY= (6.5, 6.5)"
        config = TextQualityConfig(symbol_ratio_threshold=0.50)
        result = classify_text_line(line, config)
        self.assertEqual(result.kind, DecisionKind.KEEP)
        self.assertIn("structured_text", result.reason)

    def test_tuned_keep_threshold_can_drop_borderline_structured_line(self):
        """Raising keep_score_threshold can still drop a borderline structured line."""
        line = "##.DELAY= (6.5, 6.5)"
        default_result = classify_text_line(line)
        self.assertEqual(default_result.kind, DecisionKind.KEEP)

        config = TextQualityConfig(keep_score_threshold=0.90)
        result = classify_text_line(line, config)
        self.assertEqual(result.kind, DecisionKind.DROP)
        self.assertIn("structured_text", result.reason)

    def test_structured_text_regex_recognises_hash_dollar_prefix(self):
        """Lines starting with ##. or ##$ and containing a key=value should get the structured_text bonus."""
        for line in [
            "##.OBSERVE FREQUENCY= 400.13240078",
            "##$BF1= 400.13",
            "##.PULSE SEQUENCE= gradient echo",
        ]:
            with self.subTest(line=line):
                result = classify_text_line(line)
                self.assertIn("structured_text", result.reason)

    def test_numeric_array_still_dropped_with_tuned_config(self):
        """A line of many numbers should still be dropped even with a lenient config."""
        line = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"
        config = TextQualityConfig(
            symbol_ratio_threshold=0.80,
            keep_score_threshold=0.10,
        )
        result = classify_text_line(line, config)
        self.assertEqual(result.kind, DecisionKind.DROP)
        self.assertIn("numeric_array", result.reason)


if __name__ == "__main__":
    unittest.main()
