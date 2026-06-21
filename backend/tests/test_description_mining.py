import json
import unittest

from app.domain.extraction.description_mining import (
    DESCRIPTION_FACT_MINING_SYSTEM_PROMPT,
    RawDescriptionFacts,
    augment_evidence_context_with_description_facts,
    build_description_mining_prompt,
    collect_dataset_description_sources,
    validate_description_facts,
)
from app.domain.extraction.evidence_context import EvidenceCandidate, RoutedEvidenceContext


class DescriptionMiningTests(unittest.TestCase):
    def test_collects_only_top_level_dataset_descriptions(self):
        document = {
            "description": ["Dataset acquired on 2024-01-23."],
            "activity": {"description": "Nested activity description."},
            "dataset_distribution": [{"description": ["Distribution description."]}],
        }

        sources = collect_dataset_description_sources(document)
        prompt = json.loads(build_description_mining_prompt(sources))

        self.assertEqual([source.path for source in sources], ["/description/0"])
        self.assertEqual(
            prompt,
            {
                "description_sources": [
                    {"path": "/description/0", "text": "Dataset acquired on 2024-01-23."}
                ]
            },
        )
        self.assertNotIn("activity", json.dumps(prompt))
        self.assertNotIn("dataset_distribution", json.dumps(prompt))

    def test_accepts_top_level_description_string(self):
        sources = collect_dataset_description_sources({"description": "Dataset description."})
        self.assertEqual(sources[0].path, "/description")

    def test_prompt_requires_verbatim_evidence_without_source_context_burden(self):
        self.assertIn("Do not paraphrase", DESCRIPTION_FACT_MINING_SYSTEM_PROMPT)
        self.assertIn("non-contiguous text", DESCRIPTION_FACT_MINING_SYSTEM_PROMPT)
        self.assertNotIn("source_context", DESCRIPTION_FACT_MINING_SYSTEM_PROMPT)

    def test_validates_each_fact_and_drops_bad_or_duplicate_facts(self):
        sources = collect_dataset_description_sources(
            {"description": ["Acquisition frequency was 500 MHz."]}
        )
        raw = RawDescriptionFacts.model_validate(
            {
                "facts": [
                    {
                        "source_description_path": "/description/0",
                        "category": "measurement_condition",
                        "role": "parameter",
                        "claim": "Acquisition frequency was 500 MHz.",
                        "evidence_text": "frequency was 500 MHz",
                    },
                    {
                        "source_description_path": "/description/0",
                        "category": "measurement_condition",
                        "role": "parameter",
                        "claim": "Acquisition frequency was 500 MHz.",
                        "evidence_text": "frequency was 500 MHz",
                    },
                    {
                        "source_description_path": "/description/0",
                        "category": "invented_signal",
                        "role": "other_metadata",
                        "claim": "Invalid category.",
                        "evidence_text": "500 MHz",
                    },
                    {
                        "source_description_path": "/description/0",
                        "category": "instrument_signal",
                        "role": "parameter",
                        "claim": "Unsupported text.",
                        "evidence_text": "not copied from the description",
                    },
                    {
                        "source_description_path": "/description/99",
                        "category": "other",
                        "role": "other_metadata",
                        "claim": "Unknown source.",
                        "evidence_text": "500 MHz",
                    },
                ]
            }
        )

        facts, rejections = validate_description_facts(raw, sources)

        self.assertEqual(len(facts), 1)
        self.assertTrue(facts[0].candidate_id.startswith("description:"))
        self.assertEqual(facts[0].start_idx, 12)
        self.assertEqual(
            [rejection.reason for rejection in rejections],
            [
                "duplicate_fact",
                "unknown_evidence_category",
                "evidence_text_not_verbatim",
                "unknown_source_description_path",
            ],
        )
        self.assertEqual(facts[0].source_context, "Acquisition frequency was 500 MHz.")

    def test_adapter_is_stable_portable_and_does_not_mutate_original_context(self):
        sources = collect_dataset_description_sources(
            {"description": ["Acquisition frequency was 500 MHz."]}
        )
        raw = RawDescriptionFacts.model_validate(
            {
                "facts": [
                    {
                        "source_description_path": "/description/0",
                        "category": "instrument_signal",
                        "role": "parameter",
                        "claim": "Acquisition frequency was 500 MHz.",
                        "evidence_text": "Acquisition frequency was 500 MHz.",
                    }
                ]
            }
        )
        facts, _ = validate_description_facts(raw, sources)
        repeated, _ = validate_description_facts(raw, sources)
        original = RoutedEvidenceContext(
            portable_evidence=[
                EvidenceCandidate(
                    category="other",
                    role="other_metadata",
                    claim="Source fact.",
                    evidence_text="Source fact.",
                )
            ]
        )

        augmented = augment_evidence_context_with_description_facts(original, facts)

        self.assertEqual(facts[0].candidate_id, repeated[0].candidate_id)
        self.assertEqual(len(original.portable_evidence), 1)
        self.assertEqual(len(augmented.portable_evidence), 2)
        candidate = augmented.portable_evidence[-1]
        self.assertEqual(candidate.category, "instrument_signal")
        self.assertEqual(candidate.role, "parameter")
        self.assertEqual(candidate.source_context, "Acquisition frequency was 500 MHz.")
        self.assertEqual(candidate.file_path, "draft-description:/description/0")
        self.assertEqual(candidate.evidence_match_score, 0.0)


if __name__ == "__main__":
    unittest.main()
