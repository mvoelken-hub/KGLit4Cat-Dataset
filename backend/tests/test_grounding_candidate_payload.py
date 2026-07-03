from app.domain.extraction.vocabulary import build_candidate_selection_prompt_components
from app.domain.semantics import CompactVocabResource, VocabGraphStatement, VocabQueryResult, VocabSeed
from app.services.grounding_service import GroundingService


def test_candidate_records_include_semantic_fields_and_prompt_hides_route_metadata():
    result = VocabQueryResult(
        identifier="nmrCV",
        rdf_type="owl__Class",
        seeds=[
            VocabSeed(
                uri="http://nmrML.org/nmrCV#NMR:1400215",
                rdf_type="owl__Class",
                rrf_score=1.0,
                fulltext_rank=1,
            )
        ],
        graph_statements=[
            VocabGraphStatement(
                subject_uri="http://nmrML.org/nmrCV#NMR:1400215",
                predicate="rdfs__subClassOf",
                object_uri="http://nmrML.org/nmrCV#NMR:1400214",
            )
        ],
        resources={
            "http://nmrML.org/nmrCV#NMR:1400215": CompactVocabResource(
                uri="http://nmrML.org/nmrCV#NMR:1400215",
                rdf_types=["owl__Class"],
                properties={
                    "rdfs__label": "Bruker TopSpin software",
                    "obo__IAO_0000115": "Bruker software for NMR acquisition and processing.",
                    "oboInOwl__hasExactSynonym": "TOPSPIN",
                },
            ),
            "http://nmrML.org/nmrCV#NMR:1400214": CompactVocabResource(
                uri="http://nmrML.org/nmrCV#NMR:1400214",
                rdf_types=["owl__Class"],
                properties={"rdfs__label": "Bruker NMR software"},
            ),
        },
    )

    candidates = GroundingService._candidate_records(result)

    assert candidates[0]["title"] == "Bruker TopSpin software"
    assert candidates[0]["label"] == "Bruker TopSpin software"
    assert candidates[0]["definition"] == "Bruker software for NMR acquisition and processing."
    assert candidates[0]["synonyms"] == ["TOPSPIN"]
    assert candidates[0]["related_terms"] == [
        {"relation": "broader", "label": "Bruker NMR software"}
    ]

    prompt_components = build_candidate_selection_prompt_components(
        source_value="NMR software | TOPSPIN software",
        source_context={"json_path": "/was_generated_by/0/carried_out_by/1/type"},
        candidates=candidates[:1],
    )
    candidate_prompt = next(text for _, text in prompt_components if "Candidate terms JSON" in text)

    assert "Bruker software for NMR acquisition and processing." in candidate_prompt
    assert "TOPSPIN" in candidate_prompt
    assert "Bruker NMR software" in candidate_prompt
    assert "vocabulary_identifier" not in candidate_prompt
    assert "rdf_type" not in candidate_prompt


def test_candidate_related_terms_are_from_candidate_perspective():
    result = VocabQueryResult(
        identifier="voc4cat",
        rdf_type="skos__Concept",
        seeds=[
            VocabSeed(
                uri="https://w3id.org/nfdi4cat/voc4cat_0000046",
                rdf_type="skos__Concept",
                rrf_score=1.0,
            )
        ],
        graph_statements=[
            VocabGraphStatement(
                subject_uri="https://w3id.org/nfdi4cat/voc4cat_0000046",
                predicate="skos__narrower",
                object_uri="https://w3id.org/nfdi4cat/voc4cat_0000048",
            ),
            VocabGraphStatement(
                subject_uri="https://w3id.org/nfdi4cat/voc4cat_0000048",
                predicate="skos__broader",
                object_uri="https://w3id.org/nfdi4cat/voc4cat_0000046",
            ),
        ],
        resources={
            "https://w3id.org/nfdi4cat/voc4cat_0000046": CompactVocabResource(
                uri="https://w3id.org/nfdi4cat/voc4cat_0000046",
                rdf_types=["skos__Concept"],
                properties={"skos__prefLabel": "literature research"},
            ),
            "https://w3id.org/nfdi4cat/voc4cat_0000048": CompactVocabResource(
                uri="https://w3id.org/nfdi4cat/voc4cat_0000048",
                rdf_types=["skos__Concept"],
                properties={"skos__prefLabel": "literature organization"},
            ),
        },
    )

    candidates = GroundingService._candidate_records(result)

    assert "related_terms" not in candidates[0]
