from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.domain.extraction.extraction_context import BaseExtractionModel, DefinedTerm, QualitativeAttribute, QuantitativeAttribute
from app.domain.semantics import VocabQuery


QUDT_QUANTITY_KIND_VOCAB = "http://qudt.org/vocab/quantitykind"
QUDT_UNIT_VOCAB = "http://qudt.org/vocab/unit"
QUDT_QUANTITY_KIND_RDF_TYPE = "qudt__QuantityKind"
QUDT_UNIT_RDF_TYPE = "qudt__Unit"
DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS = [
    "https://w3id.org/nfdi4cat/voc4cat",
    "http://purl.obolibrary.org/obo/chmo.owl",
    "http://nmrML.org/nmrCV",
]


class VocabularyCandidateSelection(BaseModel):
    selected_uri: str | None = Field(
        default=None,
        description="URI of the selected candidate, or null when no candidate fits.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class VocabularyFallbackQuery(BaseModel):
    vector_query: str | None = None
    fulltext_query: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def require_query(self) -> "VocabularyFallbackQuery":
        if not self.vector_query and not self.fulltext_query:
            raise ValueError("At least one fallback query string is required.")
        return self


class VocabularyTermMapping(BaseModel):
    source_value: str
    vocabulary_identifier: str | None = None
    rdf_type: str | None = None
    selected_uri: str | None = None
    selected_title: str | None = None
    confidence: float = 0.0
    reason: str = ""


class QuantityNormalization(BaseModel):
    quantity: QuantitativeAttribute
    quantity_kind: VocabularyTermMapping | None = None
    unit: VocabularyTermMapping | None = None


class QualitativeAttributeNormalization(BaseModel):
    attribute: QualitativeAttribute
    term: VocabularyTermMapping | None = None


class ProfileFieldNormalization(BaseModel):
    json_path: str
    field_name: str
    source_value: str
    term: VocabularyTermMapping | None = None


class GroundedExtractionObject(BaseModel):
    """Wraps an extraction object together with its voc4cat-grounded type.

    `defined_term` is populated when object-grounding normalization picked a voc4cat
    term for the object's `type`; otherwise the raw `type` string remains the
    authoritative label. This model is constructed only after grounding runs and
    never lives inside the original `BaseExtractionModel`/`TracedExtractionObject`
    chain, so untouched objects keep their original shape.
    """

    object_identifier: str
    object_kind: str
    extracted_object: BaseExtractionModel
    source_value: str = Field(..., description="The original raw type string of the object.")
    defined_term: DefinedTerm | None = Field(
        default=None,
        description="voc4cat term picked during object-grounding normalization, if any.",
    )
    confidence: float = 0.0
    reason: str = ""


class ExtractionNormalization(BaseModel):
    quantities: list[QuantityNormalization] = Field(default_factory=list)
    qualitative_attributes: list[QualitativeAttributeNormalization] = Field(default_factory=list)
    profile_fields: list[ProfileFieldNormalization] = Field(default_factory=list)
    grounded_objects: list[GroundedExtractionObject] = Field(default_factory=list)
    # `object_groundings` is kept as an alias view of the same records so existing
    # consumers that filter on `kind` still see the per-object grounding data.
    object_groundings: list[GroundedExtractionObject] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_object_groundings(self) -> "ExtractionNormalization":
        if self.object_groundings and not self.grounded_objects:
            self.grounded_objects = list(self.object_groundings)
        elif self.grounded_objects and not self.object_groundings:
            self.object_groundings = list(self.grounded_objects)
        return self


VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT = """
You select controlled-vocabulary terms for metadata normalization.
Return null for selected_uri unless one candidate clearly represents the supplied source value.
Use only candidate URIs from the prompt. Return only JSON.
"""


VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT = """
You create a short vocabulary search query after an initial deterministic search failed.
Use the source value and local context only. Return only JSON.
"""


VOCAB_OBJECT_GROUNDING_SELECTION_SYSTEM_PROMPT = """
You pick the single best voc4cat term that matches the type of a scientific metadata object.
The object's raw `type` string and its description/keywords are the evidence for what kind of
thing it is (e.g., a Dataset, a Method, a Resource, an Instrument, a Sample). The candidate
list contains voc4cat Concept terms with prefLabels and definitions.
Return null unless one candidate is a clear semantic match for the object's type. When in
doubt, return null and keep the original raw type string. Return only JSON.
"""


def build_quantity_kind_vocab_query(quantity: QuantitativeAttribute) -> VocabQuery:
    query_text = " ".join(
        part
        for part in (quantity.quantity_kind, quantity.identifier, quantity.unit)
        if part
    )
    return VocabQuery(
        rdf_type=QUDT_QUANTITY_KIND_RDF_TYPE,
        vector_query=query_text,
        fulltext_query=quantity.quantity_kind or query_text,
        vector_top_k=12,
        fulltext_top_k=12,
        seed_top_k=6,
        max_hops=1,
    )


def build_unit_vocab_query(quantity: QuantitativeAttribute) -> VocabQuery:
    query_text = " ".join(
        part
        for part in (quantity.unit, quantity.quantity_kind, quantity.identifier)
        if part
    )
    return VocabQuery(
        rdf_type=QUDT_UNIT_RDF_TYPE,
        vector_query=query_text,
        fulltext_query=quantity.unit or query_text,
        vector_top_k=12,
        fulltext_top_k=12,
        seed_top_k=6,
        max_hops=1,
    )


def build_qualitative_vocab_query(
    attribute: QualitativeAttribute,
    *,
    rdf_type: str,
) -> VocabQuery:
    query_text = f"{attribute.title}: {attribute.value}".strip(": ")
    return VocabQuery(
        rdf_type=rdf_type,
        vector_query=query_text,
        fulltext_query=query_text,
        vector_top_k=6,
        fulltext_top_k=6,
        seed_top_k=3,
        max_hops=1,
        max_statements_per_seed=20,
    )


def build_candidate_selection_prompt(
    *,
    source_value: str,
    source_context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    return (
        "Source value:\n"
        f"{source_value}\n\n"
        "Source context JSON:\n"
        f"{source_context}\n\n"
        "Candidate terms JSON:\n"
        f"{candidates}\n\n"
        "Select the best candidate URI, or return null if none fits."
    )


def build_object_grounding_selection_prompt(
    *,
    object_identifier: str,
    object_kind: str,
    raw_type: str,
    source_context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    return (
        "Object identifier:\n"
        f"{object_identifier}\n\n"
        "Object kind:\n"
        f"{object_kind}\n\n"
        "Raw object type:\n"
        f"{raw_type}\n\n"
        "Source context JSON:\n"
        f"{source_context}\n\n"
        "Candidate voc4cat terms JSON:\n"
        f"{candidates}\n\n"
        "Pick the single candidate that best matches the object's type, or return null if no "
        "candidate is a clear fit."
    )


def build_fallback_query_prompt(
    *,
    source_value: str,
    source_context: dict[str, Any],
    failed_candidates: list[dict[str, Any]],
) -> str:
    return (
        "The deterministic vocabulary query did not produce a fitting candidate.\n\n"
        "Source value:\n"
        f"{source_value}\n\n"
        "Source context JSON:\n"
        f"{source_context}\n\n"
        "Failed candidates JSON:\n"
        f"{failed_candidates}\n\n"
        "Create a better short vector/fulltext query for the same vocabulary."
    )
