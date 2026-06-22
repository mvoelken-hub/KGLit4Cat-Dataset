from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.domain.extraction.extraction_context import BaseExtractionModel, DefinedTerm, QualitativeAttribute, QuantitativeAttribute
from app.domain.semantics import VocabQuery


QUDT_QUANTITY_KIND_VOCAB = "http://qudt.org/vocab/quantitykind"
QUDT_UNIT_VOCAB = "http://qudt.org/vocab/unit"
QUDT_SCHEMA_VOCAB = "http://qudt.org/schema/qudt"
QUDT_QUANTITY_URI = "https://qudt.org/schema/qudt/Quantity"
QUDT_QUANTITY_KIND_URI = "https://qudt.org/schema/qudt/QuantityKind"
QUDT_UNIT_URI = "https://qudt.org/schema/qudt/Unit"
QUDT_QUANTITY_KIND_RDF_TYPE = "qudt__QuantityKind"
QUDT_UNIT_RDF_TYPE = "qudt__Unit"
DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS = [
    "https://w3id.org/nfdi4cat/voc4cat",
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


class VocabularyQueryFormulation(BaseModel):
    query: str = Field(default="", description="A concise vocabulary search phrase distilled from the source value and its semantic context.")
    reason: str = ""


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
You select a single controlled-vocabulary term that normalizes a metadata field, or return null.
Match on the physical quantity or concept the field actually measures, using the source value AND the
semantic context (dataset/entity/attribute title and description). Read each candidate's label AND definition,
not just its URI fragment.
When several candidates describe the same kind of quantity or concept, prefer the most general/plain one over a
more specific or named variant of it. But "most general" only applies among candidates that are the same
quantity - it never justifies picking a wrong-domain term.
Hard rules - return null for selected_uri when ANY of these hold:
- No candidate describes the same physical quantity or concept as the field. Sharing a token or a symbol with
  the source value is not a match. Common traps include bare axis or column codes, incidental words, and unit
  symbols or mathematical symbols; a single letter or a unit symbol by itself is never a quantity kind.
- An ordinal or extremum modifier (first/last/minimum/maximum) by itself is not a quantity kind. However, when
  it qualifies a field whose semantic context or source value clearly identifies the underlying physical
  quantity, the ordinal does NOT disqualify the match - select the candidate that matches that underlying
  quantity. Only abstain when the ordinal is the ONLY signal and no underlying quantity can be identified.
- The candidate belongs to a different scientific or engineering domain than the measurement described in the
  context (for example, a radioactivity, electrical-impedance, typography/printing, aerospace, or oceanography
  term used for a measurement in an unrelated field). A different-domain candidate is not a match even if it
  is the most general available - return null.
- The best candidate is only a superficial or adjacent match rather than the same quantity or concept.
- You are not confident the candidate is the same quantity. In that case return null with confidence 0 and a
  short reason naming the trap, rather than forcing a match.
Use only candidate URIs from the prompt. Always include selected_uri, confidence (0.0-1.0), and reason.
You MUST NOT select a candidate you just argued does not fit - return null instead. Return only JSON.
"""


VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT = """
You create a short vocabulary search query after an initial deterministic search failed.
Use the source value and local context only. Return only JSON.
"""

VOCAB_QUERY_FORMULATION_SYSTEM_PROMPT = """
You rewrite a metadata field value into a short, on-target vocabulary search phrase.
Use the source value AND the semantic context (dataset/entity/attribute title and description) to identify the
physical quantity or concept the field measures, then return that concept as a plain search phrase a controlled
vocabulary would label (e.g. a unit written as a symbol or abbreviation -> the full unit name).
Rules:
- Always reduce the value to the underlying physical quantity or concept. Never echo the raw value, an axis or
  column code, or a unit token.
- A unit symbol or abbreviation appearing in the value (e.g. %, cm, Hz, K, 1/cm) describes the unit of
  measurement, not the physical quantity. Strip it entirely from the formulated phrase - do not incorporate
  the unit name into the quantity search phrase. For example, a field named "length %" formulates as "length",
  not "length percentage"; a field named "voltage Hz" formulates as "voltage", not "voltage hertz".
- When the value is an axis or column label (for example a single letter paired with a unit, or an ordinal like
  first/last/min/max applied to an axis), name the physical quantity that axis or column measures, using the
  semantic context (what the dataset/entity actually records).
- A unit written as "1/X" or "X^-1" denotes the reciprocal of unit X; formulate it as "reciprocal X" (e.g.
  "1/m" -> "reciprocal metre"). Likewise a value with a reciprocal unit usually measures the quantity whose
  standard unit is that reciprocal unit.
- Output the plain concept name only (1-6 words). Do NOT echo ordinals, raw numbers, or the literal field name
  unless they ARE the concept.
- If you cannot identify the concept, still return your best short phrase based on the context.
Return only JSON.
"""


def build_query_formulation_prompt(
    *,
    source_value: str,
    source_context: dict[str, Any],
) -> str:
    return "".join(
        text
        for _, text in build_query_formulation_prompt_components(
            source_value=source_value,
            source_context=source_context,
        )
    )


def build_query_formulation_prompt_components(
    *,
    source_value: str,
    source_context: dict[str, Any],
) -> list[tuple[str, str]]:
    return [
        ("source_value", "Source value:\n" f"{source_value}\n\n"),
        ("source_context", "Source context JSON:\n" f"{source_context}\n\n"),
        (
            "formulation_instruction",
            "Return the short vocabulary search phrase for this field value.",
        ),
    ]


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
    return "".join(
        text
        for _, text in build_candidate_selection_prompt_components(
            source_value=source_value,
            source_context=source_context,
            candidates=candidates,
        )
    )


def build_candidate_selection_prompt_components(
    *,
    source_value: str,
    source_context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    return [
        ("source_value", "Source value:\n" f"{source_value}\n\n"),
        ("source_context", "Source context JSON:\n" f"{source_context}\n\n"),
        ("candidate_terms", "Candidate terms JSON:\n" f"{candidates}\n\n"),
        (
            "selection_instruction",
            "Select the best candidate URI, or return null if none fits.",
        ),
    ]


def build_object_grounding_selection_prompt(
    *,
    object_identifier: str,
    object_kind: str,
    raw_type: str,
    source_context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    return "".join(
        text
        for _, text in build_object_grounding_selection_prompt_components(
            object_identifier=object_identifier,
            object_kind=object_kind,
            raw_type=raw_type,
            source_context=source_context,
            candidates=candidates,
        )
    )


def build_object_grounding_selection_prompt_components(
    *,
    object_identifier: str,
    object_kind: str,
    raw_type: str,
    source_context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    return [
        ("object_identifier", "Object identifier:\n" f"{object_identifier}\n\n"),
        ("object_kind", "Object kind:\n" f"{object_kind}\n\n"),
        ("raw_object_type", "Raw object type:\n" f"{raw_type}\n\n"),
        ("source_context", "Source context JSON:\n" f"{source_context}\n\n"),
        ("candidate_terms", "Candidate voc4cat terms JSON:\n" f"{candidates}\n\n"),
        (
            "selection_instruction",
            "Pick the single candidate that best matches the object's type, or return null if no "
            "candidate is a clear fit.",
        ),
    ]


def build_fallback_query_prompt(
    *,
    source_value: str,
    source_context: dict[str, Any],
    failed_candidates: list[dict[str, Any]],
) -> str:
    return "".join(
        text
        for _, text in build_fallback_query_prompt_components(
            source_value=source_value,
            source_context=source_context,
            failed_candidates=failed_candidates,
        )
    )


def build_fallback_query_prompt_components(
    *,
    source_value: str,
    source_context: dict[str, Any],
    failed_candidates: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    return [
        (
            "fallback_reason",
            "The deterministic vocabulary query did not produce a fitting candidate.\n\n",
        ),
        ("source_value", "Source value:\n" f"{source_value}\n\n"),
        ("source_context", "Source context JSON:\n" f"{source_context}\n\n"),
        ("failed_candidates", "Failed candidates JSON:\n" f"{failed_candidates}\n\n"),
        (
            "fallback_query_instruction",
            "Create a better short vector/fulltext query for the same vocabulary.",
        ),
    ]
