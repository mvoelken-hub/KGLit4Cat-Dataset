from difflib import SequenceMatcher
import re
from typing import Any, Literal, TypeVar
from pydantic import BaseModel, Field, model_validator

T = TypeVar("T", bound=BaseModel)

# Attribute classes

class QuantitativeAttribute(BaseModel):
    """A quantity that has been measured, calculated, or selected."""
    identifier: str = Field(..., description="Unique identifier for the quantity.")
    value: str = Field(..., description="Numerical value of the quantity.")
    unit: str = Field(..., description="Unit of measurement for the quantity.")
    quantity_kind: str = Field(..., description="Kind of quantity (e.g., temperature, pressure).")

class QualitativeAttribute(BaseModel):
    """A qualitative attribute that has been observed or selected."""
    title: str = Field(..., description="Descriptive title of the qualitative attribute.")
    value: str = Field(..., description="Literal value of the qualitative attribute.")

# Extraction classes


class DefinedTerm(BaseModel):
    """A term picked from a controlled vocabulary, used to enrich a type property.

    Mirrors the dcat-ap-plus `DefinedTerm` class: a URI, an optional prefLabel, and
    the source controlled vocabulary URL.
    """

    id: str = Field(..., description="URI of the term in the controlled vocabulary.")
    title: str | None = Field(
        default=None,
        description="Preferred label (skos:prefLabel) of the term, if known.",
    )
    from_CV: str | None = Field(
        default=None,
        description="Identifier of the source controlled vocabulary.",
    )


class BaseExtractionModel(BaseModel):
    """Common shape for extraction context objects."""
    identifier: str = Field(..., description="Unique identifier for the extraction object.")
    description: str = Field(..., description="Description of the extraction object.")
    keywords: list[str] = Field(default_factory=list, description="List of keywords associated with the extraction object.")
    has_quantitative_attributes: list[QuantitativeAttribute] = Field(default_factory=list, description="List of quantitative attributes associated with the extraction object.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the extraction object.")
    type: str = Field("unknown", description="Type of the entity (e.g., sample, model).")

    def to_embedding_text(self) -> str:
        return self._join_query_parts(
            self.description,
            " ".join(self.keywords),
            self.type,
        )

    def to_fulltext_query(self) -> str:
        return self._join_query_parts(
            " ".join(self.keywords),
            self.type,
        )

    @staticmethod
    def _join_query_parts(*parts: str | None) -> str:
        return " ".join(part for part in parts if part).strip()


class DataGeneratingActivity(BaseExtractionModel):
    """An experimental, measurement, acquisition, or processing activity that produces data about a target entity."""


class Method(BaseExtractionModel):
    """A method, plan, protocol, pulse sequence, acquisition procedure, processing routine, or instrument procedure used in the experiment."""


class EvaluatedEntity(BaseExtractionModel):
    """The actual target entity evaluated by a data-generating activity, such as a sample, material, catalyst, specimen, model, or subject."""


class AgenticEntity(BaseExtractionModel):
    """An entity with agency that can perform activities, such as a person, organization, instrument, or software system."""


class Resource(BaseExtractionModel):
    """A dataset resource or generated output, such as a file, dataset, spectrum, peak table, image, report, checksum, or data artifact."""

ExtractionObjectKind = Literal[
    "DataGeneratingActivity",
    "Method",
    "EvaluatedEntity",
    "AgenticEntity",
    "Resource",
]

models_by_kind: dict[str, type[BaseExtractionModel]] = {
    "DataGeneratingActivity": DataGeneratingActivity,
    "Method": Method,
    "EvaluatedEntity": EvaluatedEntity,
    "AgenticEntity": AgenticEntity,
    "Resource": Resource,
}

kind_by_model: dict[type[BaseExtractionModel], str] = {
    DataGeneratingActivity: "DataGeneratingActivity",
    Method: "Method",
    EvaluatedEntity: "EvaluatedEntity",
    AgenticEntity: "AgenticEntity",
    Resource: "Resource",
}


class TracedExtractionObject(BaseModel):
    """A traced extraction object pairs an extracted class instance with the specific text snippet from which it was extracted, to provide traceability."""
    object_kind: ExtractionObjectKind = Field(..., description="Type of extracted object.")
    extracted_object: BaseExtractionModel = Field(..., description="The extracted object (e.g., DataGeneratingActivity, EvaluatedEntity, AgenticEntity, Resource, Method).")
    source_text: str = Field(..., description="The specific text snippet from which the object was extracted.")

    @model_validator(mode="before")
    @classmethod
    def _parse_object_by_kind(cls, data: Any):
        if not isinstance(data, dict):
            return data
        object_kind = data.get("object_kind", "")
        extracted_object = data.get("extracted_object")

        if not isinstance(extracted_object, dict):
            return data

        model: type[BaseExtractionModel] | None = models_by_kind.get(object_kind, None)

        if not model:
            return data

        return {
            **data,
            "object_kind": object_kind,
            "extracted_object": model.model_validate(extracted_object),
        }


class ExtractionContext(BaseModel):
    """Context for metadata extraction, including activities, entities, datasets, and quantities."""
    extraction_objects: list[TracedExtractionObject] = Field(default_factory=list, description="List of extracted objects with traceability.")

    @property
    def data_generating_activities(self) -> list[DataGeneratingActivity]:
        return self._objects_of_type(DataGeneratingActivity)

    @property
    def evaluated_entities(self) -> list[EvaluatedEntity]:
        return self._objects_of_type(EvaluatedEntity)

    @property
    def agentic_entities(self) -> list[AgenticEntity]:
        return self._objects_of_type(AgenticEntity)

    @property
    def resources(self) -> list[Resource]:
        return self._objects_of_type(Resource)

    @property
    def methods(self) -> list[Method]:
        return self._objects_of_type(Method)

    def _objects_of_type(self, model_type: type[T]) -> list[T]:
        return [
            trace.extracted_object
            for trace in self.extraction_objects
            if isinstance(trace.extracted_object, model_type)
        ]


EXTRACTION_CONTEXT_SYSTEM_PROMPT = f"""
You are an expert for extracting structured metadata about scientific experiments from unstructured text.
You receive a content chunk from a file in a research data package, and your task is to extract structured information about the experimental context, including:
- Data-generating activities: measurement, acquisition, analysis, processing, or generation runs that produce information about a target.
- Evaluated entities: only the actual target of observation or evaluation, such as a sample, material, catalyst, specimen, model, or subject. Do not use evaluated_entity as a fallback class.
- Agentic entities: people, organizations, instruments, or software systems that perform or control activities.
- Resources: files, datasets, spectra, peak tables, images, reports, checksums, and generated data artifacts.
- Methods: protocols, plans, pulse sequences, acquisition procedures, processing routines, and instrument procedures.
Attach quantitative attributes (measured or calculated quantities) and qualitative attributes (observed characteristics, settings, labels, and modes) to the nearest meaningful activity, method, resource, or evaluated entity.
Do not create standalone extraction objects for low-level parameter names, header fields, table-schema rows, internal format declarations, checksums, dates, software versions, numeric settings, solvents, nuclei, frequencies, delays, averages, or acquisition modes unless the text clearly presents them as a real activity, method, resource, agent, or evaluated target.
If a line only contains technical metadata and cannot be attached usefully to a meaningful object, skip it.
Return only a valid ExtractionContext JSON object with the extracted information.
Use this output schema: {ExtractionContext.model_json_schema()}
For each extraction_objects item, set object_kind to the exact class label (DataGeneratingActivity, EvaluatedEntity, AgenticEntity, Resource, or Method) and set source_text to a short exact substring copied verbatim from the chunk that supports the extracted object. Do not paraphrase source_text. Use the metadata to get a sense of the overall context, but do not extract information from it.
Work from the chunk content only. Inspect lines individually as evidence, but consolidate related lines into a small number of meaningful experimental objects instead of producing one object per header or parameter line.
"""

class ChunkMetadata(BaseModel):
    start_idx: int = Field(..., ge=0, description="Start line index of the chunk in the original file")
    end_idx: int = Field(..., ge=0, description="End line index of the chunk in the original file")
    file_path: str = Field(..., description="Path to the file from which the chunk was extracted")
    data_package_name: str = Field(..., description="Identifier of the data package to which the file belongs")
    initial_extraction_context: ExtractionContext | None = Field(None, description="Aggregated extraction context from previous chunks of the same file, if available. This can provide additional context for extraction, but may also contain noise.")

class ChunkContext(BaseModel):
    content: str
    metadata: ChunkMetadata

def build_extraction_context_prompt(
        chunk_context: ChunkContext
) -> str:
    
    return (
        "Chunk context metadata:\n"
        f"{chunk_context.metadata.model_dump_json()}\n\n"
        f"Chunk content (residual lines after text-quality filtering):\n{chunk_context.content}\n"
        "Extract structured metadata about the experimental context from the **chunk content**."
    )

def merge_extraction_context_results(
    context_list: list[ExtractionContext]
) -> ExtractionContext:
    merged_context = ExtractionContext()
    for context in context_list:
        merged_context.extraction_objects.extend(context.extraction_objects)
    return deduplicate_extraction_context_items(merged_context)

def deduplicate_extraction_context_items(
    context: ExtractionContext
) -> ExtractionContext:
    return ExtractionContext(
        extraction_objects=deduplicate_list(context.extraction_objects)
    )


def cap_extraction_context_for_prompt(
    context: ExtractionContext,
    *,
    max_json_chars: int,
) -> ExtractionContext | None:
    if max_json_chars <= 0 or not context.extraction_objects:
        return None
    if len(context.model_dump_json()) <= max_json_chars:
        return context

    capped_objects: list[TracedExtractionObject] = []
    for trace in reversed(context.extraction_objects):
        candidate_objects = [trace, *capped_objects]
        candidate = ExtractionContext(extraction_objects=candidate_objects)
        if len(candidate.model_dump_json()) <= max_json_chars:
            capped_objects = candidate_objects
            continue

        if capped_objects:
            continue

        truncated = _fit_traced_extraction_object(
            trace,
            max_json_chars=max_json_chars,
        )
        if truncated is not None:
            capped_objects = [truncated]
        break

    if not capped_objects:
        return None
    return ExtractionContext(extraction_objects=capped_objects)


def _fit_traced_extraction_object(
    trace: TracedExtractionObject,
    *,
    max_json_chars: int,
) -> TracedExtractionObject | None:
    low = 0
    high = max(
        len(getattr(trace.extracted_object, "description", "")),
        len(trace.source_text),
    )
    best: TracedExtractionObject | None = None
    while low <= high:
        mid = (low + high) // 2
        candidate_trace = _truncate_traced_extraction_object(
            trace,
            max_text_chars=mid,
        )
        candidate_context = ExtractionContext(extraction_objects=[candidate_trace])
        if len(candidate_context.model_dump_json()) <= max_json_chars:
            best = candidate_trace
            low = mid + 1
        else:
            high = mid - 1
    return best


def _truncate_traced_extraction_object(
    trace: TracedExtractionObject,
    *,
    max_text_chars: int,
) -> TracedExtractionObject:
    extracted_object = trace.extracted_object
    updates = {}
    if hasattr(extracted_object, "description"):
        updates["description"] = _truncate_text(
            extracted_object.description,
            max_text_chars=max_text_chars,
        )
    if hasattr(extracted_object, "keywords"):
        updates["keywords"] = extracted_object.keywords[:10]

    return trace.model_copy(
        update={
            "extracted_object": extracted_object.model_copy(update=updates),
            "source_text": _truncate_text(
                trace.source_text,
                max_text_chars=max_text_chars,
            ),
        }
    )


def _truncate_text(text: str, *, max_text_chars: int) -> str:
    if max_text_chars <= 0:
        return ""
    if len(text) <= max_text_chars:
        return text
    if max_text_chars <= 3:
        return text[:max_text_chars]
    return text[: max_text_chars - 3].rstrip() + "..."

# Helper

def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-./]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def token_set(s: str) -> set[str]:
    return set(norm_text(s).split())

def normalized_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(sorted(norm_text(str(item)) for item in value if str(item).strip()))
    return norm_text(str(value or ""))

def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def string_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def _quantitative_attribute_key(attr: Any) -> tuple[str, str, str, str]:
    data = attr.model_dump() if isinstance(attr, BaseModel) else dict(attr)
    return (
        normalized_value(data.get("identifier")),
        normalized_value(data.get("value")),
        normalized_value(data.get("unit")),
        normalized_value(data.get("quantity_kind")),
    )


def _qualitative_attribute_key(attr: Any) -> tuple[str, str]:
    data = attr.model_dump() if isinstance(attr, BaseModel) else dict(attr)
    return (
        normalized_value(data.get("title")),
        normalized_value(data.get("value")),
    )


def _attribute_keys(data: dict[str, Any], field: str) -> set[tuple[Any, ...]]:
    key_func = (
        _quantitative_attribute_key
        if field == "has_quantitative_attributes"
        else _qualitative_attribute_key
    )
    keys: set[tuple[Any, ...]] = set()
    for attr in data.get(field, []):
        key = key_func(attr)
        if any(key):
            keys.add(key)
    return keys


def _keyword_text(data: dict[str, Any]) -> str:
    return " ".join(sorted(normalized_value(keyword) for keyword in data.get("keywords", [])))


def are_probably_same_item(a: BaseModel, b: BaseModel) -> bool:
    ad = a.model_dump()
    bd = b.model_dump()
    if isinstance(a, TracedExtractionObject) and isinstance(b, TracedExtractionObject):
        if a.object_kind != b.object_kind:
            return False
        ad = a.extracted_object.model_dump()
        bd = b.extracted_object.model_dump()

    a_id = norm_text(ad.get("identifier", ""))
    b_id = norm_text(bd.get("identifier", ""))

    a_type = norm_text(ad.get("type", ""))
    b_type = norm_text(bd.get("type", ""))

    a_desc = norm_text(ad.get("description", ""))
    b_desc = norm_text(bd.get("description", ""))

    a_kw = _keyword_text(ad)
    b_kw = _keyword_text(bd)

    shared_quantities = _attribute_keys(ad, "has_quantitative_attributes") & _attribute_keys(
        bd,
        "has_quantitative_attributes",
    )
    shared_qualities = _attribute_keys(ad, "has_qualitative_attributes") & _attribute_keys(
        bd,
        "has_qualitative_attributes",
    )

    # 1. Strong identifier match
    if a_id and b_id and a_id == b_id:
        return True

    # 2. Very similar identifiers
    if a_id and b_id and string_sim(a_id, b_id) >= 0.92:
        return True

    # 3. Shared object type plus matching attributes can identify repeated objects
    if a_type and b_type and a_type == b_type and (shared_quantities or shared_qualities):
        return True

    if (shared_quantities or shared_qualities) and (
        string_sim(a_desc, b_desc) >= 0.65
        or jaccard(a_kw, b_kw) >= 0.50
    ):
        return True

    # 3. Description + keyword overlap
    if string_sim(a_desc, b_desc) >= 0.90:
        return True

    if jaccard(a_desc, b_desc) >= 0.75:
        return True

    if jaccard(a_kw, b_kw) >= 0.80 and string_sim(a_desc, b_desc) >= 0.70:
        return True

    return False


def merge_items(a: T, b: T) -> T:
    if isinstance(a, TracedExtractionObject) and isinstance(b, TracedExtractionObject):
        merged_object = merge_items(a.extracted_object, b.extracted_object)
        source_texts = list(dict.fromkeys(text for text in (a.source_text, b.source_text) if text))
        source_text = "\n...\n".join(sorted(source_texts, key=len, reverse=True))
        return type(a)(
            object_kind=a.object_kind,
            extracted_object=merged_object,
            source_text=source_text,
        )

    data = a.model_dump()
    other = b.model_dump()

    # Prefer longer/more informative description
    if len(other.get("description", "")) > len(data.get("description", "")):
        data["description"] = other["description"]

    # Merge keywords
    data["keywords"] = sorted(set(data.get("keywords", [])) | set(other.get("keywords", [])))

    # Merge attributes by normalized semantic keys
    for field, key_func in [
        ("has_quantitative_attributes", _quantitative_attribute_key),
        ("has_qualitative_attributes", _qualitative_attribute_key),
    ]:
        existing = data.get(field, [])
        seen = {
            key_func(x)
            for x in existing
        }

        for attr in other.get(field, []):
            key = key_func(attr)
            if key not in seen:
                existing.append(attr)
                seen.add(key)

        data[field] = existing

    return type(a)(**data)


def deduplicate_list(items: list[T]) -> list[T]:
    result = []

    for item in items:
        matched = False

        for i, existing in enumerate(result):
            if are_probably_same_item(existing, item):
                result[i] = merge_items(existing, item)
                matched = True
                break

        if not matched:
            result.append(item)

    return result
