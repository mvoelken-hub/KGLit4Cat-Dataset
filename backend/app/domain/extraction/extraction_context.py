from difflib import SequenceMatcher
from hashlib import sha1
import re
from typing import Literal, TypeVar
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

class DataGeneratingActivity(BaseModel):
    """An Activity (process) that has the objective to produce information (in form of a dataset) about another Activity or Entity."""
    identifier: str = Field(..., description="Unique identifier for the activity.")
    description: str = Field(..., description="Description of the activity.")
    keywords: list[str] = Field(default_factory=list, description="List of keywords associated with the activity.")
    has_quantitative_attributes: list[QuantitativeAttribute] = Field(default_factory=list, description="List of quantitative attributes associated with the activity.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the activity.")

class Method(BaseModel):
    """A method, Plan, Sequence, or Protocol that has been used in the experiment."""
    identifier: str = Field(..., description="Unique identifier for the method.")
    description: str = Field(..., description="Description of the method.")
    keywords: list[str] = Field(default_factory=list, description="List of keywords associated with the method.")
    has_quantitative_attributes: list[QuantitativeAttribute] = Field(default_factory=list, description="List of quantitative attributes associated with the method.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the method.")
    
class EvaluatedEntity(BaseModel):
    """An entity that has been evaluated."""
    identifier: str = Field(..., description="Unique identifier for the entity.")
    description: str = Field(..., description="Description of the entity.")
    type: str = Field(..., description="Type of the entity (e.g., sample, model).")
    has_quantitative_attributes: list[QuantitativeAttribute] = Field(default_factory=list, description="List of quantitative attributes associated with the entity.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the entity.")

    @model_validator(mode="before")
    @classmethod
    def _default_type(cls, data):
        if isinstance(data, dict) and not data.get("type"):
            return {**data, "type": "unknown"}
        return data

class AgenticEntity(BaseModel):
    """An entity that has agency, meaning it can perform activities."""
    identifier: str = Field(..., description="Unique identifier for the agentic entity.")
    description: str = Field(..., description="Description of the agentic entity.")
    type: str = Field(..., description="Type of the agentic entity (e.g., person, organization).")
    has_quantitative_attributes: list[QuantitativeAttribute] = Field(default_factory=list, description="List of quantitative attributes associated with the agentic entity.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the agentic entity.")

    @model_validator(mode="before")
    @classmethod
    def _default_type(cls, data):
        if isinstance(data, dict) and not data.get("type"):
            return {**data, "type": "unknown"}
        return data

class Resource(BaseModel):
    """Resource from a dataset. Could be either a file or a dataset-level resource."""
    identifier: str = Field(..., description="Unique identifier for the resource.")
    type: str = Field(..., description="Type of the resource (e.g., file, dataset).")
    description: str = Field(..., description="Description of the resource.")
    has_quantitative_attributes: list[QuantitativeAttribute] = Field(default_factory=list, description="List of quantitative attributes associated with the resource.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the resource.")

    @model_validator(mode="before")
    @classmethod
    def _default_type(cls, data):
        if isinstance(data, dict) and not data.get("type"):
            return {**data, "type": "unknown"}
        return data
    
ExtractionObject = DataGeneratingActivity | EvaluatedEntity | AgenticEntity | Resource | Method
ExtractionObjectType = Literal[
    "data_generating_activity",
    "evaluated_entity",
    "agentic_entity",
    "resource",
    "method",
]

class TracedExtractionObject(BaseModel):
    """A traced extraction object pairs an extracted class instance with the specific text snippet from which it was extracted, to provide traceability."""
    object_type: ExtractionObjectType = Field(..., description="Type of extracted object.")
    extracted_object: ExtractionObject = Field(..., description="The extracted object (e.g., DataGeneratingActivity, EvaluatedEntity, AgenticEntity, Resource, Method).")
    source_text: str = Field(..., description="The specific text snippet from which the object was extracted.")

    @model_validator(mode="before")
    @classmethod
    def _parse_object_by_type(cls, data):
        if not isinstance(data, dict):
            return data
        object_type = data.get("object_type")
        extracted_object = data.get("extracted_object")
        if not isinstance(extracted_object, dict):
            return data
        model_by_type = {
            "data_generating_activity": DataGeneratingActivity,
            "evaluated_entity": EvaluatedEntity,
            "agentic_entity": AgenticEntity,
            "resource": Resource,
            "method": Method,
        }
        model = model_by_type.get(object_type)
        if model is None:
            return data
        return {**data, "extracted_object": model.model_validate(extracted_object)}


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

    def _objects_of_type(self, object_type: type[T]) -> list[T]:
        return [
            trace.extracted_object
            for trace in self.extraction_objects
            if isinstance(trace.extracted_object, object_type)
        ]


def extraction_object_type(item: ExtractionObject) -> str:
    if isinstance(item, DataGeneratingActivity):
        return "data_generating_activity"
    if isinstance(item, EvaluatedEntity):
        return "evaluated_entity"
    if isinstance(item, AgenticEntity):
        return "agentic_entity"
    if isinstance(item, Resource):
        return "resource"
    if isinstance(item, Method):
        return "method"
    return "unknown"


EXTRACTION_CONTEXT_SYSTEM_PROMPT = f"""
You are an expert for extracting structured metadata about scientific experiments from unstructured text.
You receive a content chunk from a file in a research data package, and your task is to extract structured information about the experimental context, including:
- Data-generating activities (processes that produce information about other activities or entities; do not treat parameter names / headers as activities)
- Evaluated entities (entities that have been evaluated in the experiment)
- Agentic entities (entities that have agency and can perform activities)
- Resources that have been generated
- Methods that have been used in the experiment
For each class, extract any quantitative attributes (measured or calculated quantities) and qualitative attributes (observed characteristics) that are mentioned in the text.
Return only a valid ExtractionContext JSON object with the extracted information.
Use this output schema: {ExtractionContext.model_json_schema()}
For each extraction_objects item, set object_type to the exact class label and set source_text to a short exact substring copied verbatim from the chunk that supports the extracted object. Do not paraphrase source_text.
Focus on extracting as much metadata as possible from one specific chunk. Work at a low level. Look at each line individually and in context.
"""

class ChunkContext(BaseModel):
    content: str
    start_idx: int = Field(..., ge=0, description="Start line index of the chunk in the original file")
    end_idx: int = Field(..., ge=0, description="End line index of the chunk in the original file")
    file_path: str = Field(..., description="Path to the file from which the chunk was extracted")
    data_package_name: str = Field(..., description="Identifier of the data package to which the file belongs")


def build_extraction_context_prompt(
        chunk_context: ChunkContext
) -> str:
    
    dataset_line = f"Dataset name: {chunk_context.data_package_name}\n" if chunk_context.data_package_name else ""
    return (
        f"{dataset_line}"
        f"File path: {chunk_context.file_path} Line-Index-Span: {chunk_context.start_idx}-{chunk_context.end_idx}\n"
        f"Chunk content (after text-quality line filtering):\n{chunk_context.content}\n"
        "Extract structured metadata about the experimental context from this chunk."
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

# Helper

def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-./]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def token_set(s: str) -> set[str]:
    return set(norm_text(s).split())

def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def string_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def are_probably_same_item(a: BaseModel, b: BaseModel) -> bool:
    ad = a.model_dump()
    bd = b.model_dump()
    if isinstance(a, TracedExtractionObject) and isinstance(b, TracedExtractionObject):
        if a.object_type != b.object_type:
            return False
        ad = a.extracted_object.model_dump()
        bd = b.extracted_object.model_dump()

    a_id = norm_text(ad.get("identifier", ""))
    b_id = norm_text(bd.get("identifier", ""))

    a_desc = norm_text(ad.get("description", ""))
    b_desc = norm_text(bd.get("description", ""))

    a_kw = " ".join(sorted(ad.get("keywords", [])))
    b_kw = " ".join(sorted(bd.get("keywords", [])))

    # 1. Strong identifier match
    if a_id and b_id and a_id == b_id:
        return True

    # 2. Very similar identifiers
    if a_id and b_id and string_sim(a_id, b_id) >= 0.92:
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
        source_text = "\n...\n".join(source_texts)
        return type(a)(
            object_type=a.object_type,
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

    # Merge attributes by normalized keys
    for field in ["has_quantitative_attributes", "has_qualitative_attributes"]:
        existing = data.get(field, [])
        seen = {
            sha1(repr(x).encode("utf-8")).hexdigest()
            for x in existing
        }

        for attr in other.get(field, []):
            h = sha1(repr(attr).encode("utf-8")).hexdigest()
            if h not in seen:
                existing.append(attr)
                seen.add(h)

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
