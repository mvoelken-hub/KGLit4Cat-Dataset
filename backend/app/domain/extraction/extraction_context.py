from difflib import SequenceMatcher
from hashlib import sha1
import re
from typing import TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

class Quantity(BaseModel):
    """A quantity that has been measured, calculated, or selected."""
    identifier: str = Field(..., description="Unique identifier for the quantity.")
    value: str = Field(..., description="Numerical value of the quantity.")
    unit: str = Field(..., description="Unit of measurement for the quantity.")
    quantity_kind: str = Field(..., description="Kind of quantity (e.g., temperature, pressure).")

class QualitativeAttribute(BaseModel):
    """A qualitative attribute that has been observed or selected."""
    title: str = Field(..., description="Descriptive title of the qualitative attribute.")
    value: str = Field(..., description="Literal value of the qualitative attribute.")

class DataGeneratingActivity(BaseModel):
    """An Activity (process) that has the objective to produce information (in form of a dataset) about another Activity or Entity."""
    identifier: str = Field(..., description="Unique identifier for the activity.")
    description: str = Field(..., description="Description of the activity.")
    keywords: list[str] = Field(default_factory=list, description="List of keywords associated with the activity.")
    has_quantitative_attributes: list[Quantity] = Field(default_factory=list, description="List of quantitative attributes associated with the activity.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the activity.")
    
class EvaluatedEntity(BaseModel):
    """An entity that has been evaluated."""
    identifier: str = Field(..., description="Unique identifier for the entity.")
    description: str = Field(..., description="Description of the entity.")
    keywords: list[str] = Field(default_factory=list, description="List of keywords associated with the entity.")
    has_quantitative_attributes: list[Quantity] = Field(default_factory=list, description="List of quantitative attributes associated with the entity.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the entity.")

class AgenticEntity(BaseModel):
    """An entity that has agency, meaning it can perform activities."""
    identifier: str = Field(..., description="Unique identifier for the agentic entity.")
    description: str = Field(..., description="Description of the agentic entity.")
    keywords: list[str] = Field(default_factory=list, description="List of keywords associated with the agentic entity.")
    has_quantitative_attributes: list[Quantity] = Field(default_factory=list, description="List of quantitative attributes associated with the agentic entity.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the agentic entity.")

class Dataset(BaseModel):
    """A dataset that has been generated."""
    identifier: str = Field(..., description="Unique identifier for the dataset.")
    description: str = Field(..., description="Description of the dataset.")
    keywords: list[str] = Field(default_factory=list, description="List of keywords associated with the dataset.")
    has_quantitative_attributes: list[Quantity] = Field(default_factory=list, description="List of quantitative attributes associated with the dataset.")
    has_qualitative_attributes: list[QualitativeAttribute] = Field(default_factory=list, description="List of qualitative attributes associated with the dataset.")


class ExtractionContext(BaseModel):
    """Context for metadata extraction, including activities, entities, datasets, and quantities."""
    data_generating_activities: list[DataGeneratingActivity] = Field(default_factory=list, description="List of data-generating activities.")
    evaluated_entities: list[EvaluatedEntity] = Field(default_factory=list, description="List of evaluated entities.")
    agentic_entities: list[AgenticEntity] = Field(default_factory=list, description="List of agentic entities.")
    datasets: list[Dataset] = Field(default_factory=list, description="List of datasets that have been generated.")


EXTRACTION_CONTEXT_SYSTEM_PROMPT = f"""
You are an expert for extracting structured metadata about scientific experiments from unstructured text.
You receive a content chunk from a file in a research data package, and your task is to extract structured information about the experimental context, including:
- Data-generating activities (processes that produce information about other activities or entities)
- Evaluated entities (entities that have been evaluated in the experiment)
- Agentic entities (entities that have agency and can perform activities)
- Datasets that have been generated
For each entity or activity, extract any quantitative attributes (measured or calculated quantities) and qualitative attributes (observed characteristics) that are mentioned in the text.
Return only a valid ExtractionContext JSON object with the extracted information.
Focus on extracting as much metadata as possible from one specific chunk. Work at a low level; your individual result will later be combined with the results of several such extraction steps, so you don't need to try to guess the overall context.
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
        merged_context.data_generating_activities.extend(context.data_generating_activities)
        merged_context.evaluated_entities.extend(context.evaluated_entities)
        merged_context.agentic_entities.extend(context.agentic_entities)
        merged_context.datasets.extend(context.datasets)
    return deduplicate_extraction_context_items(merged_context)

def deduplicate_extraction_context_items(
    context: ExtractionContext
) -> ExtractionContext:
    return ExtractionContext(
        data_generating_activities=deduplicate_list(context.data_generating_activities),
        evaluated_entities=deduplicate_list(context.evaluated_entities),
        agentic_entities=deduplicate_list(context.agentic_entities),
        datasets=deduplicate_list(context.datasets),
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
