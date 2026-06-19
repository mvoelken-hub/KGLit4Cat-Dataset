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
    """A method, plan, protocol, acquisition procedure, processing routine, or instrument-related procedure used in the experiment."""


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
