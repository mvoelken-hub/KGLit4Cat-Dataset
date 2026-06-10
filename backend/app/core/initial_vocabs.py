from pydantic import BaseModel, Field, HttpUrl


class InitialVocab(BaseModel):
    rdf_source: HttpUrl = Field(..., description="The URL of the RDF source to import.")
    identifier: str = Field(..., description="A unique identifier for the vocab, used for referencing it in the system.")


INITIAL_VOCABS = [
    InitialVocab(
        rdf_source=HttpUrl("https://nfdi4cat.github.io/voc4cat/v2026-02-24/voc4cat.ttl"),
        identifier="https://w3id.org/nfdi4cat/voc4cat",
    ),
    InitialVocab(
        rdf_source=HttpUrl("https://qudt.org/vocab/quantitykind/"),
        identifier="http://qudt.org/vocab/quantitykind",
    ),
    InitialVocab(
        rdf_source=HttpUrl("https://qudt.org/vocab/unit/"),
        identifier="http://qudt.org/vocab/unit",
    ),
    InitialVocab(
        rdf_source=HttpUrl("https://qudt.org/vocab/constant"),
        identifier="http://qudt.org/vocab/constant",
    ),
]
