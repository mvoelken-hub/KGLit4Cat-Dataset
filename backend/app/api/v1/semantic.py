from fastapi import APIRouter, Depends, Path, UploadFile, File, Form, HTTPException, status
from typing import Annotated
from pydantic import HttpUrl, ValidationError

from app.dependencies import get_semantic_service
from app.domain.semantics import VocabNotFoundError

from app.api.v1.schemas import (
    VocabSchemeInfoResponse,
    VocabTermSchemeResponse,
    VocabEmbeddingUpdateResponse,
    VocabQueryRequest,
    VocabQueryResultResponse,
    _vocab_query_result_response,
)

from app.services.semantic_service import SemanticService

router = APIRouter(prefix="/semantic", tags=["Semantics"])

@router.post("/vocabularies", response_model=VocabSchemeInfoResponse, status_code=status.HTTP_201_CREATED)
async def import_vocabulary(
    rdf_source: Annotated[HttpUrl | UploadFile, Form(...)],
    identifier: Annotated[str, Form(...)],
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    try:
        vocab_scheme_info = await semantic_service.import_vocabulary(rdf_source, identifier)
        resp = VocabSchemeInfoResponse(
            identifier=vocab_scheme_info.identifier,
            source=vocab_scheme_info.source,
            rdf_format=vocab_scheme_info.rdf_format,
            num_triples=vocab_scheme_info.num_triples,
            description=vocab_scheme_info.description,
            vocab_term_schemes=[
                VocabTermSchemeResponse(
                    rdf_type=term_scheme.rdf_type,
                    properties=term_scheme.properties,
                    applicable_relationships=term_scheme.applicable_relationships,
                    count=term_scheme.count
                )
                for term_scheme in vocab_scheme_info.vocab_term_schemes
            ]
        )
        return resp
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    
@router.get("/vocabularies", response_model=list[str])
async def list_vocabularies(
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    return await semantic_service.list_vocabularies()

@router.post("/vocabularies/query/{identifier:path}", response_model=VocabQueryResultResponse)
async def query_vocabulary(
    query_request: VocabQueryRequest,
    identifier: str = Path(..., description="The identifier of the vocabulary scheme to query."),
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    try:
        result = await semantic_service.query_vocabulary(identifier, query_request.to_domain())
        return _vocab_query_result_response(result)
    except VocabNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

@router.post("/vocabularies/embeddings/{identifier:path}", response_model=VocabEmbeddingUpdateResponse)
async def check_pending_embedding_updates(
    identifier: str = Path(..., description="The identifier of the vocabulary scheme to check for pending embedding updates."),
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    try:
        pending_updates, task_status = await semantic_service.generate_embeddings_for_vocabulary(identifier)

        return VocabEmbeddingUpdateResponse(
            pending_updates=pending_updates,
            task_status=task_status
        )
    
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

@router.get("/vocabularies/{identifier:path}", response_model=VocabSchemeInfoResponse)
async def get_vocabulary(
    identifier: str = Path(..., description="The identifier of the vocabulary scheme to retrieve."),
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    vocab_scheme_info = await semantic_service.get_vocabulary(identifier)
    if vocab_scheme_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vocabulary scheme with identifier '{identifier}' not found.",
        )
    resp = VocabSchemeInfoResponse(
        identifier=vocab_scheme_info.identifier,
        source=vocab_scheme_info.source,
        rdf_format=vocab_scheme_info.rdf_format,
        num_triples=vocab_scheme_info.num_triples,
        description=vocab_scheme_info.description,
        vocab_term_schemes=[
            VocabTermSchemeResponse(
                rdf_type=term_scheme.rdf_type,
                properties=term_scheme.properties,
                applicable_relationships=term_scheme.applicable_relationships,
                count=term_scheme.count
            )
            for term_scheme in vocab_scheme_info.vocab_term_schemes
        ]
    )
    return resp

@router.delete("/vocabularies/{identifier:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vocabulary(
    identifier: str = Path(..., description="The identifier of the vocabulary scheme to delete."),
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    await semantic_service.delete_vocabulary(identifier)
