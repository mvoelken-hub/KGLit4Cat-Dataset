from fastapi import APIRouter, Depends, Path, UploadFile, File, Form, HTTPException, status
from typing import Annotated
from pydantic import HttpUrl

from app.dependencies import get_semantic_service
from app.domain.semantics import VocabSchemeInfo

from app.services.semantic_service import SemanticService

router = APIRouter(prefix="/semantic", tags=["Semantics"])

@router.post("/vocabularies", response_model=VocabSchemeInfo, status_code=status.HTTP_201_CREATED)
async def import_vocabulary(
    rdf_source: Annotated[HttpUrl | UploadFile, Form(...)],
    identifier: Annotated[str, Form(...)],
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    try:
        vocab_scheme_info = await semantic_service.import_vocabulary(rdf_source, identifier)
        return vocab_scheme_info
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
    
@router.get("/vocabularies/{identifier:path}", response_model=VocabSchemeInfo)
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
    return vocab_scheme_info

@router.delete("/vocabularies/{identifier:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vocabulary(
    identifier: str = Path(..., description="The identifier of the vocabulary scheme to delete."),
    semantic_service: SemanticService = Depends(get_semantic_service),
):
    await semantic_service.delete_vocabulary(identifier)