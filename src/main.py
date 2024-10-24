
from db import AsyncSessionLocal, create_async_engine, create_tables
from utils import  build_faiss_indexes, configure_redis_optimally
from models import DocumentEmbedding, Document, TextEmbedding, DocumentContentResponse,  DocumentPydantic, SemanticDataTypeResponse, AllSemanticDataTypesResponse
from models import EmbeddingRequest, SemanticSearchRequest, AdvancedSemanticSearchRequest, SimilarityRequest, SimilarityResponse
from models import EmbeddingResponse, SemanticSearchResponse, AdvancedSemanticSearchResponse, AllDocumentsResponse
from models import SemanticDataType, SemanticDataTypeEmbeddingRequest, SemanticDataTypeEmbeddingResponse, SemanticSearchRequest, SemanticSearchResponse, AllSemanticDataTypesResponse
from functions import get_or_compute_embedding,  add_model_url, download_file, decompress_data, store_document_embeddings_in_db, download_models, initialize_globals
from functions import get_list_of_corpus_identifiers_from_list_of_embedding_texts, compute_embeddings_for_document, parse_submitted_document_file_into_sentence_strings_func,prepare_string_for_embedding, sophisticated_sentence_splitter, remove_pagination_breaks, truncate_string

import asyncio
import glob
import json
import os 
import random
import tempfile
import traceback
import zipfile
from datetime import datetime
from hashlib import sha3_256
from typing import List, Optional, Dict, Any
from urllib.parse import unquote
import numpy as np
import uvicorn
import fastapi
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from contextlib import asynccontextmanager
from sqlalchemy import func, select, delete,and_
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
import faiss
import fast_vector_similarity as fvs
import uvloop
from magika import Magika
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
magika = Magika() # file type 
configure_redis_optimally()

def raise_graceful_exit():
    """
    Raise a GracefulExit exception to initiate graceful shutdown.

    Preconditions:
        None

    Postconditions:
        - A GracefulExit exception is raised

    Raises:
        GracefulExit: Always raised to signal graceful shutdown
    """
    raise GracefulExit()

# Define a custom exception for graceful shutdown
class GracefulExit(BaseException):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the FastAPI application.

    Preconditions:
    - app is a valid FastAPI instance

    Postconditions:
    - Application is initialized before yielding
    - Application shutdown is initiated after yielding

    :param app: The FastAPI application instance
    :yield: Control back to the FastAPI framework
    """
    logger.info("Starting application initialization")
    await initialize_globals()
    logger.info("Application initialization complete")
    yield
    logger.info("Application shutdown initiated")


# variables
DEFAULT_LLM_NAME = os.getenv("DEFAULT_LLM_NAME", "Meta-Llama-3-8B-Instruct.Q3_K_S")
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("DEFAULT_EMBEDDING_MODEL_NAME", "nomic-embed-text-v1.5.Q6_K")
DEFAULT_MULTI_MODAL_MODEL_NAME = os.getenv("DEFAULT_MULTI_MODAL_MODEL_NAME", "llava-llama-3-8b-v1_1-int4")
USE_RESOURCE_MONITORING = os.getenv("USE_RESOURCE_MONITORING", "1") == "1"
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MAX_THOUSANDS_OF_WORDs_FOR_DOCUMENT_EMBEDDING = int(os.getenv("MAX_THOUSANDS_OF_WORDs_FOR_DOCUMENT_EMBEDDING", "100"))
DEFAULT_COMPLETION_TEMPERATURE = float(os.getenv("DEFAULT_COMPLETION_TEMPERATURE", "0.7"))
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("DEFAULT_MAX_COMPLETION_TOKENS", "1000"))
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE = int(os.getenv("DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE", "1"))
DEFAULT_EMBEDDING_POOLING_METHOD = os.getenv("DEFAULT_EMBEDDING_POOLING_METHOD", "mean")

description_string = """
Codexify lets you create custom semantic data types by defining them using natural language and examples, and then searching over them using semantic search.
"""
app = FastAPI(title="Codexify", description=description_string, docs_url="/", lifespan=lifespan)  # Set the Swagger UI to root


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://codexify.vercel.app","http://codexify.ai"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(SQLAlchemyError) 
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """
    Handle SQLAlchemy exceptions and return a JSON response.

    Preconditions:
        - 'request' is a valid FastAPI Request object
        - 'exc' is an instance of SQLAlchemyError

    Postconditions:
        - The exception is logged
        - A JSONResponse with status code 500 is returned

    :param request: The FastAPI Request object
    :param exc: The SQLAlchemyError exception
    :return: A JSONResponse with a 500 status code and an error message
    """
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "Database error occurred"})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions and return a JSON response.

    Preconditions:
        - 'request' is a valid FastAPI Request object
        - 'exc' is an instance of Exception

    Postconditions:
        - The exception is logged
        - A JSONResponse with status code 500 is returned

    :param request: The FastAPI Request object
    :param exc: The Exception instance
    :return: A JSONResponse with a 500 status code and an error message
    """
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "An unexpected error occurred"})


@app.on_event("startup")
async def pre_startup_tasks():
    logger.info("Starting pre-startup tasks")
    logger.info("Initializing globals")
    await initialize_globals()
    logger.info("Building FAISS indexes")
    await build_faiss_indexes(force_rebuild=True)
    logger.info("Downloading models")
    await download_models()
    logger.info("Pre-startup tasks completed")

@app.get("/", include_in_schema=False)
async def load_swagger_ui():
    return fastapi.templating.get_swagger_ui_html(openapi_url="/openapi.json", title=app.title, swagger_favicon_url=app.swagger_ui_favicon_url)

@app.get("/models")
async def get_list_of_models(token: str = None) -> Dict[str, List[str]]:
    
    models_dir = os.path.join(BASE_DIRECTORY, 'models')
    model_files = glob.glob(os.path.join(models_dir, "*.bin")) + glob.glob(os.path.join(models_dir, "*.gguf"))
    model_names = sorted([os.path.splitext(os.path.basename(model_file))[0] for model_file in model_files])
    return {"model_names": model_names}

@app.post("/models", response_model=Dict[str, Any])
async def add_new_model(model_url: str, token: str = None) -> Dict[str, Any]:
    """
    Add a new model to the system.

    1. The model must be in `.gguf` format.
    2. The model must be larger than 100 MB to ensure it's a valid model file.

    Parameters:
    - `model_url`: The URL of the model weight file, which must end with `.gguf`.
    - `token`: Security token (optional).

    Returns:
    A JSON object indicating the status of the model addition and download.
    """
    

    unique_id = f"add_model_{hash(model_url)}"

    try:
        decoded_model_url = unquote(model_url)
        if not decoded_model_url.endswith('.gguf'):
            return {"status": "error", "message": "Model URL must point to a .gguf file."}
        
        corrected_model_url = add_model_url(decoded_model_url)
        _, download_status = download_models()
        status_dict = {status["url"]: status for status in download_status}
        
        if corrected_model_url in status_dict:
            return {
                "status": status_dict[corrected_model_url]["status"],
                "message": status_dict[corrected_model_url]["message"]
            }
        
        return {"status": "unknown", "message": "Unexpected error."}
    except Exception as e:
        logger.error(f"An error occurred while adding the model: {str(e)}")
        return {"status": "unknown", "message": f"An unexpected error occurred: {str(e)}"}

@app.post("/calculate_similarity")
async def calculate_similarity_between_strings(request: SimilarityRequest, req: Request, token: str = None) -> SimilarityResponse:
    logger.info(f"Received request: {request}")
    request_time = datetime.utcnow()
    request.text1 = prepare_string_for_embedding(request.text1)
    request.text2 = prepare_string_for_embedding(request.text2)
    similarity_measure = request.similarity_measure.lower()
    unique_id = f"compute_similarity_{request.text1}_{request.text2}_{request.llm_model_name}_{request.embedding_pooling_method}_{similarity_measure}"
    
    
    try:
        client_ip = req.client.host if req else "localhost"
        embedding_request1 = EmbeddingRequest(text=request.text1, llm_model_name=request.llm_model_name, embedding_pooling_method=request.embedding_pooling_method)
        embedding_request2 = EmbeddingRequest(text=request.text2, llm_model_name=request.llm_model_name, embedding_pooling_method=request.embedding_pooling_method)
        embedding1_response = await get_or_compute_embedding(request=embedding_request1, req=req, client_ip=client_ip, use_verbose=False)
        embedding2_response = await get_or_compute_embedding(request=embedding_request2, req=req, client_ip=client_ip, use_verbose=False)
        embedding1 = np.array(embedding1_response["embedding"])
        embedding2 = np.array(embedding2_response["embedding"])
        if embedding1.size == 0 or embedding2.size == 0:
            raise HTTPException(status_code=400, detail="Could not calculate embeddings for the given texts")
        params = {
            "vector_1": embedding1.tolist(),
            "vector_2": embedding2.tolist(),
            "similarity_measure": similarity_measure
        }
        similarity_stats_str = fvs.py_compute_vector_similarity_stats(json.dumps(params))
        similarity_stats_json = json.loads(similarity_stats_str)
        if similarity_measure == 'all':
            similarity_score = similarity_stats_json
        else:
            similarity_score = similarity_stats_json.get(similarity_measure, None)
            if similarity_score is None:
                raise HTTPException(status_code=400, detail="Invalid similarity measure specified")
        response_time = datetime.utcnow()
        total_time = (response_time - request_time).total_seconds()
        logger.info(f"Computed similarity using {similarity_measure} in {total_time:,.2f} seconds; similarity score: {similarity_score}")
        return {
            "text1": request.text1,
            "text2": request.text2,
            "similarity_measure": similarity_measure,
            "similarity_score": similarity_score,
            "embedding1": embedding1.tolist(),
            "embedding2": embedding2.tolist()
        }
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
@app.post("/semantic-search/basic")
async def simple_semantic_search(request: SemanticSearchRequest, req: Request, token: str = None) -> SemanticSearchResponse:
                              
    global faiss_indexes, associated_texts_by_model_and_pooling_method
    request_time = datetime.utcnow()
    request.query_text = prepare_string_for_embedding(request.query_text)
    unique_id = f"semantic_search_{request.query_text}_{request.llm_model_name}_{request.embedding_pooling_method}_{request.corpus_identifier_string}_{request.number_of_most_similar_strings_to_return}"  # Unique ID for this operation
    
    faiss_indexes, associated_texts_by_model_and_pooling_method = await build_faiss_indexes(force_rebuild=True)
    try:
        faiss_index = faiss_indexes[(request.llm_model_name, request.embedding_pooling_method)]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {request.llm_model_name} and pooling method: {request.embedding_pooling_method}")
    llm_model_name = request.llm_model_name
    embedding_pooling_method = request.embedding_pooling_method
    num_results = request.number_of_most_similar_strings_to_return
    num_results_before_corpus_filter = num_results*25
    total_entries = len(associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method])  # Get the total number of entries for the model and pooling method
    num_results = min(num_results, total_entries)  # Ensure num_results doesn't exceed the total number of entries
    num_results_before_corpus_filter = min(num_results_before_corpus_filter, total_entries)  # Ensure num_results_before_corpus_filter doesn't exceed the total number of entries
    logger.info(f"Received request to find {num_results:,} most similar strings for query text: `{request.query_text}` using model: {llm_model_name}, pooling method: {embedding_pooling_method}, and corpus: {request.corpus_identifier_string}")
    try:
        logger.info(f"Computing embedding for input text: {request.query_text}")
        embedding_request = EmbeddingRequest(text=request.query_text, llm_model_name=request.llm_model_name, embedding_pooling_method=request.embedding_pooling_method, corpus_identifier_string=request.corpus_identifier_string)
        embedding_response = await get_or_compute_embedding(embedding_request, req)                
        embedding_json = embedding_response["text_embedding_dict"]["embedding_json"]
        embedding_vector = json.loads(embedding_json)
        input_embedding = np.array(embedding_vector).astype('float32').reshape(1, -1)
        faiss.normalize_L2(input_embedding)  # Normalize the input vector for cosine similarity
        results = []  # Create an empty list to store the results
        faiss_index = faiss_indexes[(llm_model_name, embedding_pooling_method)]
        associated_texts = associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]
        list_of_corpus_identifier_strings = await get_list_of_corpus_identifiers_from_list_of_embedding_texts(associated_texts, llm_model_name, embedding_pooling_method)
        logger.info(f"Searching for the most similar string in the FAISS index using {embedding_pooling_method} embeddings")
        if faiss_index is None:
            raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {llm_model_name} and pooling method: {embedding_pooling_method}")
        similarities, indices = faiss_index.search(input_embedding.reshape(1, -1), num_results_before_corpus_filter)  # Search for num_results similar strings
        for ii in range(num_results_before_corpus_filter):
            index = indices[0][ii]
            if index < len(associated_texts):
                similarity = float(similarities[0][ii])  # Convert numpy.float32 to native float
                most_similar_text = associated_texts[index]
                corpus_identifier_string = list_of_corpus_identifier_strings[index]
                if (corpus_identifier_string == request.corpus_identifier_string) and (most_similar_text != request.query_text) and (len(results) <= num_results):
                    results.append({"search_result_text": most_similar_text, "similarity_to_query_text": similarity})
            else:
                logger.warning(f"Index {index} out of range for model {llm_model_name} and pooling method {embedding_pooling_method}")
        response_time = datetime.utcnow()
        total_time = (response_time - request_time).total_seconds()
        logger.info(f"Finished searching for the most similar string in the FAISS index in {total_time:,.2f} seconds. Found {len(results):,} results, returning the top {num_results:,}.")
        logger.info(f"Found most similar strings for query string {request.query_text}: {results}")
        if len(results) == 0:
            logger.info(f"No results found for query string {request.query_text}.")
            raise HTTPException(status_code=400, detail=f"No results found for query string {request.query_text} and model {llm_model_name} and pooling method {embedding_pooling_method} and corpus {request.corpus_identifier_string}.")
        return {"query_text": request.query_text, "corpus_identifier_string": request.corpus_identifier_string, "embedding_pooling_method": embedding_pooling_method, "results": results}  # Return the response matching the SemanticSearchResponse model
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/semantic-search/advanced")
async def advanced_semantic_search(request: AdvancedSemanticSearchRequest, req: Request, token: str = None) -> AdvancedSemanticSearchResponse:
    
    global faiss_indexes, associated_texts_by_model_and_pooling_method
    request_time = datetime.utcnow()
    request.query_text = prepare_string_for_embedding(request.query_text)   
    unique_id = f"advanced_semantic_search_{request.query_text}_{request.llm_model_name}_{request.embedding_pooling_method}_{request.similarity_filter_percentage}_{request.number_of_most_similar_strings_to_return}"
          
                  
    faiss_indexes, associated_texts_by_model_and_pooling_method = await build_faiss_indexes(force_rebuild=True)
    try:
        faiss_index = faiss_indexes[(request.llm_model_name, request.embedding_pooling_method)]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {request.llm_model_name} and pooling method: {request.embedding_pooling_method}")            
    llm_model_name = request.llm_model_name
    embedding_pooling_method = request.embedding_pooling_method
    num_results_before_corpus_filter = request.number_of_most_similar_strings_to_return*25
    logger.info(f"Received request to find most similar strings for query text: `{request.query_text}` using model: {llm_model_name}")
    try:
        logger.info(f"Computing embedding for input text: {request.query_text}")
        embedding_request = EmbeddingRequest(text=request.query_text, llm_model_name=llm_model_name, embedding_pooling_method=embedding_pooling_method)
        embedding_response = await get_or_compute_embedding(embedding_request, req)                
        embedding_json = embedding_response["text_embedding_dict"]["embedding_json"]
        embedding_vector = json.loads(embedding_json)
        input_embedding = np.array(embedding_vector).astype('float32').reshape(1, -1)                
        faiss.normalize_L2(input_embedding)
        logger.info(f"Computed embedding for input text: {request.query_text}")
        final_results = []
        faiss_index = faiss_indexes[(llm_model_name, embedding_pooling_method)]
        if faiss_index is None:
            raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {llm_model_name} and pooling method: {embedding_pooling_method}")
        num_results = max([1, int((1 - request.similarity_filter_percentage) * len(associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]))])
        num_results_before_corpus_filter = min(num_results_before_corpus_filter, len(associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]))
        similarities, indices = faiss_index.search(input_embedding, num_results_before_corpus_filter)
        filtered_indices = indices[0]
        filtered_similarities = similarities[0]
        similarity_results = []
        associated_texts = associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]
        list_of_corpus_identifier_strings = await get_list_of_corpus_identifiers_from_list_of_embedding_texts(associated_texts, llm_model_name, embedding_pooling_method)
        for idx, similarity in zip(filtered_indices, filtered_similarities):
            if idx < len(associated_texts) and list_of_corpus_identifier_strings[idx] == request.corpus_identifier_string:
                associated_text = associated_texts[idx]
                similarity_results.append((similarity, associated_text))
        similarity_results = sorted(similarity_results, key=lambda x: x[0], reverse=True)[:num_results]
        for _, associated_text in similarity_results:
            embedding_request = EmbeddingRequest(text=associated_text, llm_model_name=llm_model_name, embedding_pooling_method=embedding_pooling_method)
            embedding_response = await get_or_compute_embedding(request=embedding_request, req=req, use_verbose=False)           
            embedding_json = embedding_response["text_embedding_dict"]["embedding_json"]
            embedding_vector = json.loads(embedding_json)
            comparison__embedding = np.array(embedding_vector).astype('float32').reshape(1, -1)                 
            params = {
                "vector_1": input_embedding.tolist()[0],
                "vector_2": comparison__embedding.tolist()[0],
                "similarity_measure": "all"
            }
            similarity_stats_str = fvs.py_compute_vector_similarity_stats(json.dumps(params))
            similarity_stats_json = json.loads(similarity_stats_str)
            final_results.append({
                "search_result_text": associated_text,
                "similarity_to_query_text": similarity_stats_json
            })
        num_to_return = request.number_of_most_similar_strings_to_return if request.number_of_most_similar_strings_to_return is not None else len(final_results)
        results = sorted(final_results, key=lambda x: x["similarity_to_query_text"][request.result_sorting_metric], reverse=True)[:num_to_return]
        response_time = datetime.utcnow()
        total_time = (response_time - request_time).total_seconds()
        logger.info(f"Finished advanced search in {total_time} seconds. Found {len(results)} results.")
        return {"query_text": request.query_text, "corpus_identifier_string": request.corpus_identifier_string, "embedding_pooling_method": request.embedding_pooling_method, "results": results}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/semantic-types/")
async def create_new_semantic_data_type(request: SemanticDataTypeEmbeddingRequest, req: Request = None, token: str = None, client_ip: str = None, document_file_hash: str = None) -> SemanticDataTypeEmbeddingResponse:
    
    try:
        request.semantic_data_type.name = prepare_string_for_embedding(request.semantic_data_type.name)
        request.semantic_data_type.description = prepare_string_for_embedding(request.semantic_data_type.description)
        request.semantic_data_type.examples = [prepare_string_for_embedding(example) for example in request.semantic_data_type.examples]
        unique_id = f"create_new_semantic_data_type_{request.semantic_data_type.name}_{request.llm_model_name}_{request.embedding_pooling_method}"
        
        input_data = {
            "semantic_data_type": {
                "name": request.semantic_data_type.name,
                "description": request.semantic_data_type.description,
                "examples": request.semantic_data_type.examples
            },
            "embed_examples_separately": request.embed_examples_separately,
            "llm_model_name": request.llm_model_name,
            "embedding_pooling_method": request.embedding_pooling_method,
            "corpus_identifier_string": request.corpus_identifier_string
        }
        
        
        # Generate embedding for description
        embedding_request = EmbeddingRequest(text=request.semantic_data_type.description, llm_model_name=request.llm_model_name, embedding_pooling_method=request.embedding_pooling_method)
        embedding_response = await get_or_compute_embedding(request=embedding_request, req=req, use_verbose=False)   
        description_embedding = embedding_response["text_embedding_dict"]["embedding_json"]
        embedding_vector = json.loads(description_embedding)
        description_embedding = np.array(embedding_vector).flatten()
        
        # Generate embeddings for examples
        examples_embeddings = []
        if request.embed_examples_separately:
            for example in request.semantic_data_type.examples:
                sample_embedding_request = EmbeddingRequest(text=example, llm_model_name=request.llm_model_name, embedding_pooling_method=request.embedding_pooling_method)
                sample_embedding_response = await get_or_compute_embedding(request=sample_embedding_request, req=req, use_verbose=False)
                sample_embedding = json.loads(sample_embedding_response["text_embedding_dict"]["embedding_json"])
                examples_embeddings.append(np.array(sample_embedding).flatten())
        else:
            combined_examples = " ".join(request.semantic_data_type.examples)
            combined_embedding_request = EmbeddingRequest(text=combined_examples, llm_model_name=request.llm_model_name, embedding_pooling_method=request.embedding_pooling_method)
            combined_embedding_response = await get_or_compute_embedding(request=combined_embedding_request, req=req, use_verbose=False)
            combined_embedding = json.loads(combined_embedding_response["text_embedding_dict"]["embedding_json"])
            examples_embeddings = [np.array(combined_embedding).flatten()]

        # Combine description and examples embeddings
        if examples_embeddings:
            combined_embedding = np.concatenate([description_embedding] + examples_embeddings)
        else:
            combined_embedding = description_embedding

        # Ensure combined_embedding is 2D before normalizing
        if combined_embedding.ndim == 1:
            combined_embedding = combined_embedding.reshape(1, -1)
        
        # Convert combined_embedding to float32 before normalization
        combined_embedding = combined_embedding.astype(np.float32)
        faiss.normalize_L2(combined_embedding)
        
        async with AsyncSessionLocal() as session:
            new_semantic_data_type = SemanticDataType(
                name=request.semantic_data_type.name,
                description=request.semantic_data_type.description,
                samples=",".join(request.semantic_data_type.examples),
                combined_embedding=combined_embedding.tobytes(),
                llm_model_name=request.llm_model_name,
                embedding_pooling_method=request.embedding_pooling_method,
                corpus_identifier_string=request.corpus_identifier_string
            )
            session.add(new_semantic_data_type)
            await session.flush()  # This will assign an ID to new_semantic_data_type
            await session.commit()
        # Create and return the response
        response = SemanticDataTypeEmbeddingResponse(
            semantic_data_type_id=new_semantic_data_type.id,
            combined_embedding=[combined_embedding.tolist()],
            examples_embeddings=examples_embeddings,
            status="success"
        )
        return response
    
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc()) # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.get("/semantic-types/",
        summary="Retrieve All Semantic Data Types",
        description="""Retrieve a list of all semantic data types from the database.""",
        response_description="A JSON object containing the list of all semantic data types.")
async def get_all_semantic_data_types(req: Request, token: str = None) -> AllSemanticDataTypesResponse:
    
    try:
        logger.info("Retrieving all semantic data types from the database")
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(SemanticDataType))
            all_semantic_data_types = result.scalars().all()
            logger.info(f"Retrieved {len(all_semantic_data_types):,} semantic data types from the database")
            semantic_data_types = [
                SemanticDataTypeResponse(
                    id=sdt.id,
                    name=sdt.name,
                    description=sdt.description,
                    llm_model_name=sdt.llm_model_name,
                    embedding_pooling_method=sdt.embedding_pooling_method,
                    created_at=sdt.created_at
                )
                for sdt in all_semantic_data_types
            ]
            logger.info(f"Processed {len(semantic_data_types):,} semantic data types")
            return AllSemanticDataTypesResponse(semantic_data_types=semantic_data_types)
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/documents",
    summary="Get Embeddings for a Document",
    description="""Extract text embeddings for a document. This endpoint supports plain text, .doc/.docx (MS Word), PDF files, images (using Tesseract OCR), and many other file types supported by the textract library.

### Parameters:
- `file`: The uploaded document file (either plain text, .doc/.docx, PDF, etc.).
- `url`: URL of the document file to download (optional; in lieu of `file`).
- `hash`: SHA3-256 hash of the document file to verify integrity (optional; in lieu of `file`).
- `size`: Size of the document file in bytes to verify completeness (optional; in lieu of `file`).
- `llm_model_name`: The model used to calculate embeddings (optional).
- `embedding_pooling_method`: The method used to pool the embeddings (Choices: 'mean', 'mins_maxes', 'svd', 'svd_first_four', 'ica', 'factor_analysis', 'gaussian_random_projection'; default is 'mean').
- `corpus_identifier_string`: An optional string identifier for grouping documents into a specific corpus.
- `json_format`: The format of the JSON response (optional, see details below).
- `send_back_json_or_zip_file`: Whether to return a JSON file or a ZIP file containing the embeddings file (optional, defaults to `zip`).
- `query_text`: An optional query text to perform a semantic search with the same parameters used for the document embedding request.
- `token`: Security token (optional).

### JSON Format Options:
The format of the JSON string returned by the endpoint (default is `records`; these are the options supported by the Pandas `to_json()` function):

- `split` : dict like {`index` -> [index], `columns` -> [columns], `data` -> [values]}
- `records` : list like [{column -> value}, … , {column -> value}]
- `index` : dict like {index -> {column -> value}}
- `columns` : dict like {column -> {index -> value}}
- `values` : just the values array
- `table` : dict like {`schema`: {schema}, `data`: {data}}

### Examples:
- Plain Text: Submit a file containing plain text.
- MS Word: Submit a `.doc` or `.docx` file.
- PDF: Submit a `.pdf` file.""",
    response_description="Either a ZIP file containing the embeddings JSON file or a direct JSON response, depending on the value of `send_back_json_or_zip_file`.")
async def upload_and_process_documents(
    file: UploadFile = File(None),
    url: str = Form(None),
    hash: str = Form(None),
    size: int = Form(None),
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD,
    corpus_identifier_string: str = "", 
    json_format: str = 'records',
    send_back_json_or_zip_file: str = 'zip',
    query_text: str = None,
    token: str = None,
    req: Request = None
):
    logger.info(f"Received request with embedding_pooling_method: {embedding_pooling_method}")
    
    client_ip = req.client.host if req else "localhost"
    request_time = datetime.utcnow()
    if file:
        input_data_binary = await file.read()
        result = magika.identify_bytes(input_data_binary)
        detected_data_type = result.output.ct_label
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{detected_data_type}", delete=False)
        temp_file_path = temp_file.name
        logger.info(f"Temp file path: {temp_file_path}")
        with open(temp_file_path, 'wb') as buffer:
            buffer.write(input_data_binary)
    elif url and hash and size:
        temp_file_path = await download_file(url, size, hash)
        with open(temp_file_path, 'rb') as f:
            input_data_binary = f.read()
            result = magika.identify_bytes(input_data_binary)
            detected_data_type = result.output.ct_label
            new_temp_file_path = temp_file_path + f".{detected_data_type}"
            os.rename(temp_file_path, new_temp_file_path)
            temp_file_path = new_temp_file_path
    else:
        raise HTTPException(status_code=400, detail="Invalid input. Provide either a file or URL with hash and size.")
    try:
        hash_obj = sha3_256()
        with open(temp_file_path, 'rb') as buffer:
            for chunk in iter(lambda: buffer.read(1024), b''):
                hash_obj.update(chunk)
        document_file_hash = hash_obj.hexdigest()
        logger.info(f"SHA3-256 hash of submitted file: {document_file_hash}")

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DocumentEmbedding).filter(DocumentEmbedding.document_file_hash == document_file_hash))
            existing_document = result.scalar_one_or_none()
            if existing_document:
                logger.info(f"Duplicate document detected: {document_file_hash}. Skipping processing.")
                return JSONResponse(content={"message": "Duplicate document detected. This file has already been uploaded."}, status_code=200)

        if corpus_identifier_string == "":
            corpus_identifier_string = document_file_hash
        unique_id = f"document_embedding_{document_file_hash}_{llm_model_name}_{embedding_pooling_method}"
        # max_retries = 5
        # for attempt in range(max_retries):
            
        #     wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
        #     logger.warning(f"Attempt {attempt + 1}: Failed to acquire lock: {e}. Retrying in {wait_time:,.2f} seconds.")
        #     await asyncio.sleep(wait_time)
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DocumentEmbedding).filter(DocumentEmbedding.document_file_hash == document_file_hash, DocumentEmbedding.llm_model_name == llm_model_name, DocumentEmbedding.embedding_pooling_method == embedding_pooling_method))
            existing_document_embedding = result.scalar_one_or_none()
            if existing_document_embedding:
                logger.info("Document has been processed before, returning existing result")
                sentences = existing_document_embedding.sentences
                document_embedding_results_json_compressed_binary = existing_document_embedding.document_embedding_results_json_compressed_binary
                document_embedding_results_json_decompressed_binary = decompress_data(document_embedding_results_json_compressed_binary)
                json_content = document_embedding_results_json_decompressed_binary.decode('utf-8')
                if len(json_content) == 0:
                    raise HTTPException(status_code=400, detail="Could not retrieve document embedding results.")
                existing_document = 1
                document_embedding_request = {}
            else:
                embeddings_computed = False
                document_embedding_request = {}
                existing_document = 0
                with open(temp_file_path, 'rb') as f:
                    input_data_binary = f.read()
                result = magika.identify_bytes(input_data_binary)
                mime_type = result.output.mime_type
                sentences, thousands_of_input_words = await parse_submitted_document_file_into_sentence_strings_func(temp_file_path, mime_type)
                document_embedding_request['mime_type'] = mime_type
                document_embedding_request['filename'] = file.filename if file else os.path.basename(url)
                document_embedding_request['sentences'] = sentences
                document_embedding_request['total_number_of_sentences'] = len(sentences)
                document_embedding_request['total_words'] = sum(len(sentence.split()) for sentence in sentences)
                document_embedding_request['total_characters'] = sum(len(sentence) for sentence in sentences)
                document_embedding_request['thousands_of_input_words'] = thousands_of_input_words
                document_embedding_request['file_size_mb'] = os.path.getsize(temp_file_path) / (1024 * 1024)
                document_embedding_request['corpus_identifier_string'] = corpus_identifier_string
                document_embedding_request['embedding_pooling_method'] = embedding_pooling_method
                document_embedding_request['llm_model_name'] = llm_model_name
                document_embedding_request['document_file_hash'] = document_file_hash
                if thousands_of_input_words > MAX_THOUSANDS_OF_WORDs_FOR_DOCUMENT_EMBEDDING:
                    raise HTTPException(status_code=400, detail=f"Document contains ~{int(thousands_of_input_words*1000):,} words, more than the maximum of {MAX_THOUSANDS_OF_WORDs_FOR_DOCUMENT_EMBEDDING*1000:,} words, which would take too long to compute embeddings for. Please submit a smaller document.") 
                first_10_words_of_input_text = ' '.join(' '.join(sentences).split()[:10])
                logger.info(f"Received request to extract embeddings for document with MIME type: {mime_type} and size: {os.path.getsize(temp_file_path):,} bytes from IP address: {client_ip}; First 10 words of the document: '{first_10_words_of_input_text}...'")
                logger.info(f"Document contains ~{int(thousands_of_input_words*1000):,} words, which is within the maximum of {MAX_THOUSANDS_OF_WORDs_FOR_DOCUMENT_EMBEDDING*1000:,} words. Proceeding with embedding computation using {llm_model_name} and pooling method {embedding_pooling_method}.") 
                input_data = {
                    "filename": file.filename if file else os.path.basename(url),
                    "sentences": sentences,
                    "file_size_mb": os.path.getsize(temp_file_path) / (1024 * 1024),
                    "mime_type": mime_type
                }
                
                try:
                    if not embeddings_computed:
                        json_content = await compute_embeddings_for_document(sentences=sentences, llm_model_name=llm_model_name, embedding_pooling_method=embedding_pooling_method, corpus_identifier_string=corpus_identifier_string, client_ip=client_ip, document_file_hash=document_file_hash, file=file, original_file_content=input_data_binary, json_format=json_format)
                        logger.info(f"Done getting all regular embeddings for document containing {len(sentences):,} sentences with model {llm_model_name} and embedding pooling method {embedding_pooling_method} and corpus {corpus_identifier_string}")
                        embeddings_computed = True
                        logger.info(f"Done getting all regular embeddings for document containing {len(sentences):,} sentences with model {llm_model_name} and embedding pooling method {embedding_pooling_method} and corpus {corpus_identifier_string}")
                    await store_document_embeddings_in_db(
                        file=file,
                        filename=document_embedding_request['filename'],
                            document_file_hash=document_file_hash,
                            original_file_content=input_data_binary,
                        sentences=sentences,
                        json_content=input_data_binary,
                        llm_model_name=llm_model_name,
                        embedding_pooling_method=embedding_pooling_method,
                        corpus_identifier_string=corpus_identifier_string,
                        client_ip=client_ip,
                        request_time=request_time
                    )
                    await session.commit()
                
                except Exception as e:
                    logger.error(f"Error while computing embeddings for document: {e}")
                    traceback.print_exc()
                    raise HTTPException(status_code=400, detail="Error while computing embeddings for document")
                
        if query_text:
            use_advanced_semantic_search = 0
            if use_advanced_semantic_search:
                search_request = AdvancedSemanticSearchRequest(
                    query_text=query_text,
                    llm_model_name=llm_model_name,
                    embedding_pooling_method=embedding_pooling_method,
                    corpus_identifier_string=corpus_identifier_string,
                    similarity_filter_percentage=0.01,
                    result_sorting_metric="hoeffding_d",
                    number_of_most_similar_strings_to_return=10
                )
                logger.info(f"Performing advanced semantic search for model {llm_model_name} and pooling method {embedding_pooling_method}...")
                search_response = await advanced_semantic_search(search_request, req, token)
                search_results = search_response["results"]
            else:
                search_request = SemanticSearchRequest(
                    query_text=query_text,
                    llm_model_name=llm_model_name,
                    embedding_pooling_method=embedding_pooling_method,
                    corpus_identifier_string=corpus_identifier_string,
                    number_of_most_similar_strings_to_return=10
                )
                logger.info(f"Performing semantic search for model {llm_model_name} and pooling method {embedding_pooling_method}...")
                search_response = await simple_semantic_search(search_request, req, token)
                search_results = search_response["results"]
            logger.info(f"Advanced semantic search completed. Results for query text '{query_text}'\n: {search_results}")
            json_content_dict = {"document_embedding_request": document_embedding_request, "document_embedding_results": json.loads(json_content), "semantic_search_request": dict(search_request), "semantic_search_results": search_results} 
            json_content = json.dumps(json_content_dict)
        else:
            json_content_dict = {"document_embedding_request": document_embedding_request, "document_embedding_results": json.loads(json_content)}
            json_content = json.dumps(json_content_dict)                                
        overall_total_time = (datetime.utcnow() - request_time).total_seconds()
        json_content_length = len(json_content)
        if json_content_length > 0:
            if not existing_document:
                logger.info(f"The response took {overall_total_time:,.2f} seconds to generate, or {float(overall_total_time / (thousands_of_input_words)):,.2f} seconds per thousand input tokens and {overall_total_time / (float(json_content_length) / 1000000.0):,.2f} seconds per million output characters.") 
            if send_back_json_or_zip_file == 'json':
                logger.info(f"Returning JSON response for document containing {len(sentences):,} sentences with model {llm_model_name}; first 100 characters out of {json_content_length:,} total of JSON response: {json_content[:100]}" if 'sentences' in locals() else f"Returning JSON response; first 100 characters out of {json_content_length:,} total of JSON response: {json_content[:100]}")
                return JSONResponse(content=json.loads(json_content))
            else:
                original_filename_without_extension, _ = os.path.splitext(file.filename if file else os.path.basename(url))
                json_file_path = f"/tmp/{original_filename_without_extension}.json"
                with open(json_file_path, 'w') as json_file:
                    json_file.write(json_content)
                zip_file_path = f"/tmp/{original_filename_without_extension}.zip"
                with zipfile.ZipFile(zip_file_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(json_file_path, os.path.basename(json_file_path))
                logger.info(f"Returning ZIP response for document containing {len(sentences):,} sentences with model {llm_model_name}; first 100 characters out of {json_content_length:,} total of JSON response: {json_content[:100]}")
                return FileResponse(zip_file_path, headers={"Content-Disposition": f"attachment; filename={original_filename_without_extension}.zip"})
    
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

@app.get("/documents", response_model=AllDocumentsResponse)
async def get_all_documents(req: Request, token: str = None) -> AllDocumentsResponse:
    logger.info("Received request to retrieve all stored documents")
    
    try:
        logger.info("Retrieving all stored documents from the database")
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).options(joinedload(Document.document_embeddings))
            )
            documents = result.unique().scalars().all()
        logger.info(f"Retrieved {len(documents):,} stored documents from the database")
        return AllDocumentsResponse(documents=[DocumentPydantic.from_orm(doc) for doc in documents])
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/document-chunks/{document_hash}", response_model=DocumentContentResponse)
async def get_document_chunks(document_hash: str):
    try:
        logger.info(f"Retrieving content for document with document_hash: {document_hash}")
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).options(joinedload(Document.document_embeddings)).filter(Document.document_hash == document_hash)
            )
            document = result.scalars().first()
            if document:
                logger.info(f"Retrieved content for document with document_hash: {document_hash}")
                sentences = []
                for doc_embedding in document.document_embeddings:
                    if doc_embedding.sentences:
                        sentences.extend(json.loads(doc_embedding.sentences))
                content = ' '.join(sentences)
                return DocumentContentResponse(content=content, sentences=sentences)
            else:
                logger.warning(f"No document found with document_hash: {document_hash}")
                raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/documents/delete", response_model=Dict[str, Any])
async def delete_documents(
    document_ids: List[int],
    token: str = None,
    req: Request = None
) -> Dict[str, Any]:
    

    client_ip = req.client.host if req else "localhost"
    logger.info(f"Received request to delete documents with IDs: {document_ids} from {client_ip}")

    try:
        if not document_ids:
            raise ValueError("No document IDs provided")

        async with AsyncSessionLocal() as session:
            # First, get the document_file_hashes for the given document IDs
            document_file_hashes_query = select(DocumentEmbedding.document_file_hash).where(DocumentEmbedding.document_hash.in_(
                select(Document.document_hash).where(Document.id.in_(document_ids))
            ))
            document_file_hashes_result = await session.execute(document_file_hashes_query)
            document_file_hashes = [row[0] for row in document_file_hashes_result]

            # Delete TextEmbeddings
            text_embeddings_delete = delete(TextEmbedding).where(TextEmbedding.document_file_hash.in_(document_file_hashes))
            text_embeddings_result = await session.execute(text_embeddings_delete, execution_options={"synchronize_session": "fetch", "is_delete_using": True})
            text_embeddings_deleted = text_embeddings_result.rowcount

            # Delete DocumentEmbeddings
            doc_embeddings_delete = delete(DocumentEmbedding).where(
                and_(
                    DocumentEmbedding.document_file_hash.in_(document_file_hashes),
                    DocumentEmbedding.document_hash.in_(select(Document.document_hash).where(Document.id.in_(document_ids)))
                )
            )
            doc_embeddings_result = await session.execute(doc_embeddings_delete, execution_options={"synchronize_session": "fetch", "is_delete_using": True})
            doc_embeddings_deleted = doc_embeddings_result.rowcount

            # Delete Documents
            documents_delete = delete(Document).where(Document.id.in_(document_ids))
            documents_result = await session.execute(documents_delete, execution_options={"synchronize_session": "fetch", "is_delete_using": True})
            documents_deleted = documents_result.rowcount

            await session.commit()

            return {
                "status": "success",
                "message": f"Successfully deleted {documents_deleted} document(s) and their associated embeddings",
                "deleted_count": {
                    "documents": documents_deleted,
                    "document_embeddings": doc_embeddings_deleted,
                    "text_embeddings": text_embeddings_deleted
                }
            }
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except SQLAlchemyError as e:
        logger.error(f"Database error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error occurred: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred while deleting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting documents: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
