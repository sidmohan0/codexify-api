
from db import AsyncSessionLocal, create_async_engine, create_tables
from utils import  build_faiss_indexes, configure_redis_optimally
from models import DocumentEmbedding, Document, TextEmbedding, DocumentContentResponse,  DocumentPydantic, SemanticDataTypeResponse, AllSemanticDataTypesResponse
from models import EmbeddingRequest, SemanticSearchRequest, AdvancedSemanticSearchRequest, SimilarityRequest, SimilarityResponse
from models import EmbeddingResponse, SemanticSearchResponse, AdvancedSemanticSearchResponse, AllDocumentsResponse
from models import SemanticDataType, SemanticDataTypeEmbeddingRequest, SemanticDataTypeEmbeddingResponse, SemanticSearchRequest, SemanticSearchResponse, AllSemanticDataTypesResponse
from functions import get_or_compute_embedding, download_file, decompress_data, store_document_embeddings_in_db, initialize_globals, RedisManager
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
from utils import download_models, add_model_url
from fastapi import FastAPI, BackgroundTasks
from typing import Dict, Any
from rq.job import Job
from worker import MultiQueueWorker
import hashlib
from functions import redis_manager
from rq import Queue

import os
import redis

redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = int(os.environ.get('REDIS_PORT', 6379))

redis_client = redis.Redis(host=redis_host, port=redis_port)



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


logger = logging.getLogger(__name__)

# Initialize Redis manager for the API
multi_queue_worker = MultiQueueWorker()

@app.on_event("startup")
async def pre_startup_tasks():
    """Initialize necessary components on startup"""
    logger.info("Starting pre-startup tasks")
    try:
        logger.info("Initializing globals")
        await initialize_globals()  # This sets up redis_manager and other components
        
        logger.info("Building FAISS indexes")
        await build_faiss_indexes(force_rebuild=True)
        
        logger.info("Downloading initial models")
        await download_models()
        
        logger.info("Pre-startup tasks completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.get("/", include_in_schema=False)
async def load_swagger_ui():
    return fastapi.templating.get_swagger_ui_html(openapi_url="/openapi.json", title=app.title, swagger_favicon_url=app.swagger_ui_favicon_url)

@app.get("/models")
async def get_list_of_models(token: str = None) -> Dict[str, List[str]]:
    
    models_dir = os.path.join(BASE_DIRECTORY, 'models')
    model_files = glob.glob(os.path.join(models_dir, "*.bin")) + glob.glob(os.path.join(models_dir, "*.gguf"))
    model_names = sorted([os.path.splitext(os.path.basename(model_file))[0] for model_file in model_files])
    return {"model_names": model_names}

# @app.post("/models", response_model=Dict[str, Any])
# async def add_new_model(model_url: str, token: str = None) -> Dict[str, Any]:
#     """
#     Add a new model to the system.

#     1. The model must be in `.gguf` format.
#     2. The model must be larger than 100 MB to ensure it's a valid model file.

#     Parameters:
#     - `model_url`: The URL of the model weight file, which must end with `.gguf`.
#     - `token`: Security token (optional).

#     Returns:
#     A JSON object indicating the status of the model addition and download.
#     """
    

#     unique_id = f"add_model_{hash(model_url)}"

#     try:
#         decoded_model_url = unquote(model_url)
#         if not decoded_model_url.endswith('.gguf'):
#             return {"status": "error", "message": "Model URL must point to a .gguf file."}
        
#         corrected_model_url = add_model_url(decoded_model_url)
#         _, download_status = download_models()
#         status_dict = {status["url"]: status for status in download_status}
        
#         if corrected_model_url in status_dict:
#             return {
#                 "status": status_dict[corrected_model_url]["status"],
#                 "message": status_dict[corrected_model_url]["message"]
#             }
        
#         return {"status": "unknown", "message": "Unexpected error."}
#     except Exception as e:
#         logger.error(f"An error occurred while adding the model: {str(e)}")
#         return {"status": "unknown", "message": f"An unexpected error occurred: {str(e)}"}

@app.post("/models", response_model=Dict[str, Any])
async def add_new_model(model_url: str, token: str = None) -> Dict[str, Any]:
    """
    Add a new model to the system using RQ for background processing.
    """
    try:
        # Decode the URL if it's URL-encoded
        decoded_url = unquote(model_url).strip('"')
        logger.info(f"Processing model URL: {decoded_url}")

        # Create a unique job ID
        timestamp = datetime.now().isoformat()
        unique_id = hashlib.md5(f"{decoded_url}_{timestamp}".encode()).hexdigest()
        
        # Check existing jobs using the global redis_manager
        try:
            model_downloads_queue = redis_manager.get_queue('model_downloads')
            existing_jobs = model_downloads_queue.get_job_ids()
            for job_id in existing_jobs:
                try:
                    job = Job.fetch(job_id, connection=redis_manager.redis_sync)
                    if job and job.args and job.args[0] == decoded_url and job.get_status() != 'failed':
                        return {
                            "status": "already_queued",
                            "message": f"Model download already in progress. Job ID: {job_id}",
                            "job_id": job_id
                        }
                except Exception as fetch_err:
                    logger.warning(f"Error checking existing job {job_id}: {str(fetch_err)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error checking existing jobs: {str(e)}")

        # Enqueue the new task using the global redis_manager
        try:
            job = model_downloads_queue.enqueue(
                'worker.download_model_task',
                args=(decoded_url,),
                job_id=unique_id,
                result_ttl=86400,
                failure_ttl=86400,
                timeout='1h'
            )
            
            if not job:
                raise Exception("Job creation failed")
                
            logger.info(f"Job enqueued successfully. Job ID: {unique_id}")
            
            # Verify the job was enqueued
            verification_attempts = 3
            for attempt in range(verification_attempts):
                try:
                    enqueued_job = Job.fetch(unique_id, connection=redis_manager.redis_sync)
                    if enqueued_job:
                        return {
                            "status": "queued",
                            "message": f"Model download queued successfully. Job ID: {unique_id}",
                            "job_id": unique_id
                        }
                except Exception as fetch_err:
                    if attempt == verification_attempts - 1:
                        raise
                    logger.warning(f"Verification attempt {attempt + 1} failed: {str(fetch_err)}")
                    await asyncio.sleep(0.5)  # Wait briefly before retrying
                    
            raise Exception("Job verification failed after multiple attempts")
            
        except Exception as e:
            logger.error(f"Failed to enqueue job: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue model download: {str(e)}"
            )

    except Exception as e:
        error_msg = f"Failed to process model download request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)
    
    
@app.get("/models/status/{job_id}", response_model=Dict[str, Any])
async def get_model_status(job_id: str) -> Dict[str, Any]:
    """
    Check the status of a model download job
    """
    try:
        job = Job.fetch(job_id, connection=redis_manager.redis_sync)
        
        status_mapping = {
            'queued': {'status': 'pending', 'message': 'Download queued'},
            'started': {'status': 'pending', 'message': 'Download in progress'},
            'finished': {'status': 'completed', 'result': job.result},
            'failed': {'status': 'failed', 'error': str(job.exc_info)},
            'stopped': {'status': 'stopped', 'message': 'Download stopped'},
            'deferred': {'status': 'pending', 'message': 'Download deferred'}
        }
        
        job_status = job.get_status()
        return status_mapping.get(job_status, {
            'status': 'unknown',
            'message': f'Unknown job status: {job_status}'
        })
            
    except Exception as e:
        error_msg = f"Error fetching job status: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": error_msg
        }

@app.get("/models/jobs", response_model=Dict[str, Any])
async def list_model_jobs() -> Dict[str, Any]:
    """
    List all model download jobs and their statuses
    """
    try:
        jobs = []
        job_ids = Queue('model_downloads', connection=redis_manager.redis_sync).get_job_ids()
        
        for job_id in job_ids:
            job = Job.fetch(job_id, connection=multi_queue_worker.redis_conn)
            jobs.append({
                'job_id': job_id,
                'status': job.get_status(),
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'model_url': job.args[0] if job.args else None
            })
            
        return {
            "status": "success",
            "jobs": jobs
        }
        
    except Exception as e:
        error_msg = f"Error listing jobs: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)
      
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

@app.post("/documents", response_model=Dict[str, Any])
async def upload_and_process_documents(
    file: UploadFile = File(None),
    url: str = Form(None),
    hash: str = Form(None),
    size: int = Form(None),
    llm_model_name: str = Form(DEFAULT_EMBEDDING_MODEL_NAME),
    embedding_pooling_method: str = Form(DEFAULT_EMBEDDING_POOLING_METHOD),
    corpus_identifier_string: str = Form(""), 
    json_format: str = Form('records'),
    send_back_json_or_zip_file: str = Form('zip'),
    query_text: str = Form(None),
    token: str = Form(None),
    req: Request = None
) -> Dict[str, Any]:
    """
    Upload and process documents for embedding extraction.
    """
    try:
        # Decode the URL if it's URL-encoded
        decoded_url = unquote(url).strip('"') if url else None
        logger.info(f"Processing document URL: {decoded_url}")

        # Create a unique job ID
        timestamp = datetime.now().isoformat()
        unique_id = hashlib.md5(f"{decoded_url or (file.filename if file else 'unknown')}_{timestamp}".encode()).hexdigest()
        
        # Check existing jobs using the global redis_manager
        try:
            document_processing_queue = redis_manager.get_queue('file_uploads')
            existing_jobs = document_processing_queue.get_job_ids()
            for job_id in existing_jobs:
                try:
                    job = Job.fetch(job_id, connection=redis_manager.redis_sync)
                    if job and job.args and job.args[0] == (decoded_url or (file.filename if file else 'unknown')) and job.get_status() != 'failed':
                        return {
                            "status": "already_queued",
                            "message": f"Document processing already in progress. Job ID: {job_id}",
                            "job_id": job_id
                        }
                except Exception as fetch_err:
                    logger.warning(f"Error checking existing job {job_id}: {str(fetch_err)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error checking existing jobs: {str(e)}")

        # If a file was uploaded, save it to a temporary location
        temp_file_path = None
        if file:
            temp_file_path = f"/tmp/{unique_id}_{file.filename}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(await file.read())

        # Enqueue the new task using the global redis_manager
        try:
            job = document_processing_queue.enqueue(
                'worker.upload_file_task',
                args=(temp_file_path or decoded_url, hash, size, llm_model_name, embedding_pooling_method, corpus_identifier_string, json_format, send_back_json_or_zip_file, query_text),
                job_id=unique_id,
                result_ttl=86400,
                failure_ttl=86400,
                timeout='1h'
            )
            
            if not job:
                raise Exception("Job creation failed")
                
            logger.info(f"Job enqueued successfully. Job ID: {unique_id}")
            
            # Verify the job was enqueued
            verification_attempts = 3
            for attempt in range(verification_attempts):
                try:
                    enqueued_job = Job.fetch(unique_id, connection=redis_manager.redis_sync)
                    if enqueued_job:
                        return {
                            "status": "queued",
                            "message": f"Document processing queued successfully. Job ID: {unique_id}",
                            "job_id": unique_id
                        }
                except Exception as fetch_err:
                    if attempt == verification_attempts - 1:
                        raise
                    logger.warning(f"Verification attempt {attempt + 1} failed: {str(fetch_err)}")
                    await asyncio.sleep(0.5)  # Wait briefly before retrying
                    
            raise Exception("Job verification failed after multiple attempts")
            
        except Exception as e:
            logger.error(f"Failed to enqueue job: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue document processing: {str(e)}"
            )

    except Exception as e:
        error_msg = f"Failed to process document upload request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/documents/status/{job_id}", response_model=Dict[str, Any])
async def get_document_status(job_id: str) -> Dict[str, Any]:
    """
    Check the status of a document processing job
    """
    try:
        job = Job.fetch(job_id, connection=redis_manager.redis_sync)
        
        status_mapping = {
            'queued': {'status': 'pending', 'message': 'Processing queued'},
            'started': {'status': 'pending', 'message': 'Processing in progress'},
            'finished': {'status': 'completed', 'result': job.result},
            'failed': {'status': 'failed', 'error': str(job.exc_info)},
            'stopped': {'status': 'stopped', 'message': 'Processing stopped'},
            'deferred': {'status': 'pending', 'message': 'Processing deferred'}
        }
        
        job_status = job.get_status()
        return status_mapping.get(job_status, {
            'status': 'unknown',
            'message': f'Unknown job status: {job_status}'
        })
            
    except Exception as e:
        error_msg = f"Error fetching job status: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": error_msg
        }

@app.get("/documents/jobs", response_model=Dict[str, Any])
async def list_document_jobs() -> Dict[str, Any]:
    """
    List all document processing jobs and their statuses
    """
    try:
        jobs = []
        job_ids = Queue('file_uploads', connection=redis_manager.redis_sync).get_job_ids()
        
        for job_id in job_ids:
            job = Job.fetch(job_id, connection=redis_manager.redis_sync)
            jobs.append({
                'job_id': job_id,
                'status': job.get_status(),
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'document_info': job.args[0] if job.args else None
            })
            
        return {
            "status": "success",
            "jobs": jobs
        }
        
    except Exception as e:
        error_msg = f"Error listing jobs: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

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
    uvicorn.run(app, host="0.0.0.0", port=8089)
