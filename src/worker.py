import os
import sys
import redis
from rq import Worker, Queue, Connection, get_current_job
import logging
import multiprocessing as mp
import traceback
import platform
from hashlib import sha3_256, md5
import requests
from magika import Magika
from utils import add_model_url, download_models
from functions import (
    RedisManager,
    parse_submitted_document_file_into_sentence_strings_func
)
from functions import build_faiss_indexes, get_or_compute_embedding, get_list_of_corpus_identifiers_from_list_of_embedding_texts
from models import AdvancedSemanticSearchRequest, EmbeddingRequest, AdvancedSemanticSearchResponse
import faiss
import numpy as np
import asyncio
from sqlalchemy import select
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, DateTime
from datetime import datetime
import json
from models import Document, DocumentEmbedding
from db import AsyncSessionLocal
import urllib
import fast_vector_similarity as fvs
from typing import Dict, Any
import gc
import resource
import psutil
from functions import load_model
from db import DatabaseWriter, initialize_db, get_db_writer
# Configure logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('worker.log')
    ]
)
logger = logging.getLogger(__name__)

# At the top of the file, create a Magika instance
magika_instance = Magika()

# Add specific import for urllib
from urllib import request as urllib_request

redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = int(os.environ.get('REDIS_PORT', 6379))

redis_client = redis.Redis(host=redis_host, port=redis_port)
global redis_manager
redis_manager = RedisManager()

def setup_process():
    """Set up process-specific configurations"""
    if platform.system() == 'Darwin':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        if 'numpy' in sys.modules:
            del sys.modules['numpy']

def get_memory_limit():
    """Get memory limit in bytes"""
    # Get system memory info
    mem = psutil.virtual_memory()
    # Use 80% of available memory as the limit
    return int(mem.available * 0.8)

def limit_memory():
    """Set memory limits for the worker process"""
    try:
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        
        # Calculate new limit
        new_limit = get_memory_limit()
        
        # Ensure new limit doesn't exceed hard limit
        new_limit = min(new_limit, hard)
        
        # Set new memory limit
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
        logger.info(f"Set memory limit to {new_limit / (1024*1024):.2f} MB")
        
    except Exception as e:
        logger.warning(f"Failed to set memory limit: {str(e)}. Continuing without limit.")
        # Continue execution even if setting limit fails
        pass

def worker_process(queue_names):
    """Function to run in a separate process for handling worker tasks"""
    try:
        limit_memory()
        logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        redis_manager = RedisManager()
        asyncio.run(redis_manager.initialize())
        
        redis_conn = redis_manager.redis_sync
        with Connection(redis_conn):
            queues = [redis_manager.get_queue(name) for name in queue_names]
            worker = Worker(queues, name=f'worker-{os.getpid()}')
            worker.work(with_scheduler=True)
    except Exception as e:
        logger.error(f"Redis connection error: {str(e)}")
        raise

def download_model_task(model_url: str) -> dict:
    """Worker function to handle model download in the background."""
    job = get_current_job()
    job.meta['progress'] = 0
    job.save_meta()

    logger.info(f"Starting download task for model URL: {model_url}")
    try:
        # Clean up the URL
        decoded_url = model_url.strip('"')
        if not decoded_url.endswith('.gguf'):
            return {
                "status": "error",
                "message": "Model URL must point to a .gguf file."
            }
            
        # Add model URL to registry
        job.meta['progress'] = 10
        job.save_meta()
        corrected_url = add_model_url(decoded_url)
        
        # Get the model filename from the URL
        model_filename = os.path.basename(corrected_url)
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_filename)

        # Check if model already exists
        if os.path.exists(model_path):
            return {
                "status": "completed",
                "message": f"Model {model_filename} already exists"
            }

        # Download the model
        logger.info(f"Downloading model from {corrected_url}")
        job.meta['progress'] = 20
        job.save_meta()

        try:
            urllib_request.urlretrieve(
                corrected_url, 
                model_path,
                lambda count, block_size, total_size: job.meta.update({
                    'progress': int(20 + (count * block_size / total_size * 70))
                }) if total_size > 0 else None
            )
            
            # Verify download
            if os.path.exists(model_path):
                actual_size = os.path.getsize(model_path)
                if actual_size < 100 * 1024 * 1024:  # Less than 100MB
                    os.remove(model_path)
                    raise Exception("Downloaded file is too small to be a valid model")
                
                job.meta['progress'] = 100
                job.save_meta()
                
                return {
                    "status": "completed",
                    "message": f"Successfully downloaded model {model_filename}",
                    "file_path": model_path,
                    "file_size": actual_size
                }
            else:
                raise Exception("Failed to save model file")

        except Exception as download_error:
            if os.path.exists(model_path):
                os.remove(model_path)
            raise download_error

    except Exception as e:
        error_msg = f"Error in download task: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": error_msg
        }
    finally:
        job.meta['progress'] = 100
        job.save_meta()

async def upload_file_task(file_path_or_url: str, hash: str, size: int, llm_model_name: str, embedding_pooling_method: str, corpus_identifier_string: str, json_format: str, send_back_json_or_zip_file: str, query_text: str) -> dict:
    """Worker function to handle file upload in the background."""
    job = get_current_job()
    job.meta['progress'] = 0
    job.save_meta()

    logger.info(f"Starting file upload task for file: {file_path_or_url}")
    try:
        # Initialize database connection
        redis_manager = RedisManager()
        await redis_manager.initialize()
        
        # Get file content and metadata
        if file_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(file_path_or_url)
            response.raise_for_status()
            file_content = response.content
            file_name = file_path_or_url.split('/')[-1]
        else:
            with open(file_path_or_url, 'rb') as f:
                file_content = f.read()
            file_name = os.path.basename(file_path_or_url)
        
        job.meta['progress'] = 25
        job.save_meta()
        
        # Process file content
        result = magika_instance.identify_bytes(file_content)
        mime_type = result.output.mime_type
        
        # Compute file hash if not provided
        if not hash:
            hash_obj = sha3_256()
            hash_obj.update(file_content)
            hash = hash_obj.hexdigest()
        
        # Process the document
        sentences, thousands_of_input_words = await parse_submitted_document_file_into_sentence_strings_func(
            file_path_or_url, 
            mime_type
        )
        
        job.meta['progress'] = 50
        job.save_meta()

        # Store in database using a single session
        async with AsyncSessionLocal() as session:
            try:
                # Create and add Document
                document = Document(
                    document_hash=hash,
                    filename=file_name,  # Changed from file_name to filename to match the model
                    llm_model_name=llm_model_name,
                    corpus_identifier_string=corpus_identifier_string
                )
                session.add(document)
                await session.flush()  # Assign ID to document

                # Create and add DocumentEmbedding
                doc_embedding = DocumentEmbedding(
                    document_hash=hash,
                    document_file_hash=hash,
                    filename=file_name,  # Changed from file_name to filename
                    mimetype=mime_type,
                    llm_model_name=llm_model_name,
                    embedding_pooling_method=embedding_pooling_method,
                    sentences=json.dumps(sentences),
                    corpus_identifier_string=corpus_identifier_string,
                    file_data=file_content
                )
                session.add(doc_embedding)
                await session.flush()  # Assign ID to doc_embedding

                # Commit both records in a single transaction
                await session.commit()
                logger.info(f"Successfully saved document and embedding with hash: {hash}")

            except Exception as db_error:
                await session.rollback()
                logger.error(f"Database error: {str(db_error)}")
                raise

        job.meta['progress'] = 100
        job.save_meta()
        
        return {
            "status": "completed",
            "message": "File upload and processing completed successfully",
            "document_hash": hash,
            "file_info": {
                "file_name": file_name,
                "file_size": len(file_content),
                "mime_type": mime_type,
                "sentence_count": len(sentences),
                "word_count": int(thousands_of_input_words * 1000)
            }
        }
        
    except Exception as e:
        error_msg = f"Error in upload task: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        job.meta['progress'] = 100
        job.save_meta()
        return {
            "status": "error",
            "message": error_msg
        }
    finally:
        # Clean up temporary file
        if os.path.exists(file_path_or_url) and file_path_or_url.startswith('/tmp/'):
            os.remove(file_path_or_url)

async def scan_document_task(query_text: str, llm_model_name: str, embedding_pooling_method: str, 
                           corpus_identifier_string: str, similarity_filter_percentage: float,
                           number_of_most_similar_strings_to_return: int, result_sorting_metric: str) -> Dict[str, Any]:
    job = get_current_job()
    job.meta['progress'] = 0
    job.save_meta()
    model = None

    try:
        logger.info(f"Starting scan task with parameters: query_text_length={len(query_text)}, model={llm_model_name}")
        
        # Use the global redis_manager instead of creating a new one
        global redis_manager
        if not redis_manager:
            redis_manager = RedisManager()
            await redis_manager.initialize()
        
        # Get the db_writer from redis_manager
        db_writer = redis_manager.db_writer
        
        logger.info("Loading model...")
        try:
            model = load_model(llm_model_name)
            if not model:
                raise ValueError(f"Failed to load model: {llm_model_name}")
            logger.info("Model loaded successfully")
        except Exception as model_error:
            logger.error(f"Error loading model: {str(model_error)}")
            raise

        # Get embedding for query text
        logger.info("Creating embedding request...")
        embedding_request = EmbeddingRequest(
            text=query_text,
            llm_model_name=llm_model_name,
            embedding_pooling_method=embedding_pooling_method
        )
        
        logger.info("Computing embedding...")
        try:
            embedding_response = await get_or_compute_embedding(embedding_request, db_writer)
            input_embedding = np.array(json.loads(embedding_response["text_embedding_dict"]["embedding_json"])).astype('float32').reshape(1, -1)
            logger.info("Embedding computed successfully")
        except Exception as embed_error:
            logger.error(f"Error computing embedding: {str(embed_error)}")
            raise

        job.meta['progress'] = 50
        job.save_meta()

        # Build FAISS index
        logger.info("Building FAISS index...")
        try:
            faiss_index, associated_texts_by_model_and_pooling_method = await build_faiss_indexes(
                llm_model_name=llm_model_name,
                embedding_pooling_method=embedding_pooling_method
            )
            
            if not faiss_index:
                raise ValueError("No FAISS index could be built - no embeddings found")
            logger.info("FAISS index built successfully")
        except Exception as faiss_error:
            logger.error(f"Error building FAISS index: {str(faiss_error)}")
            raise

        # Perform search
        num_results = max([1, int((1 - similarity_filter_percentage) * len(associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]))])
        similarities, indices = faiss_index.search(input_embedding, num_results)
        
        # Process results
        filtered_indices = indices[0]
        filtered_similarities = similarities[0]
        final_results = []
        associated_texts = associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]
        
        list_of_corpus_identifier_strings = await get_list_of_corpus_identifiers_from_list_of_embedding_texts(
            associated_texts, llm_model_name, embedding_pooling_method
        )

        for idx, similarity in zip(filtered_indices, filtered_similarities):
            if idx < len(associated_texts) and list_of_corpus_identifier_strings[idx] == corpus_identifier_string:
                associated_text = associated_texts[idx]
                
                # Get detailed similarity stats
                comparison_embedding_request = EmbeddingRequest(
                    text=associated_text,
                    llm_model_name=llm_model_name,
                    embedding_pooling_method=embedding_pooling_method
                )
                comparison_embedding_response = await get_or_compute_embedding(
                    comparison_embedding_request,
                    db_writer
                )
                comparison_embedding = np.array(json.loads(
                    comparison_embedding_response["text_embedding_dict"]["embedding_json"]
                )).astype('float32').reshape(1, -1)
                
                params = {
                    "vector_1": input_embedding.tolist()[0],
                    "vector_2": comparison_embedding.tolist()[0],
                    "similarity_measure": "all"
                }
                similarity_stats = json.loads(fvs.py_compute_vector_similarity_stats(json.dumps(params)))
                
                final_results.append({
                    "search_result_text": associated_text,
                    "similarity_to_query_text": similarity_stats
                })

        # Sort and trim results
        results = sorted(
            final_results,
            key=lambda x: x["similarity_to_query_text"][result_sorting_metric],
            reverse=True
        )[:number_of_most_similar_strings_to_return]

        job.meta['progress'] = 100
        job.save_meta()

        return {
            "status": "completed",
            "results": {
                "query_text": query_text,
                "model": llm_model_name,
                "embedding_pooling_method": embedding_pooling_method,
                "corpus_identifier_string": corpus_identifier_string,
                "similarity_filter_percentage": similarity_filter_percentage,
                "number_of_most_similar_strings_to_return": number_of_most_similar_strings_to_return,
                "result_sorting_metric": result_sorting_metric,
                "results": results
            }
        }

    except Exception as e:
        error_msg = f"Error in scan_document_task: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": error_msg
        }
    finally:
        # Clean up resources
        if model:
            try:
                del model
                gc.collect()
                logger.info("Model cleanup completed")
            except Exception as cleanup_error:
                logger.error(f"Error during model cleanup: {str(cleanup_error)}")

class MultiQueueWorker:
    def __init__(self):
        setup_process()
        
        # Update Redis connection to use redis_url
        self.redis_conn = redis_client
        
        # Initialize RQ queues
        self.queue_names = ['model_downloads', 'file_uploads', 'document_scans']
        self.queues = {
            name: Queue(name, connection=self.redis_conn, is_async=True)
            for name in self.queue_names
        }

    def start_worker(self):
        """Start the RQ worker in a separate process"""
        try:
            logger.info('Initializing worker...')
            logger.info(f'Worker listening to queues: {self.queue_names}')
            
            # Start the worker in a separate process
            process = mp.Process(target=worker_process, args=(self.queue_names,))
            process.start()
            logger.info(f'Worker started successfully with PID: {process.pid}')
            process.join()
                
        except Exception as e:
            logger.error(f"Failed to start worker: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)

def main():
    """Main entry point for the worker"""
    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')
    
    try:
        worker = MultiQueueWorker()
        worker.start_worker()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
