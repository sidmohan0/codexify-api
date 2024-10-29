import os
import sys
import redis
from rq import Worker, Queue, Connection, get_current_job
import logging
import multiprocessing as mp
from functions import parse_submitted_document_file_into_sentence_strings_func, get_or_compute_embedding, get_list_of_corpus_identifiers_from_list_of_embedding_texts, prepare_string_for_embedding
import traceback
import platform
import magika
from hashlib import sha3_256
import requests
from magika import Magika
from utils import add_model_url, download_models
from os import getenv
from functions import RedisManager
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, DateTime
from datetime import datetime
import json
from models import Document, DocumentEmbedding
from db import AsyncSessionLocal, DatabaseWriter
import urllib
from fastapi import HTTPException
from models import EmbeddingRequest
import numpy as np
import fast_vector_similarity as fvs
from utils import build_faiss_indexes
import faiss
from typing import Optional

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

def setup_process():
    """Set up process-specific configurations"""
    if platform.system() == 'Darwin':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        if 'numpy' in sys.modules:
            del sys.modules['numpy']

def worker_process(queue_names):
    """Function to run in a separate process for handling worker tasks"""
    try:
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

async def scan_document_task(
    document_hash: Optional[str], 
    llm_model_name: str,
    embedding_pooling_method: str,
    corpus_identifier_string: str,
    json_format: str,
    send_back_json_or_zip_file: str,
    query_text: str,
    similarity_filter_percentage: float = 0.01,
    number_of_most_similar_strings_to_return: int = 10,
    result_sorting_metric: str = "hoeffding_d"
) -> dict:
    job = get_current_job()
    job.meta['progress'] = 0
    job.save_meta()

    logger.info(f"Starting document scan task for document hash: {document_hash}")
    try:
        # Initialize Redis manager
        redis_manager = RedisManager()
        await redis_manager.initialize()

        client_ip = "localhost"
        
        # Initialize database writer and start its processing loop
        queue = asyncio.Queue()
        db_writer = DatabaseWriter(queue)
        await db_writer.initialize_processing_hashes()
        
        # Start the database writer task and wait for it to be ready
        db_writer_task = asyncio.create_task(db_writer.dedicated_db_writer())
        await asyncio.sleep(0.1)  # Give the writer task time to start
        
        # Create embedding request
        embedding_request = EmbeddingRequest(
            text=query_text,
            llm_model_name=llm_model_name,
            embedding_pooling_method=embedding_pooling_method
        )
        
        try:
            # Get embedding without passing db_writer since it's not needed for this operation
            embedding_response = await get_or_compute_embedding(
                request=embedding_request,
                use_verbose=False,
                client_ip=client_ip
            )

            global faiss_indexes, associated_texts_by_model_and_pooling_method
            request_time = datetime.utcnow()
            query_text = prepare_string_for_embedding(query_text)   
            unique_id = f"advanced_semantic_search_{query_text}_{llm_model_name}_{embedding_pooling_method}_{similarity_filter_percentage}_{number_of_most_similar_strings_to_return}"
                
            faiss_indexes, associated_texts_by_model_and_pooling_method = await build_faiss_indexes(force_rebuild=True)
            try:
                faiss_index = faiss_indexes[(llm_model_name, embedding_pooling_method)]
            except KeyError:
                raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {llm_model_name} and pooling method: {embedding_pooling_method}")            

            num_results_before_corpus_filter = number_of_most_similar_strings_to_return * 25
            logger.info(f"Received request to find most similar strings for query text: `{query_text}` using model: {llm_model_name}")
            try:
                logger.info(f"Computing embedding for input text: {query_text}")
                embedding_request = EmbeddingRequest(text=query_text, llm_model_name=llm_model_name, embedding_pooling_method=embedding_pooling_method)
                embedding_response = await get_or_compute_embedding(embedding_request, db_writer=db_writer)  # Pass db_writer here
                embedding_json = embedding_response["text_embedding_dict"]["embedding_json"]
                embedding_vector = json.loads(embedding_json)
                input_embedding = np.array(embedding_vector).astype('float32').reshape(1, -1)                
                faiss.normalize_L2(input_embedding)
                logger.info(f"Computed embedding for input text: {query_text}")
                final_results = []

                if faiss_index is None:
                    raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {llm_model_name} and pooling method: {embedding_pooling_method}")

                num_results = max([1, int((1 - similarity_filter_percentage) * len(associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]))])
                num_results_before_corpus_filter = min(num_results_before_corpus_filter, len(associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]))
                similarities, indices = faiss_index.search(input_embedding, num_results_before_corpus_filter)
                filtered_indices = indices[0]
                filtered_similarities = similarities[0]
                similarity_results = []
                associated_texts = associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method]
                list_of_corpus_identifier_strings = await get_list_of_corpus_identifiers_from_list_of_embedding_texts(associated_texts, llm_model_name, embedding_pooling_method)
                
                for idx, similarity in zip(filtered_indices, filtered_similarities):
                    if idx < len(associated_texts) and list_of_corpus_identifier_strings[idx] == corpus_identifier_string:
                        associated_text = associated_texts[idx]
                        similarity_results.append((similarity, associated_text))
                        
                similarity_results = sorted(similarity_results, key=lambda x: x[0], reverse=True)[:num_results]
                
                for _, associated_text in similarity_results:
                    embedding_request = EmbeddingRequest(text=associated_text, llm_model_name=llm_model_name, embedding_pooling_method=embedding_pooling_method)
                    embedding_response = await get_or_compute_embedding(request=embedding_request, db_writer=db_writer, use_verbose=False)
                    embedding_json = embedding_response["text_embedding_dict"]["embedding_json"]
                    embedding_vector = json.loads(embedding_json)
                    comparison_embedding = np.array(embedding_vector).astype('float32').reshape(1, -1)                 
                    params = {
                        "vector_1": input_embedding.tolist()[0],
                        "vector_2": comparison_embedding.tolist()[0],
                        "similarity_measure": "all"
                    }
                    similarity_stats_str = fvs.py_compute_vector_similarity_stats(json.dumps(params))
                    similarity_stats_json = json.loads(similarity_stats_str)
                    final_results.append({
                        "search_result_text": associated_text,
                        "similarity_to_query_text": similarity_stats_json
                    })

                num_to_return = number_of_most_similar_strings_to_return if number_of_most_similar_strings_to_return is not None else len(final_results)
                results = sorted(final_results, key=lambda x: x["similarity_to_query_text"][result_sorting_metric], reverse=True)[:num_to_return]
                
                response_time = datetime.utcnow()
                total_time = (response_time - request_time).total_seconds()
                logger.info(f"Finished advanced search in {total_time} seconds. Found {len(results)} results.")
                
                return {
                    "query_text": query_text, 
                    "corpus_identifier_string": corpus_identifier_string,
                    "embedding_pooling_method": embedding_pooling_method,
                    "results": results
                }
            
            except Exception as e:
                logger.error(f"An error occurred while processing the request: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail="Internal Server Error")
            
        finally:
            # Clean up the db_writer_task when done
            if not db_writer_task.done():
                db_writer_task.cancel()
                try:
                    await db_writer_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        error_msg = f"Error in scan task: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        job.meta['progress'] = 100
        job.save_meta()
        return {
            "status": "error",
            "message": error_msg
        }
    finally:
        job.meta['progress'] = 100
        job.save_meta()
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
