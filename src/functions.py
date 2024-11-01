from hashlib import sha3_256, md5
from utils import build_faiss_indexes, is_redis_running, start_redis_server
from db import AsyncSessionLocal, execute_with_retry
from models import TextEmbedding, DocumentEmbedding, Document
from models import EmbeddingRequest
import os
import re
import unicodedata
import shutil
import psutil
import glob
import json
import io
import zipfile
import tempfile
import traceback
import time
from datetime import datetime
from hashlib import sha3_256
from urllib.parse import quote
import numpy as np
import pandas as pd
import textract
import zstandard as zstd
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.inspection import inspect
from fastapi import HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional, Dict, Any
from faster_whisper import WhisperModel
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava16ChatHandler
from llama_cpp import llama_types
from mutagen import File as MutagenFile
from magika import Magika
import httpx
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
import logging

from db import DatabaseWriter, initialize_db
from redis.asyncio import Redis as AsyncRedis
from redis import Redis
from redis.exceptions import LockError
import asyncio
import urllib.request
import os
import glob
import json
from filelock import FileLock, Timeout
import traceback
import llama_cpp
from typing import List, Tuple, Dict
from fastapi import HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from rq import Queue
from rq.job import Job
from utils import download_models, add_model_url
import urllib


logger = logging.getLogger(__name__)

try:
    import nvgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

embedding_model_cache = {} # Model cache to store loaded models
text_completion_model_cache = {} # Model cache to store loaded text completion models

CODEXIFY_API_SERVER_LISTEN_PORT = int(os.getenv("CODEXIFY_API_SERVER_LISTEN_PORT", "8089"))
DEFAULT_LLM_NAME = os.getenv("DEFAULT_LLM_NAME", "openchat_v3.2_super")
LLM_CONTEXT_SIZE_IN_TOKENS = int(os.getenv("LLM_CONTEXT_SIZE_IN_TOKENS", "512"))
TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS = int(os.getenv("TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS", "4000"))
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("DEFAULT_MAX_COMPLETION_TOKENS", "100"))
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE = int(os.getenv("DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE", "4"))
DEFAULT_COMPLETION_TEMPERATURE = float(os.getenv("DEFAULT_COMPLETION_TEMPERATURE", "0.7"))
MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING = int(os.getenv("MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING", "15"))
USE_PARALLEL_INFERENCE_QUEUE = os.getenv("USE_PARALLEL_INFERENCE_QUEUE", "False").lower() == "true"
MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS = int(os.getenv("MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS", "10"))
USE_VERBOSE = os.getenv("USE_VERBOSE", "False").lower() == "true"
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def is_gpu_available():
    if not GPU_AVAILABLE:
        return {
            "gpu_found": False,
            "num_gpus": 0,
            "first_gpu_vram": 0,
            "total_vram": 0,
            "error": "nvgpu module not found"
        }
    try:
        gpu_info = nvgpu.gpu_info()
        num_gpus = len(gpu_info)
        if num_gpus == 0:
            return {
                "gpu_found": False,
                "num_gpus": 0,
                "first_gpu_vram": 0,
                "total_vram": 0
            }
        first_gpu_vram = gpu_info[0]['mem_total']
        total_vram = sum(gpu['mem_total'] for gpu in gpu_info)
        return {
            "gpu_found": True,
            "num_gpus": num_gpus,
            "first_gpu_vram": first_gpu_vram,
            "total_vram": total_vram,
            "gpu_info": gpu_info
        }
    except Exception as e:
        return {
            "gpu_found": False,
            "num_gpus": 0,
            "first_gpu_vram": 0,
            "total_vram": 0,
            "error": str(e)
        }
        
# async def initialize_globals():
#     global db_writer, faiss_indexes, associated_texts_by_model_and_pooling_method, redis, lock_manager
#     if not is_redis_running():
#         logger.info("Starting Redis server...")
#         start_redis_server()
#         await asyncio.sleep(1)  # Sleep for 1 second to give Redis time to start
#     redis = Redis.from_url('redis://localhost')
#     # Remove the lock_manager initialization for now
#     # lock_manager = Aioredlock([redis])
#     # lock_manager.default_lock_timeout = 20000  # Lock timeout in milliseconds (20 seconds)
#     # lock_manager.retry_count = 5  # Number of retries
#     # lock_manager.retry_delay_min = 100  # Minimum delay between retries in milliseconds
#     # lock_manager.retry_delay_max = 1000  # Maximum delay between retries in milliseconds
#     await initialize_db()
#     queue = asyncio.Queue()
#     db_writer = DatabaseWriter(queue)
#     await db_writer.initialize_processing_hashes()
#     asyncio.create_task(db_writer.dedicated_db_writer())
#     list_of_downloaded_model_names, download_status = download_models()
#     faiss_indexes, associated_texts_by_model_and_pooling_method = await build_faiss_indexes()


# # other shared variables and methods
# db_writer = None
# faiss_indexes = None
# associated_texts_by_model_and_pooling_method = None
# redis = None
# lock_manager = None

class RedisManager:
    def __init__(self):
        self.redis_sync = None     # For RQ job queues
        self.redis_async = None    # For async operations
        self.queues = {}
        self.model_queue = None
        self.db_writer = None
        self.faiss_indexes = None
        self.associated_texts_by_model_and_pooling_method = None

    async def initialize(self):
        """Initialize Redis for async operations and job queues"""
        if not is_redis_running():
            logger.info("Starting Redis server...")
            start_redis_server()
            await asyncio.sleep(1)  # Give Redis time to start

        # Initialize Redis connections
        self.redis_sync = Redis(host='localhost', port=6379, db=0)
        self.redis_async = AsyncRedis(host='localhost', port=6379, db=0)

        # Initialize RQ queues
        self.queues['model_downloads'] = Queue('model_downloads', connection=self.redis_sync)
        self.queues['file_uploads'] = Queue('file_uploads', connection=self.redis_sync)
        self.queues['document_scans'] = Queue('document_scans', connection=self.redis_sync)

        # Initialize database writer and other components
        await self._initialize_components()

    async def _initialize_components(self):
        """Initialize database writer and other necessary components"""
        await initialize_db()
        queue = asyncio.Queue()
        self.db_writer = DatabaseWriter(queue)
        await self.db_writer.initialize_processing_hashes()
        asyncio.create_task(self.db_writer.dedicated_db_writer())
        
        # Initialize models and FAISS indexes
        list_of_downloaded_model_names, download_status = download_models()
        self.faiss_indexes, self.associated_texts_by_model_and_pooling_method = (
            await build_faiss_indexes()
        )

    def get_queue(self, queue_name):
        return self.queues.get(queue_name)

# Global instance
redis_manager = RedisManager()

async def initialize_globals():
    """Initialize global Redis manager and components"""
    global db_writer, faiss_indexes, associated_texts_by_model_and_pooling_method
    
    await redis_manager.initialize()
    
    # Update global variables
    db_writer = redis_manager.db_writer
    faiss_indexes = redis_manager.faiss_indexes
    associated_texts_by_model_and_pooling_method = redis_manager.associated_texts_by_model_and_pooling_method

    # You can access queues like this:
    # model_downloads_queue = redis_manager.get_queue('model_downloads')
    # file_uploads_queue = redis_manager.get_queue('file_uploads')
    # document_scans_queue = redis_manager.get_queue('document_scans')

    


def load_model(llm_model_name: str, raise_http_exception: bool = True):
    """Load and return a Llama model instance."""
    global USE_VERBOSE
    try:
        # Check if model is already cached
        if llm_model_name in embedding_model_cache:
            return embedding_model_cache[llm_model_name]

        # Find the model file
        models_dir = os.path.join(BASE_DIRECTORY, 'models')
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logger.error(f"No model file found matching: {llm_model_name}")
            if raise_http_exception:
                raise HTTPException(status_code=404, detail="Model file not found")
            raise FileNotFoundError(f"No model file found matching: {llm_model_name}")

        # Get the most recently modified matching file
        matching_files.sort(key=os.path.getmtime, reverse=True)
        model_file_path = matching_files[0]

        # Initialize model with GPU if available
        gpu_info = is_gpu_available()
        is_llava_multimodal_model = 'llava' in llm_model_name

        if not is_llava_multimodal_model:
            if gpu_info['gpu_found']:
                try:
                    model_instance = Llama(
                        model_path=model_file_path,
                        embedding=True,
                        n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS,
                        verbose=USE_VERBOSE,
                        n_gpu_layers=-1
                    )
                except Exception as e:
                    logger.warning(f"Failed to load model with GPU, falling back to CPU: {e}")
                    model_instance = Llama(
                        model_path=model_file_path,
                        embedding=True,
                        n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS,
                        verbose=USE_VERBOSE
                    )
            else:
                model_instance = Llama(
                    model_path=model_file_path,
                    embedding=True,
                    n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE
                )

            # Cache the model instance
            embedding_model_cache[llm_model_name] = model_instance
            return model_instance

    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_http_exception:
            raise HTTPException(status_code=404, detail="Model file not found")
        raise FileNotFoundError(f"No model file found matching: {llm_model_name}")

logger = logging.getLogger(__name__)
magika = Magika()
db_writer = None

CODEXIFY_API_SERVER_LISTEN_PORT = int(os.getenv("CODEXIFY_API_SERVER_LISTEN_PORT", "8089"))
DEFAULT_LLM_NAME = os.getenv("DEFAULT_LLM_NAME", "openchat_v3.2_super")
LLM_CONTEXT_SIZE_IN_TOKENS = int(os.getenv("LLM_CONTEXT_SIZE_IN_TOKENS", "512"))
TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS = int(os.getenv("TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS", "4000"))
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("DEFAULT_MAX_COMPLETION_TOKENS", "100"))
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE = int(os.getenv("DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE", "4"))
DEFAULT_COMPLETION_TEMPERATURE = float(os.getenv("DEFAULT_COMPLETION_TEMPERATURE", "0.7"))
MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING = int(os.getenv("MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING", "15"))
USE_PARALLEL_INFERENCE_QUEUE = os.getenv("USE_PARALLEL_INFERENCE_QUEUE", "False").lower() == "true"
MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS = int(os.getenv("MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS", "10"))
USE_VERBOSE = os.getenv("USE_VERBOSE", "False").lower() == "true"
USE_RESOURCE_MONITORING = os.getenv("USE_RESOURCE_MONITORING", "1").lower() == "true"
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "True").lower() == "true"
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    
# Core embedding functions start here:    
    
def prepare_string_for_embedding(text: str) -> str:
    # Normalize Unicode characters to NFKC form
    text = unicodedata.normalize('NFKC', text)
    # Define all possible newline and carriage return characters
    newline_chars = [
        '\r', '\n', '\r\n', '\u2028', '\u2029', '\v', '\f', 
        '\x85', '\u000A', '\u000B', '\u000C', '\u000D', '\u0085',
        '\u000D\u000A'
    ]
    # Replace all newline characters with a space
    for nl in newline_chars:
        text = text.replace(nl, ' ')
    # Replace any sequence of whitespace characters (including non-breaking spaces) with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    # Remove leading comma followed by whitespace if present
    if text.startswith(', '):
        text = text[2:].strip()
    # Remove all control characters and non-printable characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # Ensure text is ASCII-encoded to catch any remaining unusual characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Truncate to a maximum length of 5000 characters
    if len(text) > 5000:
        text = text[:5000]
    # Eliminate all blank lines
    text = ' '.join(line for line in text.splitlines() if line.strip() != '')
    #Final trimming
    text = text.strip()
    return text

def compress_data(input_data):
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')
    zstd_compression_level = 15 # 22 is the highest compression level; 15 is a good balance between compression and speed
    zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
    zstd_compressed_data = zstandard_compressor.compress(input_data)
    return zstd_compressed_data

def decompress_data(compressed_data):
    return zstd.decompress(compressed_data)

# def add_model_url(new_url: str) -> str:
#     corrected_url = new_url
#     if '/blob/main/' in new_url:
#         corrected_url = new_url.replace('/blob/main/', '/resolve/main/')
#     json_path = os.path.join(BASE_DIRECTORY, "model_urls.json")
#     with open(json_path, "r") as f:
#         existing_urls = json.load(f)
#     if corrected_url not in existing_urls:
#         logger.info(f"Model URL not found in database. Adding {new_url} now...")
#         existing_urls.append(corrected_url)
#         with open(json_path, "w") as f:
#             json.dump(existing_urls, f)
#         logger.info(f"Model URL added: {new_url}")
#     else:
#         logger.info("Model URL already exists.")        
#     return corrected_url  

async def get_embedding_from_db(text: str, llm_model_name: str, embedding_pooling_method: str):
    text_hash = sha3_256(text.encode('utf-8')).hexdigest()
    return await execute_with_retry(_get_embedding_from_db, text_hash, llm_model_name, embedding_pooling_method)

async def _get_embedding_from_db(text_hash: str, llm_model_name: str, embedding_pooling_method: str) -> Optional[TextEmbedding]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TextEmbedding)
            .filter(TextEmbedding.text_hash == text_hash,
                    TextEmbedding.llm_model_name == llm_model_name,
                    TextEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        return result.scalars().first()
    
async def get_corpus_identifier_from_embedding_text(text: str, llm_model_name: str, embedding_pooling_method: str):
    text_hash = sha3_256(text.encode('utf-8')).hexdigest()
    return await execute_with_retry(_get_corpus_identifier_from_embedding_text, text_hash, llm_model_name, embedding_pooling_method)

async def _get_corpus_identifier_from_embedding_text(text_hash: str, llm_model_name: str, embedding_pooling_method: str) -> Optional[str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TextEmbedding.corpus_identifier_string)
            .filter(TextEmbedding.text_hash == text_hash,
                    TextEmbedding.llm_model_name == llm_model_name,
                    TextEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        return result.scalar()

async def get_list_of_corpus_identifiers_from_list_of_embedding_texts(list_of_texts: List[str], llm_model_name: str, embedding_pooling_method: str):
    list_of_text_hashes = [sha3_256(text.encode('utf-8')).hexdigest() for text in list_of_texts]
    return await execute_with_retry(_get_list_of_corpus_identifiers_from_list_of_embedding_texts, list_of_text_hashes, llm_model_name, embedding_pooling_method)

async def _get_list_of_corpus_identifiers_from_list_of_embedding_texts(list_of_text_hashes: List[str], llm_model_name: str, embedding_pooling_method: str) -> List[str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TextEmbedding.corpus_identifier_string)
            .filter(TextEmbedding.text_hash.in_(list_of_text_hashes),
                    TextEmbedding.llm_model_name == llm_model_name,
                    TextEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        rows = result.scalars().all()
        return rows
    
async def get_texts_for_corpus_identifier(corpus_identifier_string: str) -> Dict[str, List[str]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DocumentEmbedding)
            .options(joinedload(DocumentEmbedding.embeddings))
            .filter(DocumentEmbedding.corpus_identifier_string == corpus_identifier_string)
        )
        document_embeddings = result.unique().scalars().all()
        texts_by_model_and_embedding_pooling_method = {(doc.llm_model_name, doc.embedding_pooling_method): [] for doc in document_embeddings}
        for document_embedding in document_embeddings:
            texts_by_model_and_embedding_pooling_method[(document_embedding.llm_model_name, document_embedding.embedding_pooling_method)].extend(
                [embedding.text for embedding in document_embedding.embeddings]
            )
    return texts_by_model_and_embedding_pooling_method

async def get_texts_for_model_and_embedding_pooling_method(llm_model_name: str, embedding_pooling_method: str) -> Dict[str, List[str]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DocumentEmbedding)
            .options(joinedload(DocumentEmbedding.embeddings))
            .filter(DocumentEmbedding.llm_model_name == llm_model_name, DocumentEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        document_embeddings = result.unique().scalars().all()
        texts_by_model_and_embedding_pooling_method = {(doc.llm_model_name, doc.embedding_pooling_method): [] for doc in document_embeddings}
        for document_embedding in document_embeddings:
            texts_by_model_and_embedding_pooling_method[(document_embedding.llm_model_name, document_embedding.embedding_pooling_method)].extend(
                [embedding.text for embedding in document_embedding.embeddings]
            )
    return texts_by_model_and_embedding_pooling_method

async def get_or_compute_embedding(embedding_request: EmbeddingRequest, db_writer) -> Dict[str, Any]:
    """Get or compute embedding for the given text."""
    try:
        # Initialize Redis manager if not already done
        redis_manager = RedisManager()
        await redis_manager.initialize()
        
        # Create cache key
        cache_key = f"embedding:{embedding_request.llm_model_name}:{embedding_request.embedding_pooling_method}:{md5(embedding_request.text.encode()).hexdigest()}"
        
        # Try to get from cache first
        try:
            cached_result = await redis_manager.redis_async.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        
        # Load the model
        model = load_model(embedding_request.llm_model_name)
        
        # If not in cache, compute embedding
        try:
            embedding_instance = await calculate_sentence_embeddings_list(
                model,  # Pass the model instance instead of text
                [embedding_request.text],  # Pass text as a list
                embedding_request.embedding_pooling_method
            )
        except Exception as e:
            logger.error(f"Error computing embedding: {str(e)}")
            raise
        
        # Store in database if db_writer is provided
        if db_writer:
            try:
                await db_writer.enqueue_write([embedding_instance])
            except Exception as e:
                logger.warning(f"Database write failed: {str(e)}")
        
        # Store in cache
        result = {
            "text_embedding_dict": {
                "text": embedding_request.text,
                "embedding_json": json.dumps(embedding_instance[0]['embedding']),  # Get first embedding since we passed a single text
                "llm_model_name": embedding_request.llm_model_name,
                "embedding_pooling_method": embedding_request.embedding_pooling_method
            }
        }
        
        try:
            await redis_manager.redis_async.set(
                cache_key, 
                json.dumps(result), 
                ex=3600  # Cache for 1 hour
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_or_compute_embedding: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def calculate_sentence_embeddings_list(model, sentences: List[str], embedding_pooling_method: str) -> List[Dict]:
    """Calculate embeddings for a list of sentences."""
    try:
        start_time = datetime.utcnow()
        embeddings_list = []
        total_chars = sum(len(s) for s in sentences)
        
        for i, sentence in enumerate(sentences, 1):
            try:
                # Get embedding from model
                if hasattr(model, 'embed'):  # For some models like nomic
                    embedding = model.embed(sentence)
                elif hasattr(model, 'create_embedding'):  # For llama.cpp models
                    embedding = model.create_embedding(sentence)['embedding']
                else:
                    raise ValueError(f"Model {type(model)} doesn't support embedding generation")
                
                # Convert to numpy array if needed
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Apply pooling
                if embedding_pooling_method == "mean":
                    pooled_embedding = np.mean(embedding, axis=0) if embedding.ndim > 1 else embedding
                elif embedding_pooling_method == "max":
                    pooled_embedding = np.max(embedding, axis=0) if embedding.ndim > 1 else embedding
                else:
                    pooled_embedding = embedding
                
                # Create hash of the embedding
                embedding_hash = sha3_256(pooled_embedding.tobytes()).hexdigest()
                
                embeddings_list.append({
                    'embedding': pooled_embedding,
                    'embedding_hash': embedding_hash
                })
                
                logger.info(f"Sentence {i} of {len(sentences)} has {len(embedding)} embeddings for text '{sentence[:50]}...'")
                
            except Exception as e:
                logger.error(f"Error processing sentence {i}: {str(e)}")
                raise
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Calculated {len(embeddings_list[0]['embedding'])}-dimensional embeddings for {len(sentences)} sentences in a total of {total_time:.1f} seconds.")
        logger.info(f"That's an average of {(total_time * 1000 / len(sentences)):.2f} ms per sentence and {(len(sentences) / total_time):.3f} sentences per second (and {(total_chars / total_time / 1000):.4f} total characters per ms) using pooling method '{embedding_pooling_method}'")
        
        return embeddings_list
    
    except Exception as e:
        logger.error(f"Error in calculate_sentence_embeddings_list: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def batch_save_embeddings_to_db(embeddings: List[TextEmbedding]):
    async with AsyncSessionLocal() as session:
        # Extract the unique embedding_hashes from the embeddings list
        embedding_hashes = [embedding.embedding_hash for embedding in embeddings]
        # Query the database for existing embeddings with the same hashes
        existing_embeddings_query = select(TextEmbedding.embedding_hash).where(TextEmbedding.embedding_hash.in_(embedding_hashes))
        result = await session.execute(existing_embeddings_query)
        existing_embedding_hashes = {row.embedding_hash for row in result}
        # Filter out embeddings that already exist in the database
        embeddings_to_insert = [embedding for embedding in embeddings if embedding.embedding_hash not in existing_embedding_hashes]
        # Batch insert the remaining embeddings
        if embeddings_to_insert:
            session.add_all(embeddings_to_insert)
            await session.commit()
            
async def compute_embeddings_for_document(sentences: list, llm_model_name: str, embedding_pooling_method: str, corpus_identifier_string: str, client_ip: str, document_file_hash: str, file: UploadFile, original_file_content: bytes, json_format: str = 'records') -> list:
    request_time = datetime.utcnow()
    sentences = [prepare_string_for_embedding(text) for text in sentences]
    model = load_model(llm_model_name)
    try:
        list_of_embedding_entry_dicts = await calculate_sentence_embeddings_list(model, sentences, embedding_pooling_method)
    except Exception as e:
        logger.error(f"Error computing embeddings for batch: {e}")
        logger.error(traceback.format_exc())
        raise
    embeddings_to_save = []
    list_of_embedding_hashes_added = []
    for embedding_entry_dict in list_of_embedding_entry_dicts:
        embedding = embedding_entry_dict['embedding']
        embedding_hash = embedding_entry_dict['embedding_hash']
        if embedding_hash in list_of_embedding_hashes_added:
            continue
        text_index = embedding_entry_dict['text_index']
        text = sentences[text_index]
        text_hash = sha3_256(text.encode('utf-8')).hexdigest()
        embedding_json = json.dumps(embedding)
        response_time = datetime.utcnow()
        total_time = (response_time - request_time).total_seconds()
        embedding_instance = TextEmbedding(
            text=text,
            text_hash=text_hash,
            embedding_hash=embedding_hash,
            llm_model_name=llm_model_name,
            embedding_pooling_method=embedding_pooling_method,
            corpus_identifier_string=corpus_identifier_string,
            embedding_json=embedding_json,
            ip_address=client_ip,
            request_time=request_time,
            response_time=response_time,
            total_time=total_time,
            document_file_hash=document_file_hash,
        )
        embeddings_to_save.append(embedding_instance)
        list_of_embedding_hashes_added.append(embedding_hash)
    logger.info(f"Storing {len(embeddings_to_save):,} text embeddings in database...")
    await batch_save_embeddings_to_db(embeddings_to_save)
    logger.info(f"Done storing {len(embeddings_to_save):,} text embeddings in database.")
    document_embedding_results_df = pd.DataFrame(list_of_embedding_entry_dicts)
    json_content = document_embedding_results_df.to_json(orient=json_format or 'records').encode()
    if file is not None:
        filename = file.filename
        await store_document_embeddings_in_db(
            file=file,
            filename=filename,
            document_file_hash=document_file_hash,
            original_file_content=original_file_content,
            sentences=sentences,
            json_content=json_content,
            llm_model_name=llm_model_name,
            embedding_pooling_method=embedding_pooling_method,
            corpus_identifier_string=corpus_identifier_string,
            client_ip=client_ip,
            request_time=request_time,
        )    
    return json_content

async def parse_submitted_document_file_into_sentence_strings_func(temp_file_path: str, mime_type: str):
    content = ""
    try:
        content = textract.process(temp_file_path, method='pdfminer', encoding='utf-8')
        content = content.decode('utf-8')
    except Exception as e:
        logger.error(f"Error while processing file: {e}, mime_type: {mime_type}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Unsupported file type or error: {e}")
    sentences = sophisticated_sentence_splitter(content)
    if len(sentences) == 0 and temp_file_path.lower().endswith('.pdf'):
        logger.info("No sentences found, attempting OCR using Tesseract.")
        try:
            content = textract.process(temp_file_path, method='tesseract', encoding='utf-8')
            content = content.decode('utf-8')
            sentences = sophisticated_sentence_splitter(content)
        except Exception as e:
            logger.error(f"Error while processing file with OCR: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail="OCR failed: {e}")
    if len(sentences) == 0:
        logger.info("No sentences found in the document")
        raise HTTPException(status_code=400, detail="No sentences found in the document")
    strings = [s.strip() for s in sentences if len(s.strip()) > MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING]
    thousands_of_input_words = round(sum(len(s.split()) for s in strings) / 1000, 2)
    return strings, thousands_of_input_words

async def _get_document_from_db(document_file_hash: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document).filter(Document.document_hash == document_file_hash))
        return result.scalar_one_or_none()

async def store_document_embeddings_in_db(file, document_file_hash: str, original_file_content: bytes, sentences: List[str], json_content: bytes, llm_model_name: str, embedding_pooling_method:str, corpus_identifier_string: str, client_ip: str, request_time: datetime, filename: str):
    if file is None:
        logger.error("Received a None file object in store_document_embeddings_in_db")
    else:
        logger.info(f"Received file: {filename} with content type: {file.content_type if file else 'Unknown'}")
    sentences = json.dumps(sentences)
    document = await _get_document_from_db(document_file_hash)
    if not document:
        document = Document(document_hash=document_file_hash, llm_model_name=llm_model_name, corpus_identifier_string=corpus_identifier_string, filename=filename)
        await db_writer.enqueue_write([document])
    document_embedding_results_json_compressed_binary = compress_data(json_content)
    document_embedding = DocumentEmbedding(
        filename=filename,
        mimetype=file.content_type if file else None,
        document_file_hash=document_file_hash,
        llm_model_name=llm_model_name,
        embedding_pooling_method=embedding_pooling_method,
        corpus_identifier_string=corpus_identifier_string,
        file_data=original_file_content,
        sentences=sentences,
        document_embedding_results_json_compressed_binary=document_embedding_results_json_compressed_binary,
        ip_address=client_ip,
        request_time=request_time,
        response_time=datetime.utcnow(),
        total_time=(datetime.utcnow() - request_time).total_seconds()
    )
    document.document_embeddings.append(document_embedding)
    document.update_hash()
    await db_writer.enqueue_write([document, document_embedding])

async def convert_document_to_sentences_func(file_path: str, mime_type: str) -> Dict[str, Any]:
    sentences, thousands_of_input_words = await parse_submitted_document_file_into_sentence_strings_func(file_path, mime_type)
    total_number_of_sentences = len(sentences)
    total_input_file_size_in_bytes = os.path.getsize(file_path)
    total_text_size_in_characters = sum(len(sentence) for sentence in sentences)
    total_words = sum(len(sentence.split()) for sentence in sentences)
    average_words_per_sentence = total_words / total_number_of_sentences if total_number_of_sentences else 0
    result = {
        "individual_sentences": sentences,
        "total_number_of_sentences": total_number_of_sentences,
        "average_words_per_sentence": average_words_per_sentence,
        "total_input_file_size_in_bytes": total_input_file_size_in_bytes,
        "total_text_size_in_characters": total_text_size_in_characters,
        "thousands_of_input_words": thousands_of_input_words
    }
    return result

async def download_file(url: str, expected_size: int, expected_hash: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    hash_obj = sha3_256()
    downloaded_size = 0
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download file")
            async for chunk in response.aiter_bytes():
                downloaded_size += len(chunk)
                if downloaded_size > expected_size:
                    os.remove(temp_file_path)
                    raise HTTPException(status_code=400, detail="Downloaded file size exceeds expected size")
                temp_file.write(chunk)
                hash_obj.update(chunk)
    temp_file.close()
    if downloaded_size != expected_size:
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail="Downloaded file size does not match expected size")
    if hash_obj.hexdigest() != expected_hash:
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail="File hash mismatch")
    return temp_file_path

def sophisticated_sentence_splitter(text):
    text = remove_pagination_breaks(text)
    pattern = r'\.(?!\s*(com|net|org|io)\s)(?![0-9])'  # Split on periods that are not followed by a space and a top-level domain or a number
    pattern += r'|[.!?]\s+'  # Split on whitespace that follows a period, question mark, or exclamation point
    pattern += r'|\.\.\.(?=\s)'  # Split on ellipses that are followed by a space
    sentences = re.split(pattern, text)
    refined_sentences = []
    temp_sentence = ""
    for sentence in sentences:
        if sentence is not None:
            temp_sentence += sentence
            if temp_sentence.count('"') % 2 == 0:  # If the number of quotes is even, then we have a complete sentence
                refined_sentences.append(temp_sentence.strip())
                temp_sentence = ""
    if temp_sentence:
        refined_sentences[-1] += temp_sentence
    return [s.strip() for s in refined_sentences if s.strip()]

def remove_pagination_breaks(text: str) -> str:
    text = re.sub(r'-(\n)(?=[a-z])', '', text) # Remove hyphens at the end of lines when the word continues on the next line
    text = re.sub(r'(?<=\w)(?<![.?!-]|\d)\n(?![\nA-Z])', ' ', text) # Replace line breaks that are not preceded by punctuation or list markers and not followed by an uppercase letter or another line break   
    return text

def truncate_string(s: str, max_length: int = 100) -> str:
    return s[:max_length]

def enqueue_model_download(model_url: str):
    queue = redis_manager.get_queue('model_downloads')
    job = queue.enqueue(download_model_task, model_url)
    return job.id

def get_job_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_manager.redis_sync)
    if job.is_finished:
        return {"status": "completed", "result": job.result, "progress": 100}
    elif job.is_failed:
        return {"status": "failed", "error": str(job.exc_info), "progress": 100}
    else:
        return {"status": "in_progress", "progress": job.meta.get('progress', 0)}

