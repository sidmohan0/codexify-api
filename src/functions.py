
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
from aioredlock import Aioredlock
import redis.asyncio as redis
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
import logging
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
        
async def initialize_globals():
    global db_writer, faiss_indexes, associated_texts_by_model_and_pooling_method, redis, lock_manager
    if not is_redis_running():
        logger.info("Starting Redis server...")
        start_redis_server()
        await asyncio.sleep(1)  # Sleep for 1 second to give Redis time to start
    import redis.asyncio
    redis = await redis.asyncio.from_url('redis://localhost')
    lock_manager = Aioredlock([redis])
    lock_manager.default_lock_timeout = 20000  # Lock timeout in milliseconds (20 seconds)
    lock_manager.retry_count = 5  # Number of retries
    lock_manager.retry_delay_min = 100  # Minimum delay between retries in milliseconds
    lock_manager.retry_delay_max = 1000  # Maximum delay between retries in milliseconds
    await initialize_db()
    queue = asyncio.Queue()
    db_writer = DatabaseWriter(queue)
    await db_writer.initialize_processing_hashes()
    asyncio.create_task(db_writer.dedicated_db_writer())
    list_of_downloaded_model_names, download_status = download_models()
    faiss_indexes, associated_texts_by_model_and_pooling_method = await build_faiss_indexes()

# other shared variables and methods
db_writer = None
faiss_indexes = None
associated_texts_by_model_and_pooling_method = None
redis = None
lock_manager = None

def download_models() -> Tuple[List[str], List[Dict[str, str]]]:
    download_status = []    
    json_path = os.path.join(BASE_DIRECTORY, "model_urls.json")
    if not os.path.exists(json_path):
        initial_model_urls = [
            "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf",
            "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q3_K_S.gguf",
            "https://huggingface.co/vonjack/bge-m3-gguf/resolve/main/bge-m3-q8_0.gguf"
        ]
        with open(json_path, "w") as f:
            json.dump(initial_model_urls, f)
    with open(json_path, "r") as f:
        list_of_model_download_urls = json.load(f)
    model_names = [os.path.basename(url) for url in list_of_model_download_urls]
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    logger.info("Checking models directory...")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")
    else:
        logger.info(f"Models directory exists: {models_dir}")
    lock = FileLock(os.path.join(models_dir, "download.lock"))
    for url, model_name_with_extension in zip(list_of_model_download_urls, model_names):
        status = {"url": url, "status": "success", "message": "File already exists."}
        filename = os.path.join(models_dir, model_name_with_extension)
        try:
            with lock.acquire(timeout=1200): # Wait up to 20 minutes for the file to be downloaded before returning failure
                if not os.path.exists(filename):
                    logger.info(f"Downloading model {model_name_with_extension} from {url}...")
                    urllib.request.urlretrieve(url, filename)
                    file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert bytes to MB
                    if file_size < 100:
                        os.remove(filename)
                        status["status"] = "failure"
                        status["message"] = "Downloaded file is too small, probably not a valid model file."
                    else:
                        logger.info(f"Downloaded: {filename}")
                else:
                    logger.info(f"File already exists: {filename}")
        except Timeout:
            logger.warning(f"Could not acquire lock for downloading {model_name_with_extension}")
            status["status"] = "failure"
            status["message"] = "Could not acquire lock for downloading."
        download_status.append(status)
    
    logger.info("Model downloads completed.")
    return model_names, download_status

def load_model(llm_model_name: str, raise_http_exception: bool = True):
    global USE_VERBOSE
    model_instance = None
    try:
        models_dir = os.path.join(BASE_DIRECTORY, 'models')
        if llm_model_name in embedding_model_cache:
            return embedding_model_cache[llm_model_name]
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logger.error(f"No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True)
        model_file_path = matching_files[0]
        gpu_info = is_gpu_available()
        if 'llava' in llm_model_name:
            is_llava_multimodal_model = 1
        else:
            is_llava_multimodal_model = 0
        if not is_llava_multimodal_model:
            if gpu_info['gpu_found']:
                try:
                    model_instance = llama_cpp.Llama(model_path=model_file_path, embedding=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS, verbose=USE_VERBOSE, n_gpu_layers=-1) # Load the model with GPU acceleration
                except Exception as e:  # noqa: F841
                    model_instance = llama_cpp.Llama(model_path=model_file_path, embedding=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS, verbose=USE_VERBOSE)
            else:
                model_instance = llama_cpp.Llama(model_path=model_file_path, embedding=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS, verbose=USE_VERBOSE) # Load the model without GPU acceleration        
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
        else:
            raise FileNotFoundError(f"No model file found matching: {llm_model_name}")

logger = logging.getLogger(__name__)
magika = Magika()
db_writer = None

SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT = int(os.getenv("SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT", "8089"))
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

def add_model_url(new_url: str) -> str:
    corrected_url = new_url
    if '/blob/main/' in new_url:
        corrected_url = new_url.replace('/blob/main/', '/resolve/main/')
    json_path = os.path.join(BASE_DIRECTORY, "model_urls.json")
    with open(json_path, "r") as f:
        existing_urls = json.load(f)
    if corrected_url not in existing_urls:
        logger.info(f"Model URL not found in database. Adding {new_url} now...")
        existing_urls.append(corrected_url)
        with open(json_path, "w") as f:
            json.dump(existing_urls, f)
        logger.info(f"Model URL added: {new_url}")
    else:
        logger.info("Model URL already exists.")        
    return corrected_url  

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

async def get_or_compute_embedding(request: EmbeddingRequest, req: Request = None, client_ip: str = None, document_file_hash: str = None, use_verbose: bool = True) -> dict:
    request_time = datetime.utcnow()  # Capture request time as datetime object
    ip_address = (
        client_ip or (req.client.host if req else "localhost")
    )  # If client_ip is provided, use it; otherwise, try to get from req; if not available, default to "localhost"
    if use_verbose:
        logger.info(f"Received request for embedding for '{request.text}' using model '{request.llm_model_name}' and embedding pooling method '{request.embedding_pooling_method}' from IP address '{ip_address}'")
    text_embedding_instance = await get_embedding_from_db(
        request.text, request.llm_model_name, request.embedding_pooling_method
    )
    if text_embedding_instance is not None: # Check if embedding exists in the database
        response_time = datetime.utcnow()  # Capture response time as datetime object
        total_time = (
            response_time - request_time
        ).total_seconds()  # Calculate time taken in seconds
        if use_verbose:
            logger.info(f"Embedding found in database for '{request.text}' using model '{request.llm_model_name}' and embedding pooling method '{request.embedding_pooling_method}'; returning in {total_time:.4f} seconds")
        return {"text_embedding_dict": text_embedding_instance.as_dict()}
    model = load_model(request.llm_model_name)
    # Compute the embedding if not in the database
    list_of_embedding_entry_dicts = await calculate_sentence_embeddings_list(model, [request.text], request.embedding_pooling_method)
    embedding_entry_dict = list_of_embedding_entry_dicts[0]
    if embedding_entry_dict is None:
        logger.error(
            f"Could not calculate the embedding for the given text: '{request.text}' using model '{request.llm_model_name} and embedding pooling method '{request.embedding_pooling_method}!'"
        )
        raise HTTPException(
            status_code=400,
            detail="Could not calculate the embedding for the given text",
        )
    else:
        embedding = embedding_entry_dict['embedding']
        embedding_hash = embedding_entry_dict['embedding_hash']
        text = request.text
        text_hash = sha3_256(text.encode('utf-8')).hexdigest()
        embedding_json = json.dumps(embedding)
        request_time = datetime.utcnow()
        response_time = datetime.utcnow()
        total_time = (response_time - request_time).total_seconds()
        embedding_instance = TextEmbedding(
            text=text,
            text_hash=text_hash,
            embedding_hash=embedding_hash,
            llm_model_name=request.llm_model_name,
            embedding_pooling_method=request.embedding_pooling_method,
            corpus_identifier_string=request.corpus_identifier_string,
            embedding_json=embedding_json,
            ip_address=client_ip,
            request_time=request_time,
            response_time=response_time,
            total_time=total_time,
            document_file_hash=document_file_hash,
        )
    word_length_of_input_text = len(request.text.split())
    if word_length_of_input_text > 0:
        if use_verbose:
            logger.info(f"Embedding calculated for '{request.text}' using model '{request.llm_model_name}' and embedding pooling method '{request.embedding_pooling_method}' in {total_time:,.2f} seconds, or an average of {total_time/word_length_of_input_text :.2f} seconds per word. Now saving to database...")
    await db_writer.enqueue_write([embedding_instance])  # Enqueue the write operation using the db_writer instance directly
    return {"text_embedding_dict": embedding_instance.as_dict()}

async def calculate_sentence_embeddings_list(llama, texts: list, embedding_pooling_method: str) -> list:
    start_time = datetime.utcnow()
    total_number_of_sentences = len(texts)
    total_characters = sum(len(s) for s in texts)
    sentence_embeddings_object = llama.create_embedding(texts)
    sentence_embeddings_list = sentence_embeddings_object['data']
    if len(sentence_embeddings_list) != len(texts):
        raise ValueError("Inconsistent number of embeddings found.")
    list_of_embedding_entry_dicts = []
    cnt = 0
    for i, current_text in enumerate(texts):
        current_set_of_embeddings = sentence_embeddings_list[i]['embedding']
        if isinstance(current_set_of_embeddings[0], list):
            number_of_embeddings = len(current_set_of_embeddings)
        else:
            number_of_embeddings = 1
            current_set_of_embeddings = [current_set_of_embeddings]
        logger.info(f"Sentence {i + 1:,} of {len(texts):,} has {number_of_embeddings:,} embeddings for text '{current_text[:50]}...'")
        embeddings = np.array(current_set_of_embeddings)
        dimension_of_token_embeddings = embeddings.shape[1]
        # Ensure embeddings have enough dimensions for the pooling method
        required_components = {
            "svd": 2,
            "svd_first_four": 4,
            "ica": 2,
            "factor_analysis": 2,
            "gaussian_random_projection": 2
        }
        if number_of_embeddings > 1:
            min_components = required_components.get(embedding_pooling_method, 1)
            if number_of_embeddings < min_components:
                padding = np.zeros((min_components - number_of_embeddings, dimension_of_token_embeddings))
                embeddings = np.vstack([embeddings, padding])
            if embedding_pooling_method == "mean":
                element_wise_mean = np.mean(embeddings, axis=0)
                flattened_vector = element_wise_mean.flatten()
            elif embedding_pooling_method == "mins_maxes":
                element_wise_min = np.min(embeddings, axis=0)
                element_wise_max = np.max(embeddings, axis=0)
                flattened_vector = np.concatenate([element_wise_min, element_wise_max], axis=0)
            elif embedding_pooling_method == "svd":
                svd = TruncatedSVD(n_components=2)
                svd_embeddings = svd.fit_transform(embeddings.T)
                flattened_vector = svd_embeddings.flatten()
            elif embedding_pooling_method == "svd_first_four":
                svd = TruncatedSVD(n_components=4)
                svd_embeddings = svd.fit_transform(embeddings.T)
                flattened_vector = svd_embeddings.flatten()
            elif embedding_pooling_method == "ica":
                ica = FastICA(n_components=2)
                ica_embeddings = ica.fit_transform(embeddings.T)
                flattened_vector = ica_embeddings.flatten()
            elif embedding_pooling_method == "factor_analysis":
                fa = FactorAnalysis(n_components=2)
                fa_embeddings = fa.fit_transform(embeddings.T)
                flattened_vector = fa_embeddings.flatten()           
            elif embedding_pooling_method == "gaussian_random_projection":
                grp = GaussianRandomProjection(n_components=2)
                grp_embeddings = grp.fit_transform(embeddings.T)
                flattened_vector = grp_embeddings.flatten()                 
            else:
                raise ValueError(f"Unknown embedding_pooling_method: {embedding_pooling_method}")
            combined_embedding = flattened_vector.tolist()
        else:
            flattened_vector = embeddings.flatten().tolist()
            combined_embedding = embeddings.flatten().tolist()
        embedding_length = len(combined_embedding)
        cnt += 1
        embedding_json = json.dumps(combined_embedding)
        embedding_hash = sha3_256(embedding_json.encode('utf-8')).hexdigest()
        embedding_entry_dict = {'text_index': i, 'text': current_text, 'embedding_pooling_method': embedding_pooling_method, 'number_of_token_embeddings_used': number_of_embeddings, 'embedding_length': embedding_length, 'embedding_hash': embedding_hash, 'embedding': combined_embedding}
        list_of_embedding_entry_dicts.append(embedding_entry_dict)
    end_time = datetime.utcnow()
    total_time = (end_time - start_time).total_seconds()
    logger.info(f"Calculated {len(flattened_vector):,}-dimensional embeddings (relative to the underlying token embedding dimensions of {dimension_of_token_embeddings:,}) for {total_number_of_sentences:,} sentences in a total of {total_time:,.1f} seconds.")
    logger.info(f"That's an average of {1000*total_time/total_number_of_sentences:,.2f} ms per sentence and {total_number_of_sentences/total_time:,.3f} sentences per second (and {total_characters/(1000*total_time):,.4f} total characters per ms) using pooling method '{embedding_pooling_method}'")
    return list_of_embedding_entry_dicts

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