from models import TextEmbedding, SemanticDataTypeEmbedding, SemanticDataType
import socket
import os
import re
import json
import io
import glob
import redis
import sys
import threading
import numpy as np
import faiss
import base64
from typing import Optional
from pathlib import Path
from typing import Any
from db import AsyncSessionLocal
from sqlalchemy import select
from collections import defaultdict
from PIL import Image
import logging
logger = logging.getLogger(__name__)

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )
        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )
        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )
        self.outnull_file.close()
        self.errnull_file.close()
    
def safe_path(base_path, file_name):
    abs_base_path = os.path.abspath(base_path)
    abs_user_path = os.path.abspath(os.path.join(base_path, file_name))
    return abs_user_path.startswith(abs_base_path), abs_user_path

def clean_filename_for_url_func(dirty_filename: str) -> str:
    clean_filename = re.sub(r'[^\w\s]', '', dirty_filename) # Remove special characters and replace spaces with underscores
    clean_filename = clean_filename.replace(' ', '_')
    return clean_filename

def is_redis_running(host='localhost', port=6379):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        return True
    except ConnectionRefusedError:
        return False
    finally:
        s.close()
        
def start_redis_server():
    try:
        result = os.system("sudo service redis-server start")
        if result == 0:
            print("Redis server started successfully.")
        else:
            logger.error(f"Failed to start Redis server, return code: {result}")
            raise Exception("Failed to start Redis server.")
    except Exception as e:
        logger.error(f"Failed to start Redis server: {e}")
        raise

def restart_redis_server():
    try:
        result = os.system("sudo service redis-server stop")
        if result != 0:
            logger.warning(f"Failed to stop Redis server, it might not be running. Return code: {result}")
        result = os.system("sudo service redis-server start")
        if result == 0:
            print("Redis server started successfully.")
        else:
            logger.error(f"Failed to start Redis server, return code: {result}")
            raise Exception("Failed to start Redis server.")
    except Exception as e:
        logger.error(f"Failed to restart Redis server: {e}")
        raise

def configure_redis_optimally(redis_host='localhost', redis_port=6379, maxmemory='1gb'):
    configured_file = 'redis_configured.txt'
    if os.path.exists(configured_file):
        print("Redis has already been configured. Skipping configuration.")
        return
    if not is_redis_running(redis_host, redis_port):
        start_redis_server()
    r = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
    output = []
    def set_config(key, value):
        try:
            response = r.config_set(key, value)
            msg = f"Successfully set {key} to {value}" if response else f"Failed to set {key} to {value}"
            output.append(msg)
            print(msg)
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to set config {key}: {e}")
            raise
    set_config('maxmemory', maxmemory)
    set_config('maxmemory-policy', 'allkeys-lru')
    max_clients = min(os.cpu_count() * 1000, 50000)
    set_config('maxclients', max_clients)
    set_config('timeout', 300)
    set_config('save', '900 1 300 10 60 10000')
    set_config('appendonly', 'yes')
    set_config('appendfsync', 'everysec')
    set_config('stop-writes-on-bgsave-error', 'no')
    output.append("Redis configuration optimized successfully.")
    output.append("Restarting Redis server to apply changes...")
    with open(configured_file, 'w') as f:
        f.write("\n".join(output))
    print("\n".join(output))
    restart_redis_server()
    
def configure_redis_in_background():
    threading.Thread(target=configure_redis_optimally).start()
    
async def build_faiss_indexes(force_rebuild=False):
    global faiss_indexes, associated_texts_by_model_and_pooling_method
    if os.environ.get("FAISS_SETUP_DONE") == "1" and not force_rebuild:
        return faiss_indexes, associated_texts_by_model_and_pooling_method
    faiss_indexes = {}
    associated_texts_by_model_and_pooling_method = defaultdict(lambda: defaultdict(list))  # Create a nested dictionary to store associated texts by model name and pooling method
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TextEmbedding.llm_model_name, TextEmbedding.text, TextEmbedding.embedding_json, TextEmbedding.embedding_pooling_method))
        embeddings_by_model_and_pooling = defaultdict(lambda: defaultdict(list))
        for row in result.fetchall():  # Process regular embeddings
            llm_model_name = row[0]
            embedding_pooling_method = row[3]
            associated_texts_by_model_and_pooling_method[llm_model_name][embedding_pooling_method].append(row[1])  # Store the associated text by model name and pooling method
            embeddings_by_model_and_pooling[llm_model_name][embedding_pooling_method].append((row[1], json.loads(row[2])))
        for llm_model_name, embeddings_by_pooling in embeddings_by_model_and_pooling.items():
            for embedding_pooling_method, embeddings in embeddings_by_pooling.items():
                logger.info(f"Building Faiss index over embeddings for model {llm_model_name} with pooling method {embedding_pooling_method}...")
                embeddings_array = np.array([e[1] for e in embeddings]).astype('float32')
                if embeddings_array.size == 0:
                    logger.error(f"No embeddings were loaded from the database for model {llm_model_name} with pooling method {embedding_pooling_method}, so nothing to build the Faiss index with!")
                    continue
                faiss.normalize_L2(embeddings_array)  # Normalize the vectors for cosine similarity
                faiss_index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Use IndexFlatIP for cosine similarity
                faiss_index.add(embeddings_array)
                faiss_indexes[(llm_model_name, embedding_pooling_method)] = faiss_index  # Store the index by model name and pooling method
    os.environ["FAISS_SETUP_DONE"] = "1"
    return faiss_indexes, associated_texts_by_model_and_pooling_method



def normalize_logprobs(avg_logprob, min_logprob, max_logprob):
    range_logprob = max_logprob - min_logprob
    return (avg_logprob - min_logprob) / range_logprob if range_logprob != 0 else 0.5

def truncate_string(s: str, max_length: int = 100) -> str:
    return s[:max_length]

def remove_pagination_breaks(text: str) -> str:
    text = re.sub(r'-(\n)(?=[a-z])', '', text) # Remove hyphens at the end of lines when the word continues on the next line
    text = re.sub(r'(?<=\w)(?<![.?!-]|\d)\n(?![\nA-Z])', ' ', text) # Replace line breaks that are not preceded by punctuation or list markers and not followed by an uppercase letter or another line break   
    return text

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

def merge_transcript_segments_into_combined_text(segments):
    if not segments:
        return "", [], []
    min_logprob = min(segment['avg_logprob'] for segment in segments)
    max_logprob = max(segment['avg_logprob'] for segment in segments)
    combined_text = ""
    sentence_buffer = ""
    list_of_metadata_dicts = []
    list_of_sentences = []
    char_count = 0
    time_start = None
    time_end = None
    total_logprob = 0.0
    segment_count = 0
    for segment in segments:
        if time_start is None:
            time_start = segment['start']
        time_end = segment['end']
        total_logprob += segment['avg_logprob']
        segment_count += 1
        sentence_buffer += segment['text'] + " "
        sentences = sophisticated_sentence_splitter(sentence_buffer)
        for sentence in sentences:
            combined_text += sentence.strip() + " "
            list_of_sentences.append(sentence.strip())
            char_count += len(sentence.strip()) + 1  # +1 for the space
            avg_logprob = total_logprob / segment_count
            model_confidence_score = normalize_logprobs(avg_logprob, min_logprob, max_logprob)
            metadata = {
                'start_char_count': char_count - len(sentence.strip()) - 1,
                'end_char_count': char_count - 2,
                'time_start': time_start,
                'time_end': time_end,
                'model_confidence_score': model_confidence_score
            }
            list_of_metadata_dicts.append(metadata)
        sentence_buffer = sentences[-1] if len(sentences) % 2 != 0 else ""
    return combined_text, list_of_metadata_dicts, list_of_sentences
    
class JSONAggregator:
    def __init__(self):
        self.completions = []
        self.aggregate_result = None

    @staticmethod
    def weighted_vote(values, weights):
        tally = defaultdict(float)
        for v, w in zip(values, weights):
            tally[v] += w
        return max(tally, key=tally.get)

    @staticmethod
    def flatten_json(json_obj, parent_key='', sep='->'):
        items = {}
        for k, v in json_obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(JSONAggregator.flatten_json(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def get_value_by_path(json_obj, path, sep='->'):
        keys = path.split(sep)
        item = json_obj
        for k in keys:
            item = item[k]
        return item

    @staticmethod
    def set_value_by_path(json_obj, path, value, sep='->'):
        keys = path.split(sep)
        item = json_obj
        for k in keys[:-1]:
            item = item.setdefault(k, {})
        item[keys[-1]] = value

    def calculate_path_weights(self):
        all_paths = []
        for j in self.completions:
            all_paths += list(self.flatten_json(j).keys())
        path_weights = defaultdict(float)
        for path in all_paths:
            path_weights[path] += 1.0
        return path_weights

    def aggregate(self):
        path_weights = self.calculate_path_weights()
        aggregate = {}
        for path, weight in path_weights.items():
            values = [self.get_value_by_path(j, path) for j in self.completions if path in self.flatten_json(j)]
            weights = [weight] * len(values)
            aggregate_value = self.weighted_vote(values, weights)
            self.set_value_by_path(aggregate, path, aggregate_value)
        self.aggregate_result = aggregate

class FakeUploadFile:
    def __init__(self, filename: str, content: Any, content_type: str = 'text/plain'):
        self.filename = filename
        self.content_type = content_type
        self.file = io.BytesIO(content)
    def read(self, size: int = -1) -> bytes:
        return self.file.read(size)
    def seek(self, offset: int, whence: int = 0) -> int:
        return self.file.seek(offset, whence)
    def tell(self) -> int:
        return self.file.tell()
    
def process_image(image_path, max_dimension=1024):
    original_path = Path(image_path)
    processed_image_path = original_path.with_stem(original_path.stem + "_processed").with_suffix(original_path.suffix)
    with Image.open(image_path) as img:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
        img.save(processed_image_path)
    return processed_image_path

def alpha_remover_func(img):
    if img.mode != 'RGBA':
        return img
    canvas = Image.new('RGBA', img.size, (255, 255, 255, 255))
    canvas.paste(img, mask=img)
    return canvas.convert('RGB')

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"    
    
def find_clip_model_path(llm_model_name: str) -> Optional[str]:
    models_dir = os.path.join(BASE_DIRECTORY, 'models')
    base_name = os.path.splitext(os.path.basename(llm_model_name))[0]
    mmproj_model_name = base_name.replace("-f16", "-mmproj-f16").replace("-int4", "-mmproj-f16")
    mmproj_files = glob.glob(os.path.join(models_dir, f"{mmproj_model_name}.gguf"))
    if not mmproj_files:
        logger.error(f"No mmproj file found matching: {mmproj_model_name}")
        return None
    return mmproj_files[0]    

