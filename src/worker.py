import os
import sys
import redis
from rq import Worker, Queue, Connection, get_current_job
import logging
import multiprocessing as mp
from functions import add_model_url, download_models
import traceback
import platform

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

def setup_process():
    """Set up process-specific configurations"""
    if platform.system() == 'Darwin':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        if 'numpy' in sys.modules:
            del sys.modules['numpy']

def worker_process(queue_names):
    """Function to run in a separate process for handling worker tasks"""
    redis_conn = redis.Redis(host='localhost', port=6379, db=0)
    with Connection(redis_conn):
        queues = [Queue(name) for name in queue_names]
        worker = Worker(queues, name=f'worker-{os.getpid()}')
        worker.work(with_scheduler=True)

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
            job.meta['progress'] = 100
            job.save_meta()
            return {
                "status": "error",
                "message": "Model URL must point to a .gguf file."
            }
            
        # Add model URL to registry and download
        job.meta['progress'] = 10
        job.save_meta()
        corrected_url = add_model_url(decoded_url)
        
        job.meta['progress'] = 20
        job.save_meta()
        _, download_status = download_models()
        
        # Check download status
        status_dict = {status["url"]: status for status in download_status}
        if corrected_url in status_dict:
            job.meta['progress'] = 100
            job.save_meta()
            return {
                "status": status_dict[corrected_url]["status"],
                "message": status_dict[corrected_url]["message"]
            }
        
        job.meta['progress'] = 100
        job.save_meta()
        return {
            "status": "completed",
            "message": "Model download completed successfully"
        }
        
    except Exception as e:
        error_msg = f"Error in download task: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        job.meta['progress'] = 100
        job.save_meta()
        return {
            "status": "error",
            "message": error_msg
        }

class MultiQueueWorker:
    def __init__(self):
        setup_process()
        
        # Initialize Redis connection
        self.redis_conn = redis.Redis(
            host='localhost', 
            port=6379, 
            db=0,
            decode_responses=True
        )
        
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
