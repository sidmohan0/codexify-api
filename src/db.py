from models import Base, TextEmbedding, DocumentEmbedding, Document
import traceback
import asyncio
import random
from sqlalchemy import select, update, UniqueConstraint, exists
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json
import os
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)
db_writer = None


SQLITE_PRAGMA_JOURNAL_MODE = os.getenv("SQLITE_PRAGMA_JOURNAL_MODE", "WAL")
SQLITE_PRAGMA_SYNCHRONOUS = os.getenv("SQLITE_PRAGMA_SYNCHRONOUS", "NORMAL")
SQLITE_PRAGMA_CACHE_SIZE = os.getenv("SQLITE_PRAGMA_CACHE_SIZE", "-1048576")
SQLITE_PRAGMA_BUSY_TIMEOUT = os.getenv("SQLITE_PRAGMA_BUSY_TIMEOUT", "2000")
SQLITE_PRAGMA_WAL_AUTOCHECKPOINT = os.getenv("SQLITE_PRAGMA_WAL_AUTOCHECKPOINT", "100")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///codexify.sqlite")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
DB_WRITE_BATCH_SIZE = int(os.getenv("DB_WRITE_BATCH_SIZE", 25))
RETRY_DELAY_BASE_SECONDS = int(os.getenv("RETRY_DELAY_BASE_SECONDS", 1))
JITTER_FACTOR = float(os.getenv("JITTER_FACTOR", 0.1))
TIME_IN_DAYS_BEFORE_RECORDS_ARE_PURGED = int(os.getenv("TIME_IN_DAYS_BEFORE_RECORDS_ARE_PURGED", 2))

engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)



async def consolidate_wal_data() -> Optional[Tuple[int, int, int]]:
    """
    Consolidate Write-Ahead Log (WAL) data by executing a full checkpoint.

    Preconditions:
    - The database engine must be initialized and connected.
    - The database must be using WAL mode.

    Postconditions:
    - If successful, returns a tuple containing checkpoint information.
    - If an error occurs, logs the error and returns None.

    Returns:
    - Optional[Tuple[int, int, int]]: A tuple containing (busy, log, checkpointed) 
      if successful, or None if an error occurred.

    Raises:
    - No exceptions are raised; errors are logged and None is returned.
    """
    consolidate_command = "PRAGMA wal_checkpoint(FULL);"
    try:
        async with engine.begin() as conn:
            result = await conn.execute(sql_text(consolidate_command))
            result_fetch = result.fetchone()
            return result_fetch
    except Exception as e:
        logger.error(f"Error during WAL consolidation: {e}")
        return None


class DatabaseWriter:
    """
    A class for managing database write operations asynchronously.

    This class handles the writing of various database models (TextEmbedding, DocumentEmbedding, Document, AudioTranscript)
    to the database, ensuring uniqueness and handling potential conflicts.

    Attributes:
        queue (asyncio.Queue): A queue to store write operations.
        processing_hashes (set): A set to keep track of hashes currently being processed.

    Methods:
        __init__(self, queue)
        _get_hash_from_operation(self, operation)
        initialize_processing_hashes(self, chunk_size=1000)
        _record_exists(self, session, operation)
        dedicated_db_writer(self)
        _update_existing_record(self, session, operation)
        _handle_integrity_error(self, e, write_operation, session)
        enqueue_write(self, write_operations)
    """

    def __init__(self, queue):
        """
        Initialize the DatabaseWriter.

        Preconditions:
            - queue is a valid asyncio.Queue object.

        Postconditions:
            - self.queue is set to the provided queue.
            - self.processing_hashes is initialized as an empty set.
        """
        self.queue = queue
        self.processing_hashes = set()

    def _get_hash_from_operation(self, operation):
        """
        Extract a unique hash from the given operation.

        Preconditions:
            - operation is an instance of TextEmbedding, DocumentEmbedding, Document, or AudioTranscript.

        Postconditions:
            - Returns a string hash if the operation is of a recognized type.
            - Returns None if the operation is not of a recognized type.
        """
        if isinstance(operation, TextEmbedding):
            return f"{operation.embedding_hash}"
        elif isinstance(operation, DocumentEmbedding):
            return f"{operation.document_embedding_results_json_compressed_binary}"
        elif isinstance(operation, Document):
            return operation.document_hash
        elif isinstance(operation, AudioTranscript):
            return operation.audio_file_hash
        return None

    async def initialize_processing_hashes(self, chunk_size=1000):
        """
        Initialize the set of processing hashes from the database.

        Preconditions:
            - The database connection is established and functional.
            - chunk_size is a positive integer.

        Postconditions:
            - self.processing_hashes is populated with existing hashes from the database.
            - Logs the initialization process, including time taken and set size.
        """
        start_time = datetime.utcnow()
        async with AsyncSessionLocal() as session:
            queries = [
                (select(TextEmbedding.embedding_hash), TextEmbedding),
                (select(DocumentEmbedding.document_embedding_results_json_compressed_binary), DocumentEmbedding),
                (select(Document.document_hash), Document),
                
            ]
            for query, model_class in queries:
                offset = 0
                while True:
                    result = await session.execute(query.limit(chunk_size).offset(offset))
                    rows = result.fetchall()
                    if not rows:
                        break
                    for row in rows:
                        if model_class == TextEmbedding:
                            hash_with_model = row[0]
                        elif model_class == DocumentEmbedding:
                            hash_with_model = row[0]
                        elif model_class == Document:
                            hash_with_model = row[0]
                        elif model_class == AudioTranscript:
                            hash_with_model = row[0]
                        self.processing_hashes.add(hash_with_model)
                    offset += chunk_size
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        if len(self.processing_hashes) > 0:
            logger.info(f"Finished initializing set of input hash/llm_model_name combinations that are either currently being processed or have already been processed. Set size: {len(self.processing_hashes)}; Took {total_time} seconds, for an average of {total_time / len(self.processing_hashes)} seconds per hash.")

    async def _record_exists(self, session, operation):
        """
        Check if a record already exists in the database.

        Preconditions:
            - session is a valid database session.
            - operation is an instance of TextEmbedding, DocumentEmbedding, Document, or AudioTranscript.

        Postconditions:
            - Returns a SQLAlchemy exists() clause result if the operation is of a recognized type.
            - Returns None if the operation is not of a recognized type.
        """
        model_class = type(operation)
        if model_class == TextEmbedding:
            return await session.execute(select(exists().where(TextEmbedding.embedding_hash == operation.embedding_hash)))
        elif model_class == DocumentEmbedding:
            return await session.execute(select(exists().where(DocumentEmbedding.document_embedding_results_json_compressed_binary == operation.document_embedding_results_json_compressed_binary)))
        elif model_class == Document:
            return await session.execute(select(exists().where(Document.document_hash == operation.document_hash)))
        
        return None

    async def dedicated_db_writer(self):
        """
        Continuously process write operations from the queue.

        Preconditions:
            - The database connection is established and functional.
            - self.queue is properly initialized and accessible.

        Postconditions:
            - Processes write operations from the queue indefinitely.
            - Handles database writes, updates, and error cases.
            - Removes processed hashes from self.processing_hashes.
        """
        while True:
            write_operations_batch = await self.queue.get()
            async with AsyncSessionLocal() as session:
                filtered_operations = []
                try:
                    if write_operations_batch:
                        for write_operation in write_operations_batch:
                            existing_record = await self._record_exists(session, write_operation)
                            if not existing_record.scalar():
                                filtered_operations.append(write_operation)
                                hash_value = self._get_hash_from_operation(write_operation)
                                if hash_value:
                                    self.processing_hashes.add(hash_value)
                            else:
                                await self._update_existing_record(session, write_operation)
                        if filtered_operations:
                            await consolidate_wal_data()  # Consolidate WAL before performing writes
                            session.add_all(filtered_operations)
                            await session.flush()  # Flush to get the IDs
                            await session.commit()
                            for write_operation in filtered_operations:
                                hash_to_remove = self._get_hash_from_operation(write_operation)
                                if hash_to_remove is not None and hash_to_remove in self.processing_hashes:
                                    self.processing_hashes.remove(hash_to_remove)
                except IntegrityError as e:
                    await self._handle_integrity_error(e, write_operation, session)
                except SQLAlchemyError as e:
                    logger.error(f"Database error: {e}")
                    await session.rollback()
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"Unexpected error: {e}\n{tb}")
                    await session.rollback()
                self.queue.task_done()

    async def _update_existing_record(self, session, operation):
        """
        Update an existing record in the database.

        Preconditions:
            - session is a valid database session.
            - operation is an instance of a SQLAlchemy model with valid attributes.

        Postconditions:
            - The existing record is updated with the new values from the operation.
            - The changes are committed to the database.
        """
        model_class = type(operation)
        primary_keys = [key.name for key in model_class.__table__.primary_key]
        unique_constraints = [c for c in model_class.__table__.constraints if isinstance(c, UniqueConstraint)]
        conditions = []
        for constraint in unique_constraints:
            if set(constraint.columns.keys()).issubset(set(operation.__dict__.keys())):
                for col in constraint.columns.keys():
                    conditions.append(getattr(model_class, col) == getattr(operation, col))
                break
        if not conditions:
            for pk in primary_keys:
                conditions.append(getattr(model_class, pk) == getattr(operation, pk))
        values = {col: getattr(operation, col) for col in operation.__dict__.keys() if col in model_class.__table__.columns.keys()}
        stmt = update(model_class).where(*conditions).values(**values)
        await session.execute(stmt)
        await session.commit()

    async def _handle_integrity_error(self, e, write_operation, session):
        """
        Handle IntegrityError exceptions during database operations.

        Preconditions:
            - e is an IntegrityError exception.
            - write_operation is the operation that caused the error.
            - session is a valid database session.

        Postconditions:
            - If the error is due to a unique constraint violation, attempts to update the existing record.
            - If the error is not due to a unique constraint violation, re-raises the exception.
        """
        unique_constraint_msg = {
            TextEmbedding: "embeddings.embedding_hash",
            DocumentEmbedding: "document_embeddings.document_embedding_results_json_compressed_binary",
            Document: "documents.document_hash",
            
        }.get(type(write_operation))
        if unique_constraint_msg and unique_constraint_msg in str(e):
            logger.warning(f"Embedding already exists in the database for given input: {e}")
            await self._update_existing_record(session, write_operation)
        else:
            raise        

    async def enqueue_write(self, write_operations):
        """
        Enqueue write operations for processing.

        Preconditions:
            - write_operations is a list of database model instances.

        Postconditions:
            - Filters out operations with hashes already in self.processing_hashes.
            - Adds new hashes to self.processing_hashes.
            - Enqueues filtered write operations to self.queue.
        """
        write_operations = [op for op in write_operations if self._get_hash_from_operation(op) not in self.processing_hashes]
        if not write_operations:
            return
        for op in write_operations:
            hash_value = self._get_hash_from_operation(op)
            if hash_value:
                self.processing_hashes.add(hash_value)
        await self.queue.put(write_operations)

async def execute_with_retry(func, *args, **kwargs):
    """
    Execute a database function with retry logic for handling database locks.

    Preconditions:
    - func is a callable asynchronous function.
    - MAX_RETRIES, RETRY_DELAY_BASE_SECONDS, and JITTER_FACTOR are defined constants.
    - logger is a configured logging object.

    Postconditions:
    - Returns the result of func if successful.
    - Raises OperationalError if the database remains locked after MAX_RETRIES attempts.

    Args:
    - func: The asynchronous function to execute.
    - *args: Positional arguments to pass to func.
    - **kwargs: Keyword arguments to pass to func.

    Returns:
    - The result of the successful execution of func.

    Raises:
    - OperationalError: If the database remains locked after MAX_RETRIES attempts.
    - Any other exception raised by func that is not a database lock error.
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return await func(*args, **kwargs)
        except OperationalError as e:
            if 'database is locked' in str(e):
                retries += 1
                sleep_time = RETRY_DELAY_BASE_SECONDS * (2 ** retries) + (random.random() * JITTER_FACTOR)
                logger.warning(f"Database is locked. Retrying ({retries}/{MAX_RETRIES})... Waiting for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
            else:
                raise
    raise OperationalError("Database is locked after multiple retries")



async def initialize_db(use_verbose: int = 0) -> None:
    """
    Initialize the database, create tables, and set SQLite PRAGMAs.

    Preconditions:
    - The global 'engine' variable is properly initialized and connected to the database.
    - The global 'logger' is properly configured for logging.
    - The following global variables are defined with appropriate values:
        SQLITE_PRAGMA_JOURNAL_MODE, SQLITE_PRAGMA_SYNCHRONOUS, SQLITE_PRAGMA_CACHE_SIZE,
        SQLITE_PRAGMA_BUSY_TIMEOUT, SQLITE_PRAGMA_WAL_AUTOCHECKPOINT
    - The 'Base' class from SQLAlchemy is properly defined with all necessary models.

    Postconditions:
    - All specified SQLite PRAGMAs are set.
    - All tables defined in 'Base.metadata' are created in the database if they don't already exist.
    - If use_verbose is True, detailed information about each PRAGMA is logged.
    - Any exceptions during table creation are silently caught and ignored.

    Args:
    use_verbose (int): If non-zero, enables verbose logging of PRAGMA operations. Defaults to 0.

    Returns:
    None

    Raises:
    No exceptions are explicitly raised by this function.
    """
    logger.info("Initializing database, creating tables, and setting SQLite PRAGMAs...")
    sqlite_pragma_settings = [
        f"PRAGMA journal_mode = {SQLITE_PRAGMA_JOURNAL_MODE};", 
        f"PRAGMA synchronous = {SQLITE_PRAGMA_SYNCHRONOUS};", 
        f"PRAGMA cache_size = {SQLITE_PRAGMA_CACHE_SIZE};", 
        f"PRAGMA busy_timeout = {SQLITE_PRAGMA_BUSY_TIMEOUT};", 
        f"PRAGMA wal_autocheckpoint = {SQLITE_PRAGMA_WAL_AUTOCHECKPOINT};"
    ]
    sqlite_pragma_settings_explanation = [
        "Set SQLite to use Write-Ahead Logging (WAL) mode (from default DELETE mode) so that reads and writes can occur simultaneously",
        "Set synchronous mode to NORMAL (from FULL) so that writes are not blocked by reads",
        "Set cache size to 1GB (from default 2MB) so that more data can be cached in memory and not read from disk; to make this 256MB, set it to -262144 instead",
        "Increase the busy timeout to 2 seconds so that the database waits",
        "Set the WAL autocheckpoint to 100 (from default 1000) so that the WAL file is checkpointed more frequently"
    ]
    assert len(sqlite_pragma_settings) == len(sqlite_pragma_settings_explanation) 
    async with engine.begin() as conn:
        for pragma_string in sqlite_pragma_settings:
            await conn.execute(sql_text(pragma_string))
            if use_verbose:
                logger.info(f"Executed SQLite PRAGMA: {pragma_string}")
                logger.info(f"Justification: {sqlite_pragma_settings_explanation[sqlite_pragma_settings.index(pragma_string)]}")
        try:
            await conn.run_sync(Base.metadata.create_all)  # Create tables if they don't exist
        except Exception as e:  # noqa: F841
            pass
    logger.info("Database initialization completed.")



    
def get_db_writer() -> DatabaseWriter:
    """
    Retrieves the existing DatabaseWriter instance.

    Preconditions:
    - The global 'db_writer' variable is initialized with a valid DatabaseWriter instance.

    Postconditions:
    - Returns the existing DatabaseWriter instance without modification.

    Returns:
    DatabaseWriter: The existing DatabaseWriter instance.

    Raises:
    - No exceptions are explicitly raised by this function.
    """
    return db_writer  # Return the existing DatabaseWriter instance

def delete_expired_rows(session_factory):
    """
    Creates and returns an asynchronous function to delete expired rows from specified models.

    Preconditions:
    - 'session_factory' is a valid async session factory for database operations.
    - The global 'TIME_IN_DAYS_BEFORE_RECORDS_ARE_PURGED' is defined and is a positive integer.
    - The models TextEmbedding, DocumentEmbedding, Document, and AudioTranscript are defined
      and have a 'created_at' attribute.

    Postconditions:
    - Returns an async function that, when called:
        - Deletes all rows from specified models that are older than the expiration time.
        - Commits the changes to the database.

    Args:
    session_factory (Callable): An async session factory for creating database sessions.

    Returns:
    Callable: An async function that performs the deletion of expired rows.

    Raises:
    - No exceptions are explicitly raised by this function, but the returned async function
      may raise exceptions related to database operations.
    """
    async def async_delete_expired_rows():
        async with session_factory() as session:
            expiration_time = datetime.utcnow() - timedelta(days=TIME_IN_DAYS_BEFORE_RECORDS_ARE_PURGED)
            models = [TextEmbedding, DocumentEmbedding, Document, AudioTranscript]
            for model in models:
                expired_rows = await session.execute(
                    select(model).where(model.created_at < expiration_time)
                )
                expired_rows = expired_rows.scalars().all()
                for row in expired_rows:
                    await session.delete(row)
            await session.commit()
    return async_delete_expired_rows


# At the end of the file, add:
def create_tables(engine):
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    engine = create_async_engine('sqlite+aiosqlite:///codexify.sqlite')  # Replace with your actual database URL
    create_tables(engine)
    print("Database tables created successfully.")
