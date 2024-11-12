from sqlalchemy import Column, String, Float, DateTime, Integer, UniqueConstraint, ForeignKey, LargeBinary
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.declarative import declared_attr
from hashlib import sha3_256
from pydantic import BaseModel, field_validator, ConfigDict
from typing import List, Optional, Union, Dict
from sqlalchemy import event
from datetime import datetime
from typing import Any
from sqlalchemy import Column, String, Float, DateTime, Integer, UniqueConstraint, ForeignKey, LargeBinary, Table, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.declarative import declared_attr
from hashlib import sha3_256
from pydantic import BaseModel, field_validator, ConfigDict
from typing import List, Optional, Union, Dict
from sqlalchemy import event
from datetime import datetime
from typing import Any
from pydantic import Field
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import numpy as np
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    LargeBinary,
    ForeignKey,
    UniqueConstraint,
    event
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from hashlib import sha3_256
import os

from datetime import datetime





Base = declarative_base()
DEFAULT_LLM_NAME = os.getenv("DEFAULT_LLM_NAME", "Meta-Llama-3-8B-Instruct.Q3_K_S")
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("DEFAULT_EMBEDDING_MODEL_NAME", "nomic-embed-text-v1.5.Q6_K")
DEFAULT_MULTI_MODAL_MODEL_NAME = os.getenv("DEFAULT_MULTI_MODAL_MODEL_NAME", "llava-llama-3-8b-v1_1-int4")
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("DEFAULT_MAX_COMPLETION_TOKENS", "100"))
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE = int(os.getenv("DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE", "4"))
DEFAULT_COMPLETION_TEMPERATURE = float(os.getenv("DEFAULT_COMPLETION_TEMPERATURE", "0.7"))
DEFAULT_EMBEDDING_POOLING_METHOD = os.getenv("DEFAULT_EMBEDDING_POOLING_METHOD", "mean")

class SerializerMixin:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    def as_dict(self):
        return {c.key: getattr(self, c.key) for c in self.__table__.columns}
    
class TextEmbedding(Base, SerializerMixin):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    text_hash = Column(String, index=True)
    embedding_pooling_method = Column(String, index=True)
    embedding_hash = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)
    embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime, default=datetime.utcnow)
    response_time = Column(DateTime)
    total_time = Column(Float)

    # Foreign keys
    document_file_hash = Column(String, ForeignKey('document_embeddings.document_file_hash'), nullable=True)
    semantic_data_type_id = Column(Integer, ForeignKey('semantic_data_types.id'), nullable=True)
    embedding_type = Column(String, index=True)  # 'description', 'samples', 'combined' for semantic data types

    # Relationships
    document = relationship(
        "DocumentEmbedding",
        back_populates="embeddings",
        foreign_keys=[document_file_hash]
    )
    semantic_data_type = relationship(
        "SemanticDataType",
        back_populates="text_embeddings",
        foreign_keys=[semantic_data_type_id]
    )

    __table_args__ = (UniqueConstraint('embedding_hash', name='_embedding_hash_uc'),)

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)


class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    document_hash = Column(String, ForeignKey('documents.document_hash'))
    filename = Column(String)
    mimetype = Column(String)
    document_file_hash = Column(String, index=True)
    embedding_pooling_method = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)
    file_data = Column(LargeBinary)  # To store the original file
    sentences = Column(String)
    document_embedding_results_json_compressed_binary = Column(LargeBinary)  # To store the embedding results JSON
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    embeddings = relationship("TextEmbedding", back_populates="document", foreign_keys=[TextEmbedding.document_file_hash])
    __table_args__ = (UniqueConstraint('document_embedding_results_json_compressed_binary', name='_document_embedding_results_json_compressed_binary_uc'),)
    document = relationship("Document", back_populates="document_embeddings", foreign_keys=[document_hash])

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

class Document(Base, SerializerMixin):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    llm_model_name = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)    
    document_hash = Column(String, index=True)
    document_embeddings = relationship("DocumentEmbedding", back_populates="document", foreign_keys=[DocumentEmbedding.document_hash])
    def update_hash(self):  # Concatenate specific attributes from the document_embeddings relationship
        hash_data = "".join([emb.filename + emb.mimetype for emb in self.document_embeddings])
        self.document_hash = sha3_256(hash_data.encode('utf-8')).hexdigest()

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "llm_model_name": self.llm_model_name,
            "corpus_identifier_string": self.corpus_identifier_string,
            "document_hash": self.document_hash,
            "document_embeddings": [emb.to_dict() for emb in self.document_embeddings] if self.document_embeddings else []
        }

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

@event.listens_for(Document.document_embeddings, 'append')
def update_document_hash_on_append(target, value, initiator):
    target.update_hash()
@event.listens_for(Document.document_embeddings, 'remove')
def update_document_hash_on_remove(target, value, initiator):
    target.update_hash()

class SemanticDataTypeBase(BaseModel):
    name: str
    description: str
    examples: List[str]


class SemanticDataType(Base, SerializerMixin):
    __tablename__ = "semantic_data_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    samples = Column(String)  # Comma-separated list of samples
    other_fields = Column(String)  # Placeholder for other fields
    data_type_hash = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    embedding_pooling_method = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    combined_embedding = Column(LargeBinary)  # Store as binary blob

    # Relationships
    text_embeddings = relationship(
        "TextEmbedding",
        back_populates="semantic_data_type",
        cascade="all, delete-orphan"
    )
    semantic_data_type_embeddings = relationship(
        "SemanticDataTypeEmbedding",
        back_populates="semantic_data_type",
        cascade="all, delete-orphan"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    def update_hash(self):
        """Update the data_type_hash based on name, description, and samples."""
        hash_data = self.name + (self.description or '') + (self.samples or '') + (self.other_fields or '')
        self.data_type_hash = sha3_256(hash_data.encode('utf-8')).hexdigest()

# Event listeners to automatically update the hash when the object is inserted or updated
@event.listens_for(SemanticDataType, 'before_insert')
@event.listens_for(SemanticDataType, 'before_update')
def receive_before_insert_update(mapper, connection, target):
    target.update_hash()

class SemanticDataTypeEmbedding(Base, SerializerMixin):
    __tablename__ = "semantic_data_type_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    semantic_data_type_id = Column(Integer, ForeignKey('semantic_data_types.id'), nullable=False)
    embedding_type = Column(String, index=True)  # 'description', 'samples', 'combined'
    embedding_hash = Column(String, index=True, unique=True)
    embedding_pooling_method = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)
    embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime, default=datetime.utcnow)
    response_time = Column(DateTime)
    total_time = Column(Float)

    # Relationships
    semantic_data_type = relationship("SemanticDataType", back_populates="semantic_data_type_embeddings")

    __table_args__ = (UniqueConstraint('embedding_hash', name='_embedding_hash_uc'),)

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

class SemanticDataTypeEmbeddingRequest(BaseModel):
    semantic_data_type: SemanticDataTypeBase
    embed_examples_separately: bool = True
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD
    corpus_identifier_string: str = "codex_1"

class SemanticDataTypeEmbeddingResponse(BaseModel):
    semantic_data_type_id: int
    combined_embedding: List[List[Any]] = Field(..., description="A list of float values representing the embedding of the semantic data type description")
    examples_embeddings: List[List[float]]
    status: str

class SemanticDataTypeResponse(BaseModel):
    id: int
    name: str
    description: str
    llm_model_name: str
    embedding_pooling_method: str
    created_at: datetime

class AllSemanticDataTypesResponse(BaseModel):
    semantic_data_types: List[SemanticDataTypeResponse]

from pydantic import BaseModel

class DocumentEmbeddingPydantic(BaseModel):
    id: int
    filename: str
    mimetype: str
    document_file_hash: str
    embedding_pooling_method: str
    llm_model_name: str
    corpus_identifier_string: str

    class Config:
        from_attributes = True

class DocumentPydantic(BaseModel):
    id: int
    filename: str
    llm_model_name: str
    corpus_identifier_string: str
    document_hash: str
    document_embeddings: List[DocumentEmbeddingPydantic] = []

    class Config:
        from_attributes = True

class AllDocumentsResponse(BaseModel):
    documents: List[DocumentPydantic]

class DocumentContentResponse(BaseModel):
    content: str
    sentences: List[str]

# Request/Response models start here:

class EmbeddingRequest(BaseModel):
    text: str = ""
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD
    corpus_identifier_string: str = ""

class SimilarityRequest(BaseModel):
    text1: str = ""
    text2: str = ""
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD
    similarity_measure: str = "all"
    @field_validator('similarity_measure')
    def validate_similarity_measure(cls, value):
        valid_measures = ["all", "spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_dependency_measure", "hoeffding_d"]
        if value.lower() not in valid_measures:
            raise ValueError(f"Invalid similarity measure. Supported measures are: {', '.join(valid_measures)}")
        return value.lower()
    
class SimilarityResponse(BaseModel):
    text1: str
    text2: str
    similarity_measure: str
    embedding_pooling_method: str
    similarity_score: Union[float, Dict[str, float]]  # Now can be either a float or a dictionary
    embedding1: List[float]
    embedding2: List[float]
class SemanticSearchRequest(BaseModel):
    query_text: str = ""
    number_of_most_similar_strings_to_return: int = 10
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD
    corpus_identifier_string: str = ""
        
class SemanticSearchResponse(BaseModel):
    query_text: str
    corpus_identifier_string: str
    embedding_pooling_method: str
    results: List[dict]  # List of similar strings and their similarity scores using cosine similarity with Faiss (in descending order)

class SemanticSearchResult(BaseModel):
    search_result_text: str
    similarity_to_query_text: float

class AdvancedSemanticSearchRequest(BaseModel):
    query_text: str = ""
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD
    corpus_identifier_string: str = ""
    similarity_filter_percentage: float = 0.01
    number_of_most_similar_strings_to_return: int = 10
    result_sorting_metric: str = "hoeffding_d"
    @field_validator('result_sorting_metric')
    def validate_similarity_measure(cls, value):
        valid_measures = ["spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_dependency_measure", "hoeffding_d"]
        if value.lower() not in valid_measures:
            raise ValueError(f"Invalid similarity measure. Supported measures are: {', '.join(valid_measures)}")
        return value.lower()
    
class AdvancedSemanticSearchResponse(BaseModel):
    query_text: str
    corpus_identifier_string: str
    embedding_pooling_method: str
    results: List[Dict[str, Union[str, float, Dict[str, float]]]]

class EmbeddingResponse(BaseModel):
    id: int
    text: str
    text_hash: str
    embedding_pooling_method: str
    embedding_hash: str
    llm_model_name: str
    corpus_identifier_string: str
    embedding_json: str
    ip_address: Optional[str]
    request_time: datetime
    response_time: datetime
    total_time: float
    document_file_hash: Optional[str]
    embedding: List[float]



class SemanticDataTypeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[List[str]] = None



class UploadedTextCreate(BaseModel):
    content: str

class UploadedTextResponse(BaseModel):
    id: int
    content: str
    content_hash: str
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DetectionResult(BaseModel):
    semantic_data_type_name: str
    text_segment: str
    confidence_score: float
    start_index: int
    end_index: int

class DetectionResponse(BaseModel):
    uploaded_text_id: int
    detections: List[DetectionResult]

class TextDetectionRequest(BaseModel):
    content: str
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD
    corpus_identifier_string: str = ""
    similarity_threshold: float = 0.8

class TextDetectionResponse(BaseModel):
    uploaded_text_id: int
    detections: List[DetectionResult]

    model_config = ConfigDict(from_attributes=True)

class SemanticDataTypes(BaseModel):
    __tablename__ = "semanticdatatypes"
    semantic_data_types: List[SemanticDataType]

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)





class DocumentEmbeddingResponse(BaseModel):
    id: Optional[int]
    document_hash: str
    filename: Optional[str]
    mimetype: Optional[str]
    document_file_hash: str
    embedding_pooling_method: str
    llm_model_name: str
    corpus_identifier_string: str

    model_config = ConfigDict(from_attributes=True)



class SemanticTypeSearchRequest(BaseModel):
    query_text: str = ""
    llm_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_pooling_method: str = DEFAULT_EMBEDDING_POOLING_METHOD
    corpus_identifier_string: str = ""
    similarity_filter_percentage: float = 0.01
    number_of_most_similar_strings_to_return: int = 10
    result_sorting_metric: str = "hoeffding_d"
    @field_validator('result_sorting_metric')
    def validate_similarity_measure(cls, value):
        valid_measures = ["spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_dependency_measure", "hoeffding_d"]
        if value.lower() not in valid_measures:
            raise ValueError(f"Invalid similarity measure. Supported measures are: {', '.join(valid_measures)}")
        return value.lower()
    
class SemanticTypeSearchResponse(BaseModel):
    query_text: str
    corpus_identifier_string: str
    embedding_pooling_method: str
    results: List[Dict[str, Union[str, float, Dict[str, float]]]]

# Add after the other request/response models

class AnnotationRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: str

class AnnotationResponse(BaseModel):
    entities: List[Entity]

# Add after the annotation models

class AnonymizationRequest(BaseModel):
    text: str

class AnonymizationResponse(BaseModel):
    text: str
    entities: List[Entity]  # Reusing the Entity model from annotations

class ScanPatternRequest(BaseModel):
    document_corpus_id: str = Field(..., description="Corpus identifier string containing the document to scan")
    pattern_corpus_id: str = Field(..., description="Corpus identifier string containing the patterns to match against")
    operation: str = Field(..., description="Type of scan operation (e.g. PII, SENSITIVE)")
    llm_model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL_NAME)
    embedding_pooling_method: str = Field(default=DEFAULT_EMBEDDING_POOLING_METHOD) 
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    model_config = ConfigDict(from_attributes=True)

class Match(BaseModel):
    text: str = Field(..., description="Matched text content")
    start: int = Field(..., description="Start position of match")
    end: int = Field(..., description="End position of match") 
    match_type: str = Field(..., description="Type of match found")
    confidence_score: float = Field(..., description="Confidence score of the match")
    semantic_type: str = Field(..., description="Semantic type of the match")
    document_id: int = Field(..., description="ID of the document containing the match")
    filename: str = Field(..., description="Name of the file containing the match")

    model_config = ConfigDict(from_attributes=True)

class ScanPatternResponse(BaseModel):
    operation: str = Field(..., description="Scan operation performed")
    document_corpus_id: str = Field(..., description="Corpus containing the scanned document")
    pattern_corpus_id: str = Field(..., description="Corpus containing the patterns")
    matches: List[Match] = Field(default_factory=list)
    total_matches: int = Field(..., description="Total number of matches found")
    total_sentences_scanned: int = Field(..., description="Number of sentences scanned")
    processing_time: float = Field(..., description="Time taken to process in seconds")

    model_config = ConfigDict(from_attributes=True)
