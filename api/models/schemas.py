"""
Modelos Pydantic para la API
"""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ProcessingJob(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    input_file: str
    output_file: Optional[str] = None
    error: Optional[str] = None
    # Campos para generación de datos sintéticos
    generated_count: Optional[int] = None
    source_file: Optional[str] = None
    generation_type: Optional[str] = None

class BatchProcessRequest(BaseModel):
    bucket_name: str
    file_patterns: List[str] = ["*.png", "*.jpg", "*.jpeg"]
    output_bucket: str = "document-restored"

class TrainingRequest(BaseModel):
    clean_bucket: str = "document-clean"
    degraded_bucket: str = "document-degraded"
    epochs: int = 10
    batch_size: int = 2

class SyntheticDataRequest(BaseModel):
    source_bucket: str
    source_file: str
    target_count: int = 10
    generation_type: str = "degradation"  # degradation, variation
    output_bucket: str = "document-training"

class ImageQualityAnalysis(BaseModel):
    sharpness: float
    gradient: float
    contrast: float
    noise: float
    quality_score: float

class ClassificationResult(BaseModel):
    classification: str
    confidence: float
    metrics: ImageQualityAnalysis
    filename: str
    input_method: Optional[str] = None

class UploadResult(BaseModel):
    status: str
    bucket: str
    filename: str
    original_filename: Optional[str] = None
    size: int
    content_type: Optional[str] = None
    url: Optional[str] = None

class JobStats(BaseModel):
    total: int
    pending: int
    processing: int
    completed: int
    failed: int

class DatasetStats(BaseModel):
    bucket: str
    count: int
    total_size_mb: Optional[float] = None
    files: Optional[List[str]] = None
    error: Optional[str] = None
