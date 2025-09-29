"""
Week 7: FastAPI Backend for NewsAI
Production-ready API for Arabic news classification and editorial assistance
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag_engine.rag_classifier import RAGPublishabilityClassifier
from src.rag_engine.enhanced_editorial import EnhancedEditorialSystem
from config.settings import *

# Initialize FastAPI app
app = FastAPI(
    title="NewsAI API",
    description="RAG-powered Arabic news classification and editorial assistance for Wakalat Al Anba2",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialize once)
classifier = None
editorial_system = None

# Request/Response models
class ArticleInput(BaseModel):
    text: str = Field(..., description="Arabic article text to classify")
    article_id: Optional[str] = Field(None, description="Optional article identifier")

class ClassificationResult(BaseModel):
    article_id: str
    decision: str = Field(..., description="موافق or مرفوض")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    processing_time: float
    timestamp: datetime
    similar_examples_count: int

class EditorialInput(BaseModel):
    text: str = Field(..., description="Arabic article text to edit")
    article_type: str = Field("general", description="Type of article: general, economic, political")
    article_id: Optional[str] = Field(None, description="Optional article identifier")

class EditorialResult(BaseModel):
    article_id: str
    original_text: str
    edited_text: str
    changes_applied: List[str]
    style_template_used: str
    quality_score: float
    processing_time: float
    timestamp: datetime

class BatchProcessingJob(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    total_articles: int
    processed_articles: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[List[Dict]] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    system_info: Dict[str, Any]

# In-memory job storage (use Redis in production)
active_jobs: Dict[str, BatchProcessingJob] = {}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize AI systems on startup"""
    global classifier, editorial_system
    
    print("Starting NewsAI API...")
    print("Initializing RAG classifier...")
    classifier = RAGPublishabilityClassifier()
    
    print("Initializing editorial system...")
    editorial_system = EnhancedEditorialSystem()
    editorial_system.fix_vector_search()
    
    print("NewsAI API ready!")

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health and system status"""
    
    import psutil
    
    system_info = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "classifier_loaded": classifier is not None,
        "editorial_system_loaded": editorial_system is not None
    }
    
    return HealthCheck(
        status="healthy" if all([classifier, editorial_system]) else "degraded",
        timestamp=datetime.now(),
        system_info=system_info
    )

# Classification endpoints
@app.post("/classify", response_model=ClassificationResult)
async def classify_article(article: ArticleInput):
    """Classify a single Arabic article for publishability"""
    
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    article_id = article.article_id or str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Classify article
        result = classifier.classify_article(article.text)
        processing_time = time.time() - start_time
        
        # Get similar examples count (if available)
        similar_count = getattr(result, 'similar_examples_count', 0)
        
        return ClassificationResult(
            article_id=article_id,
            decision=result['decision'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            processing_time=processing_time,
            timestamp=datetime.now(),
            similar_examples_count=similar_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify/batch")
async def create_batch_classification_job(articles: List[ArticleInput], background_tasks: BackgroundTasks):
    """Create a batch classification job"""
    
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    job_id = str(uuid.uuid4())
    
    # Create job record
    job = BatchProcessingJob(
        job_id=job_id,
        status="pending",
        total_articles=len(articles),
        processed_articles=0,
        created_at=datetime.now()
    )
    
    active_jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_batch_classification, job_id, articles)
    
    return {"job_id": job_id, "status": "created", "total_articles": len(articles)}

async def process_batch_classification(job_id: str, articles: List[ArticleInput]):
    """Process batch classification in background"""
    
    job = active_jobs[job_id]
    job.status = "processing"
    
    results = []
    
    try:
        for i, article in enumerate(articles):
            article_id = article.article_id or f"{job_id}_{i}"
            start_time = time.time()
            
            try:
                result = classifier.classify_article(article.text)
                processing_time = time.time() - start_time
                
                classification_result = {
                    "article_id": article_id,
                    "decision": result['decision'],
                    "confidence": result['confidence'],
                    "reasoning": result['reasoning'],
                    "processing_time": processing_time,
                    "status": "success"
                }
                
            except Exception as e:
                classification_result = {
                    "article_id": article_id,
                    "decision": "مرفوض",
                    "confidence": 0.0,
                    "reasoning": f"خطأ في المعالجة: {str(e)}",
                    "processing_time": 0.0,
                    "status": "error"
                }
            
            results.append(classification_result)
            job.processed_articles = i + 1
        
        job.status = "completed"
        job.results = results
        job.completed_at = datetime.now()
        
    except Exception as e:
        job.status = "failed"
        job.results = [{"error": str(e)}]
        job.completed_at = datetime.now()

@app.get("/classify/batch/{job_id}")
async def get_batch_job_status(job_id: str):
    """Get batch job status and results"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

# Editorial endpoints
@app.post("/edit", response_model=EditorialResult)
async def edit_article(article: EditorialInput):
    """Edit an approved Arabic article using RAG"""
    
    if not editorial_system:
        raise HTTPException(status_code=503, detail="Editorial system not initialized")
    
    article_id = article.article_id or str(uuid.uuid4())
    
    try:
        # Edit article
        result = editorial_system.enhanced_article_editing(
            article_text=article.text,
            article_type=article.article_type
        )
        
        return EditorialResult(
            article_id=article_id,
            original_text=result['original_text'],
            edited_text=result['edited_text'],
            changes_applied=result['changes_applied'],
            style_template_used=result['style_template_used'],
            quality_score=result['quality_score'],
            processing_time=result['processing_time'],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Editorial error: {str(e)}")

# Complete workflow endpoint
@app.post("/process")
async def process_article_complete(article: ArticleInput):
    """Complete article processing: classification + editing if approved"""
    
    if not all([classifier, editorial_system]):
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
    article_id = article.article_id or str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Step 1: Classification
        classification = classifier.classify_article(article.text)
        
        result = {
            "article_id": article_id,
            "classification": {
                "decision": classification['decision'],
                "confidence": classification['confidence'],
                "reasoning": classification['reasoning']
            },
            "editorial": None,
            "total_processing_time": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 2: Editorial (only if approved)
        if classification['decision'] == 'موافق':
            editorial_result = editorial_system.enhanced_article_editing(article.text)
            
            result["editorial"] = {
                "edited_text": editorial_result['edited_text'],
                "changes_applied": editorial_result['changes_applied'],
                "quality_score": editorial_result['quality_score']
            }
        
        result["total_processing_time"] = time.time() - start_time
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Analytics endpoints
@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get system analytics summary"""
    
    # Calculate stats from active jobs
    total_jobs = len(active_jobs)
    completed_jobs = sum(1 for job in active_jobs.values() if job.status == "completed")
    total_articles = sum(job.total_articles for job in active_jobs.values())
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "total_articles_processed": total_articles,
        "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
        "system_uptime_hours": (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).seconds / 3600
    }

# Vector database endpoints
@app.get("/vector-db/stats")
async def get_vector_db_stats():
    """Get vector database statistics"""
    
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        stats = classifier.vector_store.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

@app.post("/vector-db/search")
async def search_similar_articles(query: dict):
    """Search for similar articles in vector database"""
    
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        query_text = query.get("text", "")
        collection = query.get("collection", "approved_articles")
        top_k = query.get("top_k", 5)
        
        # Use the fixed search method
        if hasattr(classifier.vector_store, 'search_similar_articles'):
            results = classifier.vector_store.search_similar_articles(
                query_text=query_text,
                collection_name=collection,
                top_k=top_k
            )
        else:
            # Fallback method
            results = []
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": str(exc)}
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Something went wrong"}
    )

# Development test endpoint
@app.get("/test")
async def test_endpoint():
    """Test endpoint for development"""
    
    test_article = "الرياض في 28 سبتمبر /واس/ أكد وزير الخارجية أهمية التعاون الدولي."
    
    try:
        # Test classification
        if classifier:
            classification = classifier.classify_article(test_article)
            classification_status = "working"
        else:
            classification = None
            classification_status = "not initialized"
        
        # Test editorial
        if editorial_system:
            editorial = editorial_system.enhanced_article_editing(test_article)
            editorial_status = "working"
        else:
            editorial = None
            editorial_status = "not initialized"
        
        return {
            "message": "NewsAI API Test",
            "classifier_status": classification_status,
            "editorial_status": editorial_status,
            "sample_classification": classification['decision'] if classification else None,
            "sample_confidence": classification['confidence'] if classification else None
        }
        
    except Exception as e:
        return {
            "message": "NewsAI API Test",
            "error": str(e),
            "classifier_status": "error",
            "editorial_status": "error"
        }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting NewsAI API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )