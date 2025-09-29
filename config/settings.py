"""
NewsAI Configuration Settings - Complete Version with Ollama Support
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ANALYTICS_DIR = DATA_DIR / "analytics"

# Raw data files
MAIN_DATASET = DATA_DIR / "akhbar_sharq_awsat_fixed.xlsx"
EDITORIAL_DATASET = DATA_DIR / "akhbar_sharq_awsat_qbl_b3d_fixed.xlsx"

# LLM Configuration - Choose your provider
LLM_PROVIDER = "ollama"  # Options: "ollama", "openai", "anthropic"
LLM_MODEL = "aya:8b"     # For Ollama: "aya:8b", "llama3.1:8b", "mistral:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Only needed if using OpenAI

# Embedding model configuration
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_ALTERNATIVES = [
    "sentence-transformers/distiluse-base-multilingual-cased",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-large"
]

# Processing parameters
CHUNK_SIZE = 512
OVERLAP_SIZE = 50
BATCH_SIZE = 32
MIN_ARTICLE_LENGTH = 100
MIN_WORD_COUNT = 20

# Quality thresholds (based on Week 1 analysis results)
QUALITY_THRESHOLD = 0.5          # For RAG inclusion
PUBLISHABILITY_THRESHOLD = 0.7   # For approval recommendation  
LOW_QUALITY_THRESHOLD = 0.3      # Below this = automatic rejection
HIGH_QUALITY_THRESHOLD = 0.8     # High confidence approval

# Vector database configuration
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
VECTOR_DB_PROVIDER = "chroma"    # Options: "chroma", "weaviate", "pinecone"

VECTOR_DB_COLLECTIONS = {
    'approved_articles': 'approved_news_articles',
    'rejected_articles': 'rejected_news_articles', 
    'editorial_examples': 'editorial_transformations',
    'style_guidelines': 'agency_style_rules'
}

# Arabic text processing settings
ARABIC_PROCESSING = {
    'remove_diacritics': True,
    'normalize_characters': True,
    'extract_metadata': True,
    'clean_whitespace': True,
    'handle_rtl': True
}

# Dataset statistics (from Week 1 analysis)
DATASET_STATS = {
    'total_articles': 16495,
    'approved_articles': 630,
    'rejected_articles': 15865,
    'approval_rate': 0.0382,        # 3.82%
    'avg_approved_length': 1548,    # chars (from your analysis)
    'avg_rejected_length': 1598,    # chars (from your analysis)
    'editorial_examples': 1042,
    'optimal_length_range': (1000, 6000),  # Sweet spot from your analysis
    'avg_quality_score': 0.784     # From preprocessing tests
}

# Embedding settings (Week 2)
EMBEDDING_SETTINGS = {
    'model_name': EMBEDDING_MODEL,
    'max_length': 512,
    'batch_size': 64,
    'device': 'cpu',  # Change to 'cuda' if GPU available
    'normalize_embeddings': True,
    'show_progress': True
}

# RAG settings (Week 3-4)
RAG_SETTINGS = {
    'retrieval_k': 5,               # Number of similar examples to retrieve
    'similarity_threshold': 0.7,    # Minimum similarity for relevance
    'context_window': 2048,         # Max tokens for LLM context
    'temperature': 0.1,             # Low temperature for consistent results
    'max_tokens': 1024,            # Max response length
    'top_p': 0.9,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

# Stage 1: Publishability Classifier Settings
STAGE1_SETTINGS = {
    'confidence_threshold': 0.8,    # High confidence decisions
    'human_review_threshold': 0.6,  # Send to human if below this
    'auto_reject_threshold': 0.3,   # Auto-reject if below this
    'features_to_use': [
        'quality_score',
        'text_length', 
        'has_attribution',
        'has_agency_tag',
        'has_quotes',
        'metadata_completeness'
    ]
}

# Stage 2: Editorial Assistant Settings  
STAGE2_SETTINGS = {
    'min_editing_confidence': 0.7,
    'preserve_meaning': True,
    'apply_agency_style': True,
    'max_length_change_pct': 30,    # Don't change length by more than 30%
    'required_elements': [
        'agency_tag',
        'proper_attribution', 
        'clear_structure'
    ]
}

# API settings (Week 7-8)
API_SETTINGS = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': True,
    'auto_reload': True,
    'cors_origins': ["*"],
    'max_upload_size': 10 * 1024 * 1024,  # 10MB
    'rate_limit': "100/minute"
}

# Logging configuration
LOGGING_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / 'logs' / 'newsai.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Performance monitoring
MONITORING_SETTINGS = {
    'track_processing_time': True,
    'track_quality_scores': True,
    'track_approval_accuracy': True,
    'save_metrics_interval': 3600,  # Save every hour
    'metrics_file': ANALYTICS_DIR / 'performance_metrics.json'
}

# Development vs Production settings
ENVIRONMENT = os.getenv('NEWSAI_ENV', 'development')

if ENVIRONMENT == 'production':
    API_SETTINGS['debug'] = False
    LOGGING_SETTINGS['level'] = 'WARNING'
    RAG_SETTINGS['temperature'] = 0.05  # More conservative in production
    
# Database connection settings (for Week 8)
DATABASE_SETTINGS = {
    'type': 'postgresql',  # Options: 'sqlite', 'postgresql', 'mysql'
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432),
    'database': os.getenv('DB_NAME', 'newsai'),
    'username': os.getenv('DB_USER', 'newsai'),
    'password': os.getenv('DB_PASSWORD', ''),
    'pool_size': 10,
    'max_overflow': 20
}

# Cache settings
CACHE_SETTINGS = {
    'provider': 'redis',  # Options: 'redis', 'memory'
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': os.getenv('REDIS_PORT', 6379),
    'db': 0,
    'ttl': 3600,  # 1 hour default TTL
    'max_connections': 10
}

# Batch processing settings
BATCH_SETTINGS = {
    'max_batch_size': 1000,
    'processing_timeout': 300,  # 5 minutes
    'retry_attempts': 3,
    'parallel_workers': 4,
    'queue_backend': 'redis'
}

# Security settings
SECURITY_SETTINGS = {
    'api_key_required': False,  # Set to True in production
    'allowed_ips': [],  # Empty list allows all IPs
    'rate_limiting': True,
    'input_validation': True,
    'sanitize_output': True
}

# File upload settings
UPLOAD_SETTINGS = {
    'allowed_extensions': ['.txt', '.docx', '.pdf', '.xlsx'],
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'upload_dir': PROJECT_ROOT / 'uploads',
    'virus_scan': False  # Set to True if antivirus available
}

# Ensure directories exist
directories_to_create = [
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,
    ANALYTICS_DIR,
    PROJECT_ROOT / 'logs',
    UPLOAD_SETTINGS['upload_dir']
]

for directory in directories_to_create:
    directory.mkdir(exist_ok=True, parents=True)

# Validation
def validate_configuration():
    """Validate configuration settings"""
    errors = []
    
    # Check required files exist
    if not MAIN_DATASET.exists():
        errors.append(f"Main dataset not found: {MAIN_DATASET}")
    
    if not EDITORIAL_DATASET.exists():
        errors.append(f"Editorial dataset not found: {EDITORIAL_DATASET}")
    
    # Check LLM configuration
    if LLM_PROVIDER == "ollama" and not OLLAMA_BASE_URL:
        errors.append("Ollama base URL not configured")
    
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OpenAI API key not configured")
    
    # Check thresholds make sense
    if QUALITY_THRESHOLD >= PUBLISHABILITY_THRESHOLD:
        errors.append("Quality threshold should be lower than publishability threshold")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# Run validation and print status
if __name__ == "__main__":
    print("NewsAI Configuration Loaded")
    print("=" * 40)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Vector DB: {VECTOR_DB_PROVIDER} at {VECTOR_DB_PATH}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Quality threshold: {QUALITY_THRESHOLD}")
    print(f"Publishability threshold: {PUBLISHABILITY_THRESHOLD}")
    
    if validate_configuration():
        print("\n✅ Configuration Valid - Ready for Week 2!")
    else:
        print("\n❌ Configuration Issues Found")
    
    print(f"\nDataset Summary:")
    print(f"  Total articles: {DATASET_STATS['total_articles']:,}")
    print(f"  Approval rate: {DATASET_STATS['approval_rate']:.2%}")
    print(f"  Average quality score: {DATASET_STATS['avg_quality_score']:.3f}")
    print(f"  Optimal length range: {DATASET_STATS['optimal_length_range']}")