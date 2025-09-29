#!/usr/bin/env python3
"""
Week 2 Validation Script
Checks if all Week 2 components are properly set up and working
"""

import sys
import os
from pathlib import Path
import importlib.util
from config.settings import *

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'chromadb',
        'sentence_transformers', 
        'torch',
        'tqdm',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_project_structure():
    """Check if required files and directories exist"""
    required_paths = [
        'config/settings.py',
        'src/preprocessing/text_cleaner.py',
        'src/rag_engine',  # Directory
        'data/akhbar_sharq_awsat_fixed.xlsx',
        'data/akhbar_sharq_awsat_qbl_b3d_fixed.xlsx'
    ]
    
    missing_paths = []
    for path_str in required_paths:
        path = project_root / path_str
        if not path.exists():
            missing_paths.append(path_str)
    
    return missing_paths

def test_configuration():
    """Test configuration loading"""
    try:
        config_tests = {
            'Vector DB path configured': VECTOR_DB_PATH is not None,
            'Embedding model configured': EMBEDDING_MODEL is not None,
            'Collections configured': len(VECTOR_DB_COLLECTIONS) > 0,
            'Quality thresholds set': QUALITY_THRESHOLD is not None
        }
        return config_tests
    except Exception as e:
        return {'Configuration loading': f'Error: {e}'}

def test_vector_store():
    """Test vector store functionality"""
    try:
        from src.rag_engine.vector_store import NewsVectorStore
        
        # Initialize vector store
        vs = NewsVectorStore()
        
        # Test embedding generation
        test_texts = ["تجربة النص العربي", "Test Arabic text processing"]
        embeddings = vs.generate_embeddings(test_texts)
        
        # Get collection stats
        stats = vs.get_collection_stats()
        
        tests = {
            'Vector store initialization': True,
            'Embedding generation': embeddings.shape[0] == len(test_texts),
            'Collections created': len(stats) > 0,
            'ChromaDB working': all(stat['status'] != 'error' for stat in stats.values())
        }
        
        return tests, stats
        
    except Exception as e:
        return {'Vector store test': f'Error: {e}'}, {}

def test_preprocessing():
    """Test preprocessing module"""
    try:
        from src.preprocessing.text_cleaner import NewsArticleProcessor
        
        processor = NewsArticleProcessor()
        test_text = "الرئيس يؤكد أهمية التعاون الدولي في مواجهة التحديات"
        
        result = processor.process_article(test_text)
        
        tests = {
            'Preprocessing module import': True,
            'Article processing': result['processing_status'] == 'success',
            'Quality analysis': 'quality_score' in result['quality_analysis'],
            'Metadata extraction': len(result['metadata']) > 0
        }
        
        return tests
        
    except Exception as e:
        return {'Preprocessing test': f'Error: {e}'}

def main():
    """Run all validation tests"""
    print("NewsAI Week 2 Validation")
    print("=" * 40)
    
    # Check dependencies
    print("\n1. Checking Dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing packages: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
    else:
        print("✅ All required packages installed")
    
    # Check project structure
    print("\n2. Checking Project Structure...")
    missing_paths = check_project_structure()
    if missing_paths:
        print("❌ Missing files/directories:")
        for path in missing_paths:
            print(f"   - {path}")
    else:
        print("✅ All required files and directories exist")
    
    # Test configuration
    print("\n3. Testing Configuration...")
    config_tests = test_configuration()
    for test_name, result in config_tests.items():
        if isinstance(result, bool) and result:
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}: {result}")
    
    # Test preprocessing
    print("\n4. Testing Preprocessing...")
    preprocessing_tests = test_preprocessing()
    for test_name, result in preprocessing_tests.items():
        if isinstance(result, bool) and result:
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}: {result}")
    
    # Test vector store
    print("\n5. Testing Vector Store...")
    if missing_deps or missing_paths:
        print("❌ Skipping vector store test due to missing dependencies")
    else:
        vector_tests, collection_stats = test_vector_store()
        for test_name, result in vector_tests.items():
            if isinstance(result, bool) and result:
                print(f"✅ {test_name}")
            else:
                print(f"❌ {test_name}: {result}")
        
        if collection_stats:
            print("\n   Collection Status:")
            for collection, stats in collection_stats.items():
                print(f"     {collection}: {stats['count']} articles ({stats['status']})")
    
    # Overall status
    print("\n" + "=" * 40)
    
    all_deps_ok = len(missing_deps) == 0
    all_files_ok = len(missing_paths) == 0
    
    if all_deps_ok and all_files_ok:
        print("✅ Week 2 Setup: READY")
        print("\nNext steps:")
        print("1. Run: python notebooks/week2/01_vector_database_setup.ipynb")
        print("2. Process your full dataset")
        print("3. Test similarity search functionality")
    else:
        print("❌ Week 2 Setup: INCOMPLETE")
        print("\nPlease fix the issues above before proceeding")
    
    return all_deps_ok and all_files_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)