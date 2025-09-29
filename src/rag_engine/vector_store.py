"""
Vector Store Module for NewsAI
Handles embedding generation and vector database operations
"""

import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *
from src.preprocessing.text_cleaner import NewsArticleProcessor

class NewsVectorStore:
    """Vector database for Arabic news articles"""
    
    def __init__(self, persist_directory: str = None):
        """Initialize vector store"""
        self.persist_directory = persist_directory or str(VECTOR_DB_PATH)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Load embedding model
        print(f"Loading embedding model: {EMBEDDING_SETTINGS['model_name']}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_SETTINGS['model_name'],
            device=EMBEDDING_SETTINGS['device']
        )
        
        # Initialize collections
        self.collections = {}
        self._setup_collections()
        
        # Initialize processor
        self.processor = NewsArticleProcessor()
        
        print(f"Vector store initialized at: {self.persist_directory}")
    
    def _setup_collections(self):
        """Create or get existing collections"""
        for key, name in VECTOR_DB_COLLECTIONS.items():
            try:
                self.collections[key] = self.client.get_or_create_collection(
                    name=name,
                    metadata={
                        "description": f"Collection for {key}",
                        "created_by": "NewsAI",
                        "version": "1.0"
                    }
                )
                print(f"Collection '{name}' ready")
            except Exception as e:
                print(f"Error creating collection {name}: {e}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        batch_size = batch_size or EMBEDDING_SETTINGS['batch_size']
        
        if not texts:
            return np.array([])
        
        # Clean texts
        clean_texts = [str(text) if text else "" for text in texts]
        
        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(clean_texts), batch_size), desc="Generating embeddings"):
            batch = clean_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                normalize_embeddings=EMBEDDING_SETTINGS.get('normalize_embeddings', True),
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def prepare_article_metadata(self, row: pd.Series, processed_result: Dict = None) -> Dict:
        """Prepare metadata for an article"""
        metadata = {
            'story_id': str(row.get('StoryId', '')),
            'story_date': str(row.get('StoryDate', '')),
            'track_id': str(row.get('TrackId', '')),
            'dist': str(row.get('dist', '')),
            'is_approved': bool(row.get('is_approved', False)),
            'is_distributed': bool(row.get('is_distributed', False)),
            'text_length': int(len(str(row.get('Story', '')))),
            'word_count': int(len(str(row.get('Story', '')).split()))
        }
        
        # Add processed metadata if available
        if processed_result:
            quality_analysis = processed_result.get('quality_analysis', {})
            article_metadata = processed_result.get('metadata', {})
            
            metadata.update({
                'quality_score': float(quality_analysis.get('quality_score', 0)),
                'has_agency_tag': bool(article_metadata.get('has_agency_tag', False)),
                'has_attribution': bool(article_metadata.get('has_attribution', False)),
                'has_quotes': bool(article_metadata.get('has_quotes', False)),
                'has_location_dateline': bool(article_metadata.get('has_location_dateline', False)),
                'processing_status': processed_result.get('processing_status', 'unknown')
            })
        
        return metadata
    
    def add_articles_to_collection(self, df: pd.DataFrame, collection_key: str, 
                                 process_articles: bool = True) -> Dict:
        """Add articles to a specific collection"""
        
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found")
        
        collection = self.collections[collection_key]
        
        print(f"Processing {len(df)} articles for collection '{collection_key}'")
        
        # Process articles if requested
        processed_data = {}  # Use dict to store by index
        if process_articles:
            print("Running text preprocessing...")
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
                result = self.processor.process_article(row['Story'])
                result['original_index'] = idx
                processed_data[idx] = result  # Store with DataFrame index as key
        
        # Prepare data for embedding
        texts = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Get processed result if available
            processed_result = processed_data.get(idx) if processed_data else None
            
            # Use cleaned text if available, otherwise original
            text = processed_result.get('cleaned_text', row['Story']) if processed_result else row['Story']
            
            if text and str(text).strip():
                texts.append(str(text))
                metadatas.append(self.prepare_article_metadata(row, processed_result))
                ids.append(f"{collection_key}_{row['StoryId']}")
        
        if not texts:
            return {'status': 'error', 'message': 'No valid texts to embed'}
        
        print(f"Generating embeddings for {len(texts)} articles...")
        embeddings = self.generate_embeddings(texts)
        
        # Add to collection in batches
        batch_size = EMBEDDING_SETTINGS['batch_size']
        added_count = 0
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Adding to vector DB"):
            end_idx = min(i + batch_size, len(texts))
            
            try:
                collection.add(
                    embeddings=embeddings[i:end_idx].tolist(),
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                added_count += (end_idx - i)
            except Exception as e:
                print(f"Error adding batch {i}-{end_idx}: {e}")
        
        result = {
            'status': 'success',
            'collection': collection_key,
            'articles_processed': len(df),
            'articles_added': added_count,
            'collection_size': collection.count()
        }
        
        print(f"Added {added_count} articles to {collection_key}")
        return result
    
    def search_similar(self, query_text: str, collection_key: str, 
                      top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """Search for similar articles in a collection"""
        
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found")
        
        collection = self.collections[collection_key]
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text], normalize_embeddings=True)
        
        # Search
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=filters
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'id': results['ids'][0][i],
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'similarity': 1 - results['distances'][0][i] if results['distances'] else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics for all collections"""
        stats = {}
        
        for key, collection in self.collections.items():
            try:
                count = collection.count()
                stats[key] = {
                    'name': VECTOR_DB_COLLECTIONS[key],
                    'count': count,
                    'status': 'active' if count > 0 else 'empty'
                }
            except Exception as e:
                stats[key] = {
                    'name': VECTOR_DB_COLLECTIONS[key],
                    'count': 0,
                    'status': f'error: {e}'
                }
        
        return stats
    
    def save_processing_report(self, results: List[Dict], output_file: str = None):
        """Save processing results to file"""
        output_file = output_file or (ANALYTICS_DIR / "vector_db_processing_report.json")
        
        report = {
            'processing_results': results,
            'collection_stats': self.get_collection_stats(),
            'settings': {
                'embedding_model': EMBEDDING_SETTINGS['model_name'],
                'vector_db_path': str(self.persist_directory),
                'total_collections': len(self.collections)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Processing report saved to: {output_file}")
        return output_file


def test_vector_store():
    """Test vector store functionality"""
    print("Testing Vector Store...")
    
    # Initialize
    vs = NewsVectorStore()
    
    # Test embedding generation
    test_texts = [
        "الرئيس يؤكد أهمية التعاون الدولي",
        "وزير الخارجية يلتقي نظيره الأمريكي",
        "ارتفاع أسعار النفط في الأسواق العالمية"
    ]
    
    embeddings = vs.generate_embeddings(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity search (if data exists)
    stats = vs.get_collection_stats()
    print("Collection Stats:")
    for key, stat in stats.items():
        print(f"  {key}: {stat['count']} articles ({stat['status']})")
    
    return vs

if __name__ == "__main__":
    test_vector_store()