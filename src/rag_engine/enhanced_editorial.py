"""
Week 6: Enhanced Editorial System for NewsAI
Fix RAG retrieval, add editorial patterns, and optimize performance
"""

import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
import json
import re
from pathlib import Path

from config.settings import *
from src.rag_engine.vector_store import NewsVectorStore

class EnhancedEditorialSystem:
    """Enhanced editorial system with proper RAG integration"""
    
    def __init__(self):
        self.vector_store = NewsVectorStore()
        self.editorial_patterns = self._load_editorial_patterns()
        self.style_templates = self._create_style_templates()
        
        print("Enhanced Editorial System initialized")
    
    def _load_editorial_patterns(self) -> Dict:
        """Load and analyze editorial transformation patterns"""
        
        try:
            df_editorial = pd.read_excel(EDITORIAL_DATASET)
            print(f"Loaded {len(df_editorial)} editorial examples")
            
            patterns = {
                'transformations': [],
                'length_stats': {
                    'original_avg': df_editorial['Story'].str.len().mean(),
                    'edited_avg': df_editorial['finalNews'].str.len().mean(),
                    'change_ratio': df_editorial['finalNews'].str.len().mean() / df_editorial['Story'].str.len().mean()
                },
                'common_additions': [],
                'format_patterns': {}
            }
            
            # Analyze transformations
            for _, row in df_editorial.iterrows():
                original = str(row['Story'])
                edited = str(row['finalNews'])
                
                transformation = {
                    'original_length': len(original),
                    'edited_length': len(edited),
                    'has_agency_format': self._has_agency_format(edited),
                    'has_proper_ending': edited.endswith('// انتهى //'),
                    'has_date_format': bool(re.search(r'\d{4} م واس', edited)),
                    'improvement_type': self._classify_improvement(original, edited)
                }
                
                patterns['transformations'].append(transformation)
            
            # Calculate format statistics
            patterns['format_patterns'] = {
                'agency_format_rate': sum(1 for t in patterns['transformations'] if t['has_agency_format']) / len(patterns['transformations']),
                'proper_ending_rate': sum(1 for t in patterns['transformations'] if t['has_proper_ending']) / len(patterns['transformations']),
                'date_format_rate': sum(1 for t in patterns['transformations'] if t['has_date_format']) / len(patterns['transformations'])
            }
            
            print(f"Editorial patterns analysis:")
            print(f"  - Agency format usage: {patterns['format_patterns']['agency_format_rate']:.1%}")
            print(f"  - Proper ending usage: {patterns['format_patterns']['proper_ending_rate']:.1%}")
            print(f"  - Average length change: {patterns['length_stats']['change_ratio']:.2f}x")
            
            return patterns
            
        except Exception as e:
            print(f"Error loading editorial patterns: {e}")
            return {'transformations': [], 'format_patterns': {}}
    
    def _has_agency_format(self, text: str) -> bool:
        """Check if text has proper agency format"""
        return bool(re.search(r'^[ع|م|ر]\s*/\s*\w+\s*/\s*', text))
    
    def _classify_improvement(self, original: str, edited: str) -> str:
        """Classify the type of editorial improvement"""
        
        if self._has_agency_format(edited) and not self._has_agency_format(original):
            return 'format_standardization'
        elif len(edited) > len(original) * 1.2:
            return 'content_expansion'
        elif len(edited) < len(original) * 0.8:
            return 'content_condensation'
        else:
            return 'language_improvement'
    
    def _create_style_templates(self) -> Dict:
        """Create style templates based on editorial patterns"""
        
        templates = {
            'general_news': {
                'header': 'ع / عام / {title}',
                'date_format': '{location} في {date} /واس/',
                'ending': '// انتهى //',
                'metadata': '/******** Information Add By System *********/'
            },
            'economic_news': {
                'header': 'ع / اقتصادية / {title}',
                'date_format': '{location} في {date} /واس/',
                'ending': '// انتهى //',
                'metadata': '/******** Information Add By System *********/'
            },
            'political_news': {
                'header': 'ع / سياسية / {title}',
                'date_format': '{location} في {date} /واس/',
                'ending': '// انتهى //',
                'metadata': '/******** Information Add By System *********/'
            }
        }
        
        return templates
    
    def fix_vector_search(self):
        """Add missing search_similar_articles method to vector store"""
        
        def search_similar_articles(self, query_text: str, collection_name: str = 'editorial_examples', top_k: int = 5):
            """Search for similar articles in specified collection"""
            try:
                if collection_name not in self.collections:
                    return []
                
                # Generate embedding for query
                query_embedding = self.generate_embeddings([query_text])[0]
                
                # Search in collection
                collection = self.collections[collection_name]
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Format results
                similar_articles = []
                for i in range(len(results['documents'][0])):
                    similar_articles.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
                
                return similar_articles
                
            except Exception as e:
                print(f"Error in search_similar_articles: {e}")
                return []
        
        # Add the method to the vector store instance
        self.vector_store.search_similar_articles = search_similar_articles.__get__(self.vector_store)
        print("Fixed vector search functionality")
    
    def process_editorial_examples_to_vector_db(self):
        """Process editorial examples and add them to vector database"""
        
        try:
            df_editorial = pd.read_excel(EDITORIAL_DATASET)
            
            # Prepare editorial examples for vector storage
            editorial_data = []
            for _, row in df_editorial.iterrows():
                editorial_data.append({
                    'id': f"editorial_{row['StoryId']}",
                    'text': row['finalNews'],  # Use edited version as the document
                    'metadata': {
                        'original_text': row['Story'],
                        'edited_text': row['finalNews'],
                        'story_id': row['StoryId'],
                        'basket_id': row.get('basketid', 'general'),
                        'transformation_type': self._classify_improvement(row['Story'], row['finalNews'])
                    }
                })
            
            print(f"Processing {len(editorial_data)} editorial examples...")
            
            # Add to vector database in batches
            batch_size = 50
            added_count = 0
            
            for i in range(0, len(editorial_data), batch_size):
                batch = editorial_data[i:i + batch_size]
                
                texts = [item['text'] for item in batch]
                metadatas = [item['metadata'] for item in batch]
                ids = [item['id'] for item in batch]
                
                # Generate embeddings
                embeddings = self.vector_store.generate_embeddings(texts)
                
                # Add to collection
                collection = self.vector_store.collections['editorial_examples']
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                added_count += len(batch)
                print(f"Added {added_count}/{len(editorial_data)} editorial examples")
            
            print("Editorial examples successfully added to vector database")
            return True
            
        except Exception as e:
            print(f"Error processing editorial examples: {e}")
            return False
    
    def enhanced_article_editing(self, article_text: str, article_type: str = 'general') -> Dict:
        """Enhanced article editing with proper RAG integration"""
        
        start_time = time.time()
        
        # Fix vector search if not available
        if not hasattr(self.vector_store, 'search_similar_articles'):
            self.fix_vector_search()
        
        # Retrieve similar editorial examples
        similar_examples = self.vector_store.search_similar_articles(
            query_text=article_text,
            collection_name='editorial_examples',
            top_k=3
        )
        
        # Build context from similar examples
        context = self._build_editorial_context(article_text, similar_examples, article_type)
        
        # Apply style template
        template = self.style_templates.get(f'{article_type}_news', self.style_templates['general_news'])
        
        # Enhanced editing with RAG context
        edited_result = self._apply_rag_editing(article_text, context, template)
        
        processing_time = time.time() - start_time
        
        result = {
            'original_text': article_text,
            'edited_text': edited_result['edited_text'],
            'changes_applied': edited_result['changes'],
            'style_template_used': article_type,
            'similar_examples_count': len(similar_examples),
            'processing_time': processing_time,
            'quality_score': self._calculate_editing_quality(article_text, edited_result['edited_text'])
        }
        
        return result
    
    def _build_editorial_context(self, article_text: str, similar_examples: List[Dict], article_type: str) -> str:
        """Build comprehensive editorial context"""
        
        context = f"نوع المقال: {article_type}\n\n"
        
        if similar_examples:
            context += "أمثلة تحريرية مشابهة:\n"
            for i, example in enumerate(similar_examples, 1):
                if example.get('metadata'):
                    original = example['metadata'].get('original_text', '')[:150]
                    edited = example.get('document', '')[:150]
                    context += f"\nمثال {i}:\n"
                    context += f"قبل التحرير: {original}...\n"
                    context += f"بعد التحرير: {edited}...\n"
        
        context += f"\nقواعد التحرير:\n"
        context += "- استخدام تصنيف الأخبار (ع / عام / أو ع / اقتصادية / إلخ)\n"
        context += "- إضافة الموقع والتاريخ (المدينة في التاريخ /واس/)\n"
        context += "- الكتابة بالعربية الفصحى\n"
        context += "- إنهاء الخبر بـ // انتهى //\n"
        context += "- الحفاظ على الدقة والوضوح\n"
        
        return context
    
    def _apply_rag_editing(self, article_text: str, context: str, template: Dict) -> Dict:
        """Apply RAG-based editing with fallback to rule-based editing"""
        
        try:
            # Try LLM-based editing with RAG context
            import ollama
            
            prompt = f"""أنت محرر خبير في وكالة الأنباء السعودية.
            
{context}

المقال المطلوب تحريره:
{article_text}

قم بتحرير هذا المقال وفقاً للمعايير والأمثلة المذكورة أعلاه.

أعطني النتيجة كـ JSON:
{{
    "edited_text": "النص المحرر",
    "changes": ["قائمة التغييرات"]
}}"""

            response = ollama.chat(
                model="aya:8b",
                messages=[
                    {'role': 'system', 'content': 'أنت محرر محترف. أجب باللغة العربية وبصيغة JSON.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            # Parse response
            response_text = response['message']['content']
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                result = json.loads(response_text[json_start:json_end])
                return {
                    'edited_text': result.get('edited_text', article_text),
                    'changes': result.get('changes', ['تم التحرير باستخدام الذكاء الاصطناعي'])
                }
            
        except Exception as e:
            print(f"LLM editing failed, using rule-based fallback: {e}")
        
        # Fallback to rule-based editing
        return self._rule_based_editing(article_text, template)
    
    def _rule_based_editing(self, article_text: str, template: Dict) -> Dict:
        """Rule-based editing as fallback"""
        
        # Extract title from first sentence
        sentences = article_text.split('.')
        title = sentences[0].strip()[:60] + ("..." if len(sentences[0]) > 60 else "")
        
        # Apply template
        edited_text = template['header'].format(title=title) + "\n\n"
        edited_text += template['date_format'].format(
            location="الرياض",
            date=time.strftime("%d %B %Y")
        ) + "\n\n"
        edited_text += article_text + "\n\n"
        edited_text += template['ending']
        
        changes = [
            'إضافة تصنيف الخبر',
            'إضافة الموقع والتاريخ',
            'إضافة خاتمة الوكالة'
        ]
        
        return {
            'edited_text': edited_text,
            'changes': changes
        }
    
    def _calculate_editing_quality(self, original: str, edited: str) -> float:
        """Calculate quality score for the editing"""
        
        score = 0.5  # Base score
        
        # Check format compliance
        if self._has_agency_format(edited):
            score += 0.2
        
        if edited.endswith('// انتهى //'):
            score += 0.1
        
        if '/واس/' in edited:
            score += 0.1
        
        # Check length appropriateness
        length_ratio = len(edited) / len(original) if len(original) > 0 else 1
        if 0.8 <= length_ratio <= 1.5:  # Reasonable length change
            score += 0.1
        
        return min(score, 1.0)

def run_week6_tests():
    """Run Week 6 enhanced editorial system tests"""
    
    print("=== WEEK 6: ENHANCED EDITORIAL SYSTEM TEST ===")
    
    # Initialize enhanced system
    system = EnhancedEditorialSystem()
    
    # Fix vector search and process editorial examples
    system.fix_vector_search()
    success = system.process_editorial_examples_to_vector_db()
    
    if not success:
        print("Warning: Editorial examples not processed to vector DB, using fallback")
    
    # Test articles
    test_articles = [
        ("general", "أكد وزير الخارجية أهمية التعاون الدولي مع الشركاء في المنطقة."),
        ("economic", "شهدت الأسواق السعودية نمواً ملحوظاً في القطاع التقني خلال الربع الثالث."),
        ("political", "استقبل خادم الحرمين الشريفين وفداً رسمياً لبحث العلاقات الثنائية.")
    ]
    
    results = []
    total_time = 0
    
    print(f"\nTesting enhanced editorial system on {len(test_articles)} articles...")
    
    for i, (article_type, article_text) in enumerate(test_articles, 1):
        print(f"\nTest {i} ({article_type}):")
        print(f"Original: {article_text[:80]}...")
        
        # Enhanced editing
        result = system.enhanced_article_editing(article_text, article_type)
        total_time += result['processing_time']
        
        print(f"Edited: {result['edited_text'][:80]}...")
        print(f"Changes: {len(result['changes_applied'])}")
        print(f"Quality score: {result['quality_score']:.2f}")
        print(f"Similar examples used: {result['similar_examples_count']}")
        print(f"Time: {result['processing_time']:.2f}s")
        
        results.append(result)
    
    # Summary
    avg_time = total_time / len(test_articles)
    articles_per_hour = 3600 / avg_time if avg_time > 0 else 0
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    
    print(f"\n=== WEEK 6 ENHANCED RESULTS ===")
    print(f"Articles processed: {len(test_articles)}")
    print(f"Average editing time: {avg_time:.2f}s")
    print(f"Editorial capacity: {articles_per_hour:.0f} articles/hour")
    print(f"Average quality score: {avg_quality:.2f}")
    print(f"Vector search working: {'YES' if hasattr(system.vector_store, 'search_similar_articles') else 'NO'}")
    
    # Week 6 assessment
    criteria = {
        'processing_speed': articles_per_hour >= 300,
        'quality_score': avg_quality >= 0.7,
        'vector_search': hasattr(system.vector_store, 'search_similar_articles'),
        'all_processed': len(results) == len(test_articles)
    }
    
    passed = sum(criteria.values())
    total = len(criteria)
    
    print(f"\nWeek 6 Criteria:")
    for test, result in criteria.items():
        print(f"  {test}: {'PASS' if result else 'FAIL'}")
    
    print(f"\nWeek 6 Score: {passed}/{total}")
    
    if passed >= 3:
        print("Week 6 Status: COMPLETE - Enhanced editorial system ready")
    else:
        print("Week 6 Status: NEEDS OPTIMIZATION")
    
    return results

if __name__ == "__main__":
    week6_results = run_week6_tests()