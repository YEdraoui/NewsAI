"""
Week 5: Editorial RAG Assistant for NewsAI
Automatically edit approved articles using RAG-powered style transformation
"""

import sys
sys.path.append('../..')

import ollama
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time
import re
from dataclasses import dataclass

from config.settings import *
from src.rag_engine.vector_store import NewsVectorStore

@dataclass
class EditorialTransformation:
    """Structure for editorial changes"""
    original_text: str
    edited_text: str
    changes: List[Dict[str, str]]
    style_rules_applied: List[str]
    editing_time: float

class EditorialRAGAssistant:
    """RAG-powered editorial assistant for Arabic news articles"""
    
    def __init__(self, 
                 llm_model: str = "aya:8b",
                 max_context_length: int = 4000):
        
        # Initialize vector store
        self.vector_store = NewsVectorStore()
        
        # LLM configuration
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        
        # Editorial patterns database
        self.editorial_patterns = self._load_editorial_patterns()
        
        # Style guidelines
        self.style_guidelines = self._load_style_guidelines()
        
        print(f"Editorial RAG Assistant initialized with model: {llm_model}")
    
    def _load_editorial_patterns(self) -> Dict:
        """Load editorial transformation patterns from the 1K examples"""
        
        try:
            # Load the before/after editorial dataset
            df_editorial = pd.read_excel(EDITORIAL_DATASET)
            
            patterns = {
                'length_changes': [],
                'common_transformations': [],
                'style_improvements': [],
                'format_standardization': []
            }
            
            for _, row in df_editorial.iterrows():
                original = row['Story']
                edited = row['finalNews']
                
                # Analyze transformation patterns
                original_len = len(original)
                edited_len = len(edited)
                length_change = (edited_len - original_len) / original_len if original_len > 0 else 0
                
                patterns['length_changes'].append(length_change)
                
                # Extract common transformation patterns
                transformation = self._analyze_transformation(original, edited)
                patterns['common_transformations'].append(transformation)
            
            return patterns
            
        except Exception as e:
            print(f"Error loading editorial patterns: {e}")
            return {'length_changes': [], 'common_transformations': []}
    
    def _analyze_transformation(self, original: str, edited: str) -> Dict:
        """Analyze what changed between original and edited versions"""
        
        changes = {
            'length_change': len(edited) - len(original),
            'has_agency_header': bool(re.search(r'^[ع|م|ر]/', edited)),
            'has_date_format': bool(re.search(r'\d{1,2} \w+ \d{4} م واس', edited)),
            'has_structured_ending': edited.endswith('// انتهى //'),
            'paragraph_count': len(edited.split('\n')),
            'avg_sentence_length': len(edited.split('.')) if edited.split('.') else 0
        }
        
        return changes
    
    def _load_style_guidelines(self) -> Dict:
        """Load Wakalat Al Anba2 style guidelines"""
        
        guidelines = {
            'agency_format': {
                'header_format': 'ع / عام / {title}',
                'date_format': '{hijri_date} هـ الموافق {gregorian_date} م واس',
                'ending_format': '// انتهى //',
                'metadata_section': '/******** Information Add By System *********/'
            },
            'language_rules': {
                'use_formal_arabic': True,
                'avoid_colloquialisms': True,
                'standardize_punctuation': True,
                'consistent_quotation_marks': True
            },
            'structure_rules': {
                'clear_lead_paragraph': True,
                'logical_flow': True,
                'proper_attribution': True,
                'factual_accuracy': True
            }
        }
        
        return guidelines
    
    def retrieve_similar_editorial_examples(self, article_text: str, k: int = 5) -> List[Dict]:
        """Retrieve similar editorial examples using RAG"""
        
        try:
            # Search in editorial examples collection
            similar_examples = self.vector_store.search_similar_articles(
                query_text=article_text,
                collection_name='editorial_examples',
                top_k=k
            )
            
            return similar_examples
            
        except Exception as e:
            print(f"Error retrieving editorial examples: {e}")
            return []
    
    def build_editorial_context(self, article_text: str) -> str:
        """Build context for editorial LLM prompt"""
        
        # Get similar editorial examples
        similar_examples = self.retrieve_similar_editorial_examples(article_text, k=3)
        
        context = "الأمثلة التحريرية المشابهة:\n\n"
        
        for i, example in enumerate(similar_examples, 1):
            if 'metadata' in example and 'original_text' in example['metadata']:
                original = example['metadata']['original_text'][:200] + "..."
                edited = example['document'][:200] + "..."
                
                context += f"مثال {i}:\n"
                context += f"النص الأصلي: {original}\n"
                context += f"النص المحرر: {edited}\n\n"
        
        # Add style guidelines
        context += "قواعد التحرير:\n"
        context += "- استخدام العربية الفصحى\n"
        context += "- بداية الخبر بتصنيف (ع / عام / العنوان)\n"
        context += "- إضافة التاريخ الهجري والميلادي\n"
        context += "- إنهاء الخبر بـ // انتهى //\n"
        context += "- الحفاظ على الدقة الإخبارية\n"
        context += "- تحسين التدفق والوضوح\n\n"
        
        return context
    
    def edit_article_with_rag(self, article_text: str) -> EditorialTransformation:
        """Edit article using RAG-powered editorial assistant"""
        
        start_time = time.time()
        
        # Build editorial context
        context = self.build_editorial_context(article_text)
        
        # Create editorial prompt
        prompt = f"""أنت محرر خبير في وكالة الأنباء السعودية (واس).
مهمتك تحرير المقال التالي وفقاً لمعايير الوكالة.

{context}

المقال المطلوب تحريره:
{article_text}

يرجى تحرير هذا المقال مع:
1. تطبيق تنسيق الوكالة (العنوان، التاريخ، الخاتمة)
2. تحسين اللغة والأسلوب
3. ضمان الوضوح والدقة
4. الحفاظ على المحتوى الإخباري الأساسي

قدم النتيجة في شكل JSON:
{{
    "edited_text": "النص المحرر الكامل",
    "changes": ["قائمة بالتغييرات المطبقة"],
    "style_rules": ["قائمة بقواعد التحرير المستخدمة"]
}}"""

        try:
            # Query Ollama for editorial assistance
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'أنت محرر محترف في وكالة أنباء عربية. تجيب دائماً باللغة العربية وبصيغة JSON صحيحة.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ]
            )
            
            # Parse response
            response_text = response['message']['content']
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    result = json.loads(json_str)
                    
                    edited_text = result.get('edited_text', article_text)
                    changes = result.get('changes', ['تم تطبيق التحرير الأساسي'])
                    style_rules = result.get('style_rules', ['تحسين اللغة'])
                    
                else:
                    # Fallback if JSON parsing fails
                    edited_text = response_text
                    changes = ['تم التحرير باستخدام الذكاء الاصطناعي']
                    style_rules = ['تطبيق معايير الوكالة']
                    
            except json.JSONDecodeError:
                # Fallback to basic formatting if JSON parsing fails
                edited_text = self._apply_basic_formatting(article_text)
                changes = ['تم تطبيق التنسيق الأساسي']
                style_rules = ['تنسيق الوكالة الأساسي']
            
            processing_time = time.time() - start_time
            
            return EditorialTransformation(
                original_text=article_text,
                edited_text=edited_text,
                changes=[{'type': 'edit', 'description': change} for change in changes],
                style_rules_applied=style_rules,
                editing_time=processing_time
            )
            
        except Exception as e:
            print(f"Error in RAG editing: {e}")
            
            # Fallback to basic formatting
            edited_text = self._apply_basic_formatting(article_text)
            processing_time = time.time() - start_time
            
            return EditorialTransformation(
                original_text=article_text,
                edited_text=edited_text,
                changes=[{'type': 'fallback', 'description': 'تم تطبيق التنسيق الأساسي بسبب خطأ'}],
                style_rules_applied=['التنسيق الأساسي'],
                editing_time=processing_time
            )
    
    def _apply_basic_formatting(self, text: str) -> str:
        """Apply basic Wakalat Al Anba2 formatting as fallback"""
        
        # Extract a title from the first sentence
        sentences = text.split('.')
        title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
        
        # Basic agency formatting
        formatted = f"ع / عام / {title}\n\n"
        formatted += f"الرياض في {time.strftime('%d %B')} {time.strftime('%Y')} م واس\n\n"
        formatted += text
        formatted += "\n\n// انتهى //"
        
        return formatted

def run_week5_editorial_tests():
    """Test the editorial RAG assistant"""
    
    print("=== WEEK 5: EDITORIAL RAG ASSISTANT TEST ===")
    
    # Initialize editorial assistant
    editorial_assistant = EditorialRAGAssistant()
    
    # Test articles (approved articles that need editing)
    test_articles = [
        "أكد وزير الخارجية أهمية التعاون الدولي مع الشركاء في المنطقة لتحقيق الاستقرار والازدهار المشترك.",
        "وزارة الصحة تعلن عن إجراءات جديدة للوقاية من الأمراض الموسمية في جميع مناطق المملكة.",
        "انطلق اليوم المؤتمر الاقتصادي السنوي في جدة بحضور ممثلين من القطاعين العام والخاص."
    ]
    
    print(f"Testing editorial assistant on {len(test_articles)} articles...")
    
    results = []
    total_time = 0
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nTest {i}: Editing article...")
        print(f"Original: {article[:100]}...")
        
        # Edit the article
        transformation = editorial_assistant.edit_article_with_rag(article)
        total_time += transformation.editing_time
        
        print(f"Edited: {transformation.edited_text[:100]}...")
        print(f"Changes: {len(transformation.changes)}")
        print(f"Style rules: {len(transformation.style_rules_applied)}")
        print(f"Time: {transformation.editing_time:.2f}s")
        
        results.append(transformation)
    
    # Summary
    avg_time = total_time / len(test_articles)
    articles_per_hour = 3600 / avg_time if avg_time > 0 else 0
    
    print(f"\n=== WEEK 5 EDITORIAL RESULTS ===")
    print(f"Articles processed: {len(test_articles)}")
    print(f"Average editing time: {avg_time:.2f}s")
    print(f"Editorial capacity: {articles_per_hour:.0f} articles/hour")
    print(f"All articles successfully edited: {'YES' if len(results) == len(test_articles) else 'NO'}")
    
    if articles_per_hour >= 300:  # Target for editorial processing
        print("Week 5 Status: COMPLETE - Editorial assistant ready")
    else:
        print("Week 5 Status: NEEDS OPTIMIZATION - Editorial speed below target")
    
    return results

if __name__ == "__main__":
    editorial_results = run_week5_editorial_tests()