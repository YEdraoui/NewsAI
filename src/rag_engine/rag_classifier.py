"""
RAG Publishability Classifier for NewsAI
Determines whether Arabic news articles should be approved or rejected
"""

import ollama
import json
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *
from src.rag_engine.vector_store import NewsVectorStore
from src.preprocessing.text_cleaner import NewsArticleProcessor

class RAGPublishabilityClassifier:
    """RAG-powered news article publishability classifier"""
    
    def __init__(self):
        """Initialize the classifier"""
        self.vector_store = NewsVectorStore()
        self.processor = NewsArticleProcessor()
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model = LLM_MODEL
        
        print(f"RAG Classifier initialized with model: {self.model}")
    
    def retrieve_similar_examples(self, article_text: str, top_k: int = 5) -> Dict[str, List]:
        """Retrieve similar approved and rejected examples"""
        
        # Search approved articles
        approved_examples = self.vector_store.search_similar(
            query_text=article_text,
            collection_key='approved_articles',
            top_k=top_k
        )
        
        # Search rejected articles  
        rejected_examples = self.vector_store.search_similar(
            query_text=article_text,
            collection_key='rejected_articles',
            top_k=top_k
        )
        
        return {
            'approved': approved_examples,
            'rejected': rejected_examples
        }
    
    def format_examples_for_prompt(self, examples: Dict[str, List]) -> str:
        """Format retrieved examples for LLM prompt"""
        
        prompt_context = ""
        
        # Format approved examples
        if examples['approved']:
            prompt_context += "أمثلة على الأخبار المقبولة (تم الموافقة عليها):\n\n"
            for i, example in enumerate(examples['approved'][:3], 1):
                similarity = example.get('similarity', 0)
                text_preview = example['document'][:200] + "..."
                quality_score = example['metadata'].get('quality_score', 'غير متوفر')
                
                prompt_context += f"مثال مقبول {i} (تشابه: {similarity:.2f}, جودة: {quality_score}):\n"
                prompt_context += f"{text_preview}\n\n"
        
        # Format rejected examples
        if examples['rejected']:
            prompt_context += "أمثلة على الأخبار المرفوضة (تم رفضها):\n\n"
            for i, example in enumerate(examples['rejected'][:3], 1):
                similarity = example.get('similarity', 0)
                text_preview = example['document'][:200] + "..."
                quality_score = example['metadata'].get('quality_score', 'غير متوفر')
                
                prompt_context += f"مثال مرفوض {i} (تشابه: {similarity:.2f}, جودة: {quality_score}):\n"
                prompt_context += f"{text_preview}\n\n"
        
        return prompt_context
    
    def build_classification_prompt(self, article_text: str, examples_context: str, 
                                  article_metadata: Dict = None) -> str:
        """Build the complete prompt for classification"""
        
        metadata_info = ""
        if article_metadata:
            quality_score = article_metadata.get('quality_score', 'غير متوفر')
            has_attribution = article_metadata.get('has_attribution', False)
            has_agency_tag = article_metadata.get('has_agency_tag', False)
            text_length = article_metadata.get('text_length', len(article_text))
            
            metadata_info = f"""
معلومات تحليل المقال:
- درجة الجودة: {quality_score}
- يحتوي على إسناد: {'نعم' if has_attribution else 'لا'}
- يحتوي على علامة الوكالة: {'نعم' if has_agency_tag else 'لا'}
- طول النص: {text_length} حرف
"""
        
        prompt = f"""
أنت محرر خبير في وكالة الأنباء السعودية (واس). مهمتك تحديد ما إذا كان هذا المقال الإخباري يجب الموافقة عليه أم رفضه للنشر.

معايير الموافقة:
- المقال يحتوي على معلومات موثوقة ومفيدة
- يتبع معايير الجودة الصحفية
- طول مناسب وتفاصيل كافية
- إسناد واضح للمصادر
- يتماشى مع معايير وكالة الأنباء السعودية

معايير الرفض:
- معلومات غير موثوقة أو مضللة
- نص قصير جداً أو غير مكتمل
- عدم وجود إسناد للمصادر
- جودة كتابة ضعيفة
- لا يتماشى مع معايير الوكالة

{examples_context}

المقال المطلوب تقييمه:
{article_text}

{metadata_info}

استناداً إلى الأمثلة أعلاه ومعايير وكالة الأنباء السعودية، قم بتحليل هذا المقال وتقديم:

1. القرار: "موافق" أو "مرفوض"
2. درجة الثقة: من 0.0 إلى 1.0
3. السبب: شرح مفصل للقرار باللغة العربية
4. المراجع: أي من الأمثلة المشابهة التي أثرت على قرارك

قدم إجابتك في شكل JSON:
{{
    "decision": "موافق" أو "مرفوض",
    "confidence": 0.85,
    "reasoning": "شرح مفصل للقرار...",
    "similar_examples": ["مثال مقبول 1", "مثال مرفوض 2"],
    "quality_indicators": ["مؤشر 1", "مؤشر 2"]
}}
"""
        
        return prompt
    
    def query_ollama(self, prompt: str) -> Dict:
        """Query Ollama model for classification"""
        
        try:
            response = self.ollama_client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': RAG_SETTINGS['temperature'],
                    'top_p': RAG_SETTINGS['top_p'],
                    'num_predict': RAG_SETTINGS['max_tokens']
                }
            )
            
            # Extract response content
            response_text = response['message']['content']
            
            # Try to parse JSON response
            try:
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                
                if start != -1 and end != 0:
                    json_str = response_text[start:end]
                    result = json.loads(json_str)
                    return result
                else:
                    # Fallback parsing
                    return self._parse_text_response(response_text)
                    
            except json.JSONDecodeError:
                return self._parse_text_response(response_text)
                
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return {
                'decision': 'مرفوض',
                'confidence': 0.0,
                'reasoning': f'خطأ في المعالجة: {str(e)}',
                'similar_examples': [],
                'quality_indicators': []
            }
    
    def _parse_text_response(self, response_text: str) -> Dict:
        """Fallback parser for non-JSON responses"""
        
        # Simple parsing logic
        decision = 'مرفوض'
        if any(word in response_text for word in ['موافق', 'مقبول', 'approve']):
            decision = 'موافق'
        
        # Extract confidence if mentioned
        confidence = 0.5
        import re
        conf_match = re.search(r'(\d+\.?\d*)', response_text)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except:
                confidence = 0.5
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reasoning': response_text[:500],
            'similar_examples': [],
            'quality_indicators': []
        }
    
    def classify_article(self, article_text: str, include_preprocessing: bool = True) -> Dict:
        """Main classification method"""
        
        if not article_text or not article_text.strip():
            return {
                'status': 'error',
                'error': 'النص فارغ أو غير صالح',
                'decision': 'مرفوض',
                'confidence': 0.0
            }
        
        try:
            # Preprocess article if requested
            article_metadata = None
            processed_text = article_text
            
            if include_preprocessing:
                processed_result = self.processor.process_article(article_text)
                if processed_result['processing_status'] == 'success':
                    processed_text = processed_result['cleaned_text']
                    article_metadata = {
                        'quality_score': processed_result['quality_analysis']['quality_score'],
                        'has_attribution': processed_result['metadata'].get('has_attribution', False),
                        'has_agency_tag': processed_result['metadata'].get('has_agency_tag', False),
                        'text_length': len(processed_text)
                    }
            
            # Retrieve similar examples
            examples = self.retrieve_similar_examples(processed_text)
            
            # Format context
            examples_context = self.format_examples_for_prompt(examples)
            
            # Build prompt
            prompt = self.build_classification_prompt(
                processed_text, 
                examples_context, 
                article_metadata
            )
            
            # Query LLM
            llm_result = self.query_ollama(prompt)
            
            # Combine results
            result = {
                'status': 'success',
                'decision': llm_result.get('decision', 'مرفوض'),
                'confidence': llm_result.get('confidence', 0.0),
                'reasoning': llm_result.get('reasoning', ''),
                'similar_examples': {
                    'approved': examples['approved'][:3],
                    'rejected': examples['rejected'][:3]
                },
                'quality_indicators': llm_result.get('quality_indicators', []),
                'article_metadata': article_metadata,
                'processing_time': 'N/A'  # Could add timing
            }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'decision': 'مرفوض',
                'confidence': 0.0
            }
    
    def test_classification(self, test_articles: List[str]) -> List[Dict]:
        """Test classifier on multiple articles"""
        
        results = []
        for i, article in enumerate(test_articles):
            print(f"Testing article {i+1}/{len(test_articles)}")
            result = self.classify_article(article)
            results.append(result)
        
        return results


def test_rag_classifier():
    """Test the RAG classifier with sample articles"""
    
    # Initialize classifier
    classifier = RAGPublishabilityClassifier()
    
    # Test articles
    test_articles = [
        # Good quality article
        """
        الرياض في 15 سبتمبر /واس/ أكد صاحب السمو الملكي الأمير محمد بن سلمان بن عبدالعزيز ولي العهد رئيس مجلس الوزراء، أهمية تعزيز التعاون الاستراتيجي بين المملكة العربية السعودية والولايات المتحدة الأمريكية في مختلف المجالات.
        جاء ذلك خلال استقبال سموه في قصر اليمامة بالرياض اليوم، وزير الخارجية الأمريكي أنتوني بلينكن والوفد المرافق له.
        وبحث الجانبان خلال اللقاء، أوجه التعاون الثنائي بين البلدين، والمستجدات الإقليمية والدولية ذات الاهتمام المشترك.
        """,
        
        # Poor quality article
        """
        خبر مهم جداً
        شيء حدث اليوم
        """,
        
        # Medium quality article
        """
        أعلنت وزارة الصحة اليوم عن تسجيل حالات جديدة في المملكة.
        وأوضحت الوزارة أن العدد في ازدياد مستمر.
        """
    ]
    
    print("Testing RAG Publishability Classifier")
    print("=" * 50)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nTest Article {i}:")
        print("-" * 30)
        print(article.strip()[:100] + "...")
        
        result = classifier.classify_article(article)
        
        if result['status'] == 'success':
            print(f"Decision: {result['decision']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Reasoning: {result['reasoning'][:100]}...")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_rag_classifier()