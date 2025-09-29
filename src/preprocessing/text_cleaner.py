"""
Arabic Text Preprocessing Module for NewsAI
Handles cleaning, normalization, and feature extraction for Arabic news articles
"""

import re
import string
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

class ArabicTextCleaner:
    """Comprehensive Arabic text cleaning and normalization"""
    
    def __init__(self):
        # Arabic-specific patterns
        self.arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        self.arabic_punctuation = '،؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؜؝؞؟ؠ'
        
        # Character normalization patterns
        self.normalization_patterns = {
            # Alef variations → standard alef
            '[إأآا]': 'ا',
            # Yeh variations → standard yeh
            'ى': 'ي',
            # Teh marbuta → heh
            'ة': 'ه',
            # Remove tatweel (kashida)
            'ـ': '',
            # Normalize spaces
            '\s+': ' '
        }
        
        # News-specific patterns
        self.news_patterns = {
            'agency_tag': re.compile(r'/[^/]*ش[^/]*أ[^/]*/', re.UNICODE),
            'date_arabic': re.compile(r'\d{1,2}\s+(يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s*/?\s*\d{0,4}', re.UNICODE),
            'location_dateline': re.compile(r'^[ا-ي\s]{2,20}\s+(في|فى)\s+\d', re.UNICODE),
            'attribution': re.compile(r'(قال|أضاف|أوضح|أكد|صرح|ذكر|أشار|بين)', re.UNICODE),
            'source_mention': re.compile(r'(وكالة|مصدر|مراسل|نقلا عن|حسب)', re.UNICODE),
            'quotes': re.compile(r'["""''«»]', re.UNICODE)
        }
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (harakat)"""
        return self.arabic_diacritics.sub('', text) if text else ""
    
    def normalize_arabic_chars(self, text: str) -> str:
        """Normalize Arabic character variations"""
        if not text:
            return ""
        
        normalized = text
        for pattern, replacement in self.normalization_patterns.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def clean_whitespace_and_punctuation(self, text: str) -> str:
        """Clean excessive whitespace and punctuation"""
        if not text:
            return ""
        
        # Multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Multiple punctuation to single
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_metadata(self, text: str) -> Dict[str, any]:
        """Extract structured metadata from article text"""
        metadata = {
            'has_agency_tag': False,
            'agency_tag': None,
            'has_date': False,
            'date_info': None,
            'has_location_dateline': False,
            'location_info': None,
            'has_attribution': False,
            'attribution_count': 0,
            'has_source_mention': False,
            'has_quotes': False,
            'quote_count': 0
        }
        
        if not text:
            return metadata
        
        # Agency tag detection
        agency_match = self.news_patterns['agency_tag'].search(text)
        if agency_match:
            metadata['has_agency_tag'] = True
            metadata['agency_tag'] = agency_match.group().strip()
        
        # Date detection
        date_match = self.news_patterns['date_arabic'].search(text)
        if date_match:
            metadata['has_date'] = True
            metadata['date_info'] = date_match.group().strip()
        
        # Location dateline
        location_match = self.news_patterns['location_dateline'].search(text)
        if location_match:
            metadata['has_location_dateline'] = True
            metadata['location_info'] = location_match.group().strip()
        
        # Attribution patterns
        attribution_matches = self.news_patterns['attribution'].findall(text)
        if attribution_matches:
            metadata['has_attribution'] = True
            metadata['attribution_count'] = len(attribution_matches)
        
        # Source mentions
        if self.news_patterns['source_mention'].search(text):
            metadata['has_source_mention'] = True
        
        # Quotes
        quote_matches = self.news_patterns['quotes'].findall(text)
        if quote_matches:
            metadata['has_quotes'] = True
            metadata['quote_count'] = len(quote_matches)
        
        return metadata
    
    def extract_title_and_body(self, text: str) -> Tuple[str, str]:
        """Attempt to separate title from body content"""
        if not text:
            return "", ""
        
        # Try multiple heuristics
        lines = text.split('\n')
        if len(lines) > 1 and len(lines[0]) < 150:
            # First line might be title if it's reasonably short
            title = lines[0].strip()
            body = '\n'.join(lines[1:]).strip()
            return title, body
        
        # Try by looking for location dateline
        location_match = self.news_patterns['location_dateline'].search(text)
        if location_match:
            # Everything before location dateline could be title
            start = location_match.start()
            if start > 10 and start < 200:
                title = text[:start].strip()
                body = text[start:].strip()
                return title, body
        
        # Fallback: use first sentence as title
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) > 1 and len(sentences[0]) < 200:
            title = sentences[0].strip()
            body = text[len(title):].strip()
            return title, body
        
        # If all else fails, return empty title and full text as body
        return "", text
    
    def clean_text(self, text: str, remove_diacritics: bool = True, 
                   normalize_chars: bool = True) -> str:
        """Complete text cleaning pipeline"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove diacritics
        if remove_diacritics:
            text = self.remove_diacritics(text)
        
        # Normalize characters
        if normalize_chars:
            text = self.normalize_arabic_chars(text)
        
        # Clean whitespace and punctuation
        text = self.clean_whitespace_and_punctuation(text)
        
        return text


class ArticleQualityAnalyzer:
    """Analyze article quality and extract quality indicators"""
    
    def __init__(self, min_length: int = 100, min_words: int = 20):
        self.min_length = min_length
        self.min_words = min_words
        
        # Quality indicators
        self.quality_patterns = {
            'proper_attribution': re.compile(r'(قال|أضاف|أوضح|أكد|صرح)', re.UNICODE),
            'has_quotes': re.compile(r'["""''«»]', re.UNICODE),
            'has_numbers': re.compile(r'\d+'),
            'proper_structure': re.compile(r'[.!?]\s+[ا-ي]', re.UNICODE),
            'has_context': re.compile(r'(بسبب|نتيجة|من أجل|حيث|كما|أيضا)', re.UNICODE)
        }
        
        # Low quality indicators
        self.low_quality_patterns = {
            'very_short': lambda text: len(text) < 100,
            'very_long': lambda text: len(text) > 5000,
            'too_many_numbers': lambda text: len(re.findall(r'\d+', text)) > 20,
            'repetitive': self._check_repetitive,
            'poor_structure': lambda text: len(re.findall(r'[.!?]', text)) < 3
        }
    
    def _check_repetitive(self, text: str) -> bool:
        """Check if text has repetitive patterns"""
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for word repetition
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any word appears more than 10% of the time, it's repetitive
        max_freq = max(word_freq.values())
        return max_freq > len(words) * 0.1
    
    def analyze_quality(self, text: str) -> Dict[str, any]:
        """Comprehensive quality analysis"""
        if not text:
            return {'quality_score': 0, 'issues': ['empty_text']}
        
        quality_features = {}
        quality_issues = []
        
        # Basic metrics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]', text))
        
        quality_features.update({
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': char_count / max(sentence_count, 1),
            'avg_word_length': char_count / max(word_count, 1)
        })
        
        # Check minimum requirements
        if char_count < self.min_length:
            quality_issues.append('too_short')
        
        if word_count < self.min_words:
            quality_issues.append('insufficient_words')
        
        # Check quality patterns
        for pattern_name, pattern in self.quality_patterns.items():
            if pattern.search(text):
                quality_features[pattern_name] = True
            else:
                quality_features[pattern_name] = False
        
        # Check low quality indicators
        for issue_name, checker in self.low_quality_patterns.items():
            if callable(checker):
                if checker(text):
                    quality_issues.append(issue_name)
            else:
                if checker.search(text):
                    quality_issues.append(issue_name)
        
        # Calculate overall quality score
        positive_indicators = sum([
            quality_features.get('proper_attribution', False),
            quality_features.get('has_quotes', False),
            quality_features.get('proper_structure', False),
            quality_features.get('has_context', False),
            char_count >= self.min_length,
            word_count >= self.min_words,
            sentence_count >= 3
        ])
        
        # Penalty for quality issues
        penalty = len(quality_issues) * 0.1
        quality_score = max(0, (positive_indicators / 7.0) - penalty)
        
        return {
            'quality_score': quality_score,
            'quality_features': quality_features,
            'quality_issues': quality_issues,
            'metrics': {
                'char_count': char_count,
                'word_count': word_count,
                'sentence_count': sentence_count
            }
        }


class NewsArticleProcessor:
    """Main processor for Arabic news articles"""
    
    def __init__(self):
        self.cleaner = ArabicTextCleaner()
        self.quality_analyzer = ArticleQualityAnalyzer()
    
    def process_article(self, article_text: str, extract_metadata: bool = True) -> Dict:
        """Process a single article through complete pipeline"""
        
        if not article_text or pd.isna(article_text):
            return {
                'original_text': article_text,
                'cleaned_text': "",
                'title': "",
                'body': "",
                'metadata': {},
                'quality_analysis': {'quality_score': 0, 'quality_issues': ['empty_text']},
                'processing_status': 'failed',
                'error': 'Empty or invalid text'
            }
        
        try:
            # Clean the text
            cleaned_text = self.cleaner.clean_text(article_text)
            
            # Extract title and body
            title, body = self.cleaner.extract_title_and_body(cleaned_text)
            
            # Extract metadata if requested
            metadata = {}
            if extract_metadata:
                metadata = self.cleaner.extract_metadata(cleaned_text)
            
            # Analyze quality
            quality_analysis = self.quality_analyzer.analyze_quality(cleaned_text)
            
            return {
                'original_text': article_text,
                'cleaned_text': cleaned_text,
                'title': title,
                'body': body,
                'metadata': metadata,
                'quality_analysis': quality_analysis,
                'processing_status': 'success',
                'error': None
            }
            
        except Exception as e:
            return {
                'original_text': article_text,
                'cleaned_text': "",
                'title': "",
                'body': "",
                'metadata': {},
                'quality_analysis': {'quality_score': 0, 'quality_issues': ['processing_error']},
                'processing_status': 'failed',
                'error': str(e)
            }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'Story') -> pd.DataFrame:
        """Process entire dataframe of articles"""
        
        results = []
        for idx, row in df.iterrows():
            article_text = row[text_column]
            result = self.process_article(article_text)
            
            # Add original row data
            result.update({
                'original_index': idx,
                'story_id': row.get('StoryId', None),
                'story_date': row.get('StoryDate', None),
                'track_id': row.get('TrackId', None),
                'dist': row.get('dist', None)
            })
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_processing_report(self, processed_df: pd.DataFrame) -> Dict:
        """Generate summary report of processing results"""
        
        total_articles = len(processed_df)
        successful_processing = (processed_df['processing_status'] == 'success').sum()
        failed_processing = total_articles - successful_processing
        
        # Quality score distribution
        quality_scores = processed_df['quality_analysis'].apply(lambda x: x.get('quality_score', 0))
        
        # Common quality issues
        all_issues = []
        for quality_analysis in processed_df['quality_analysis']:
            all_issues.extend(quality_analysis.get('quality_issues', []))
        
        issue_counts = pd.Series(all_issues).value_counts()
        
        report = {
            'processing_summary': {
                'total_articles': total_articles,
                'successful_processing': successful_processing,
                'failed_processing': failed_processing,
                'success_rate': successful_processing / total_articles if total_articles > 0 else 0
            },
            'quality_analysis': {
                'mean_quality_score': quality_scores.mean(),
                'median_quality_score': quality_scores.median(),
                'high_quality_articles': (quality_scores >= 0.7).sum(),
                'medium_quality_articles': ((quality_scores >= 0.4) & (quality_scores < 0.7)).sum(),
                'low_quality_articles': (quality_scores < 0.4).sum()
            },
            'common_issues': issue_counts.head(10).to_dict(),
            'text_statistics': {
                'avg_original_length': processed_df['original_text'].str.len().mean(),
                'avg_cleaned_length': processed_df['cleaned_text'].str.len().mean(),
                'avg_title_length': processed_df['title'].str.len().mean(),
                'articles_with_titles': (processed_df['title'].str.len() > 0).sum()
            }
        }
        
        return report


# Utility functions for testing and validation
def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with sample Arabic text"""
    
    # Sample Arabic news text
    sample_text = """
    الرئيس يؤكد أهمية التعاون الدولي
    القاهرة في 15 سبتمبر /أ ش أ/ أكد الرئيس خلال اجتماعه اليوم أهمية تعزيز التعاون الدولي في مختلف المجالات.
    وقال الرئيس: "إن التعاون الدولي ضروري لمواجهة التحديات المشتركة"، مشيراً إلى أهمية الحوار البناء.
    وأضاف أن الدولة تسعى إلى تطوير علاقاتها مع جميع الدول الصديقة.
    """
    
    processor = NewsArticleProcessor()
    result = processor.process_article(sample_text)
    
    print("Preprocessing Test Results:")
    print("=" * 40)
    print(f"Original length: {len(result['original_text'])}")
    print(f"Cleaned length: {len(result['cleaned_text'])}")
    print(f"Title: {result['title']}")
    print(f"Quality score: {result['quality_analysis']['quality_score']:.2f}")
    print(f"Metadata: {result['metadata']}")
    print(f"Issues: {result['quality_analysis']['quality_issues']}")
    
    return result

if __name__ == "__main__":
    test_preprocessing_pipeline()