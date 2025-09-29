"""
Week 7: API Testing Script for NewsAI Backend
Test all API endpoints and functionality
"""

import requests
import json
import time
from typing import Dict, List

class NewsAIAPITester:
    """Test suite for NewsAI FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):  # Changed from 8000 to 8001
        self.base_url = base_url
        self.test_results = []
        
    def test_health_check(self) -> Dict:
        """Test health check endpoint"""
        
        try:
            response = requests.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "health_check",
                    "status": "PASS",
                    "message": f"API healthy - Status: {data['status']}",
                    "details": data
                }
            else:
                return {
                    "test": "health_check",
                    "status": "FAIL",
                    "message": f"Health check failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "test": "health_check",
                "status": "FAIL", 
                "message": f"Connection error: {str(e)}"
            }
    
    def test_classification(self) -> Dict:
        """Test article classification endpoint"""
        
        try:
            test_article = {
                "text": "الرياض في 28 سبتمبر /واس/ أكد وزير الخارجية أهمية التعاون الدولي مع الشركاء في المنطقة.",
                "article_id": "test_001"
            }
            
            response = requests.post(
                f"{self.base_url}/classify",
                json=test_article
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "classification",
                    "status": "PASS",
                    "message": f"Classification: {data['decision']} (confidence: {data['confidence']:.2f})",
                    "details": {
                        "decision": data['decision'],
                        "confidence": data['confidence'],
                        "processing_time": data['processing_time']
                    }
                }
            else:
                return {
                    "test": "classification",
                    "status": "FAIL",
                    "message": f"Classification failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "test": "classification",
                "status": "FAIL",
                "message": f"Classification error: {str(e)}"
            }
    
    def test_editorial(self) -> Dict:
        """Test editorial endpoint"""
        
        try:
            test_article = {
                "text": "أكد وزير الخارجية أهمية التعاون الدولي مع الشركاء.",
                "article_type": "general",
                "article_id": "edit_test_001"
            }
            
            response = requests.post(
                f"{self.base_url}/edit",
                json=test_article
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "editorial",
                    "status": "PASS",
                    "message": f"Editorial completed - Quality: {data['quality_score']:.2f}",
                    "details": {
                        "changes_count": len(data['changes_applied']),
                        "quality_score": data['quality_score'],
                        "processing_time": data['processing_time']
                    }
                }
            else:
                return {
                    "test": "editorial",
                    "status": "FAIL",
                    "message": f"Editorial failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "test": "editorial", 
                "status": "FAIL",
                "message": f"Editorial error: {str(e)}"
            }
    
    def test_complete_workflow(self) -> Dict:
        """Test complete processing workflow"""
        
        try:
            test_article = {
                "text": "الرياض في 28 سبتمبر /واس/ وزير الصحة يعلن عن برنامج جديد للوقاية.",
                "article_id": "workflow_test_001"
            }
            
            response = requests.post(
                f"{self.base_url}/process",
                json=test_article
            )
            
            if response.status_code == 200:
                data = response.json()
                classification = data['classification']
                editorial = data.get('editorial')
                
                message = f"Workflow: {classification['decision']} (confidence: {classification['confidence']:.2f})"
                if editorial:
                    message += f" + Edited (quality: {editorial['quality_score']:.2f})"
                
                return {
                    "test": "complete_workflow",
                    "status": "PASS",
                    "message": message,
                    "details": {
                        "classification_decision": classification['decision'],
                        "classification_confidence": classification['confidence'],
                        "editorial_applied": editorial is not None,
                        "total_time": data['total_processing_time']
                    }
                }
            else:
                return {
                    "test": "complete_workflow",
                    "status": "FAIL", 
                    "message": f"Workflow failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "test": "complete_workflow",
                "status": "FAIL",
                "message": f"Workflow error: {str(e)}"
            }
    
    def test_batch_processing(self) -> Dict:
        """Test batch processing functionality"""
        
        try:
            test_articles = [
                {"text": "خبر اقتصادي مهم من الرياض.", "article_id": "batch_001"},
                {"text": "إعلان وزارة الصحة عن إجراءات جديدة.", "article_id": "batch_002"},
                {"text": "مؤتمر تقني في جدة اليوم.", "article_id": "batch_003"}
            ]
            
            # Create batch job
            response = requests.post(
                f"{self.base_url}/classify/batch",
                json=test_articles
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data['job_id']
                
                # Wait for processing
                time.sleep(5)
                
                # Check job status
                status_response = requests.get(f"{self.base_url}/classify/batch/{job_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    return {
                        "test": "batch_processing",
                        "status": "PASS",
                        "message": f"Batch job {status_data['status']} - {status_data['processed_articles']}/{status_data['total_articles']} articles",
                        "details": {
                            "job_id": job_id,
                            "job_status": status_data['status'],
                            "total_articles": status_data['total_articles'],
                            "processed_articles": status_data['processed_articles']
                        }
                    }
                else:
                    return {
                        "test": "batch_processing",
                        "status": "FAIL",
                        "message": f"Batch status check failed: {status_response.status_code}"
                    }
            else:
                return {
                    "test": "batch_processing", 
                    "status": "FAIL",
                    "message": f"Batch creation failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "test": "batch_processing",
                "status": "FAIL", 
                "message": f"Batch processing error: {str(e)}"
            }
    
    def test_analytics(self) -> Dict:
        """Test analytics endpoints"""
        
        try:
            response = requests.get(f"{self.base_url}/analytics/summary")
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "analytics",
                    "status": "PASS",
                    "message": f"Analytics working - {data['total_jobs']} jobs tracked",
                    "details": data
                }
            else:
                return {
                    "test": "analytics",
                    "status": "FAIL",
                    "message": f"Analytics failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "test": "analytics",
                "status": "FAIL",
                "message": f"Analytics error: {str(e)}"
            }
    
    def run_all_tests(self) -> None:
        """Run all API tests"""
        
        print("=== WEEK 7: API TESTING ===")
        print("Testing NewsAI FastAPI Backend...\n")
        
        tests = [
            self.test_health_check,
            self.test_classification,
            self.test_editorial,
            self.test_complete_workflow,
            self.test_batch_processing,
            self.test_analytics
        ]
        
        for test_func in tests:
            print(f"Running {test_func.__name__}...")
            result = test_func()
            self.test_results.append(result)
            
            status_emoji = "✅" if result['status'] == 'PASS' else "❌"
            print(f"  {status_emoji} {result['message']}")
            
            if result['status'] == 'FAIL':
                print(f"     Details: {result.get('message', 'No details')}")
            
            time.sleep(1)  # Brief pause between tests
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary"""
        
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        total_tests = len(self.test_results)
        
        print(f"\n=== WEEK 7 API TEST SUMMARY ===")
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {passed_tests/total_tests:.1%}")
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print("Week 7 Status: COMPLETE - API ready for production")
        else:
            print("Week 7 Status: NEEDS WORK - Some endpoints failing")
        
        print(f"\nDetailed results:")
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"  {status_icon} {result['test']}: {result['status']}")

def run_week7_api_tests():
    """Run Week 7 API tests"""
    
    print("Starting NewsAI API tests...")
    print("Make sure the API server is running on http://localhost:8000")
    print("If not running, start it with: python src/api/main.py")
    print()
    
    # Wait a moment for user to start server if needed
    input("Press Enter when API server is ready...")
    
    tester = NewsAIAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    import uvicorn
    
    print("Starting NewsAI API server...")
    uvicorn.run(
        "src.api.main:app",  # Changed from app to import string
        host="0.0.0.0",
        port=8000,
        reload=False,  # Changed from True to False
        log_level="info"
    )