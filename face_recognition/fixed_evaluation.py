import numpy as np
import mysql.connector
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import base64
import os
import json
from datetime import datetime

class HybridFYPEvaluator:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "payment_facial"
        }
        self.results = {
            'embedding_evaluation': {},
            'live_testing': [],
            'combined_metrics': {}
        }
    
    def connect_db(self):
        """Connect to database"""
        return mysql.connector.connect(**self.db_config)
    
    def load_embeddings_from_db(self):
        """Load embeddings from database like your code"""
        print("[INFO] Loading embeddings from database...")
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, embedding FROM face_embeddings")
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        X, y = [], []
        for user_id, emb_blob in data:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            X.append(emb)
            y.append(user_id)
        
        return np.array(X), np.array(y)
    
    def evaluate_embeddings(self):
        """Evaluate using database embeddings (your existing approach)"""
        print("\n🧪 PART 1: Database Embedding Evaluation")
        print("="*50)
        
        X, y = self.load_embeddings_from_db()
        
        if len(np.unique(y)) < 2:
            print("❌ Not enough users for evaluation.")
            return None
        
        print(f"[INFO] Total Samples: {len(X)}, Unique Users: {len(set(y))}")
        
        # Show distribution of samples per user
        unique, counts = np.unique(y, return_counts=True)
        print(f"[INFO] Samples per user: {dict(zip(unique, counts))}")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42)
        
        print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # Train SVM
        print("[INFO] Training SVM model...")
        clf = svm.SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results['embedding_evaluation'] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(X),
            'unique_users': len(set(y)),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'samples_per_user': dict(zip(unique.tolist(), counts.tolist()))
        }
        
        # Print results
        print("\n✅ Database Embedding Evaluation Results:")
        print(f"   Accuracy  : {acc:.4f} ({acc:.1%})")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1-score  : {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return clf
    
    def image_to_base64(self, image_path):
        """Convert image to base64"""
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def test_live_recognition(self, image_path, expected_user=None, test_type="legitimate"):
        """Test live recognition via API"""
        try:
            image_b64 = self.image_to_base64(image_path)
            
            payload = {
                'image_base64': image_b64,
                'user_id': expected_user
            }
            
            response = requests.post(f"{self.api_url}/recognize", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Determine if test passed
                if test_type == "legitimate":
                    passed = (result.get('status') == 'success' and 
                             result.get('user_id') == expected_user)
                elif test_type == "unknown":
                    passed = result.get('status') == 'failed'
                
                return {
                    'image': os.path.basename(image_path),
                    'test_type': test_type,
                    'expected_user': expected_user,
                    'recognized_user': result.get('user_id'),
                    'similarity': result.get('similarity', 0),
                    'status': result.get('status'),
                    'passed': passed,
                    'message': result.get('message', '')
                }
            else:
                return {
                    'image': os.path.basename(image_path),
                    'test_type': test_type,
                    'expected_user': expected_user,
                    'error': f"HTTP {response.status_code}",
                    'passed': False
                }
        except Exception as e:
            return {
                'image': os.path.basename(image_path),
                'test_type': test_type,
                'expected_user': expected_user,
                'error': str(e),
                'passed': False
            }
    
    def get_registered_users(self):
        """Get list of registered users from database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM face_embeddings")
        users = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return users
    
    def evaluate_live_testing(self, test_data_path):
        """Evaluate with live images"""
        print("\n🎥 PART 2: Live Image Testing")
        print("="*50)
        
        registered_users = self.get_registered_users()
        print(f"[INFO] Found {len(registered_users)} registered users: {registered_users}")
        
        # Test registered users
        registered_path = os.path.join(test_data_path, "registered_users")
        if os.path.exists(registered_path):
            print(f"\n📋 Testing Registered Users...")
            for img_file in os.listdir(registered_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract user ID from filename or use first registered user
                    user_id = img_file.split('_')[0] if '_' in img_file else registered_users[0]
                    if user_id not in registered_users:
                        user_id = registered_users[0]  # Fallback to first user
                    
                    img_path = os.path.join(registered_path, img_file)
                    result = self.test_live_recognition(img_path, user_id, "legitimate")
                    
                    if result:
                        self.results['live_testing'].append(result)
                        status = "✅ PASS" if result['passed'] else "❌ FAIL"
                        sim = result.get('similarity', 0)
                        print(f"   {img_file}: {status} (similarity: {sim:.3f})")
        
        # Test unknown people
        unknown_path = os.path.join(test_data_path, "unknown_people")
        if os.path.exists(unknown_path):
            print(f"\n🚫 Testing Unknown People...")
            for img_file in os.listdir(unknown_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(unknown_path, img_file)
                    result = self.test_live_recognition(img_path, None, "unknown")
                    
                    if result:
                        self.results['live_testing'].append(result)
                        status = "✅ PASS" if result['passed'] else "❌ FAIL"
                        sim = result.get('similarity', 0)
                        print(f"   {img_file}: {status} (similarity: {sim:.3f})")
    
    def calculate_live_metrics(self):
        """Calculate metrics for live testing"""
        if not self.results['live_testing']:
            return {}
        
        legitimate_tests = [r for r in self.results['live_testing'] if r['test_type'] == 'legitimate']
        unknown_tests = [r for r in self.results['live_testing'] if r['test_type'] == 'unknown']
        
        legitimate_accuracy = sum(1 for r in legitimate_tests if r['passed']) / len(legitimate_tests) if legitimate_tests else 0
        unknown_accuracy = sum(1 for r in unknown_tests if r['passed']) / len(unknown_tests) if unknown_tests else 0
        
        total_passed = sum(1 for r in self.results['live_testing'] if r['passed'])
        overall_accuracy = total_passed / len(self.results['live_testing'])
        
        return {
            'overall_accuracy': overall_accuracy,
            'legitimate_accuracy': legitimate_accuracy,
            'unknown_rejection_rate': unknown_accuracy,
            'total_tests': len(self.results['live_testing']),
            'legitimate_tests': len(legitimate_tests),
            'unknown_tests': len(unknown_tests)
        }
    
    def create_visualizations(self):
        """Create simple visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Database Evaluation Metrics
        if self.results['embedding_evaluation']:
            db_metrics = self.results['embedding_evaluation']
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metrics_values = [db_metrics['accuracy'], db_metrics['precision'], 
                            db_metrics['recall'], db_metrics['f1_score']]
            
            bars1 = ax1.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax1.set_title('Database Embedding Evaluation')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars1, metrics_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Confusion Matrix
        if self.results['embedding_evaluation']:
            cm = np.array(self.results['embedding_evaluation']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('Confusion Matrix (Database)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
        
        # 3. Live Testing Results
        live_metrics = self.calculate_live_metrics()
        if live_metrics:
            categories = ['Overall\nAccuracy', 'User\nRecognition', 'Unknown\nRejection']
            values = [live_metrics['overall_accuracy'], 
                     live_metrics['legitimate_accuracy'],
                     live_metrics['unknown_rejection_rate']]
            
            bars3 = ax3.bar(categories, values, color=['#9467bd', '#8c564b', '#e377c2'])
            ax3.set_title('Live Testing Results')
            ax3.set_ylabel('Accuracy Rate')
            ax3.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars3, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1%}', ha='center', va='bottom')
        
        # 4. Similarity Distribution
        if self.results['live_testing']:
            similarities = [r.get('similarity', 0) for r in self.results['live_testing'] if 'similarity' in r]
            legitimate_sims = [r['similarity'] for r in self.results['live_testing'] 
                              if r['test_type'] == 'legitimate' and 'similarity' in r]
            unknown_sims = [r['similarity'] for r in self.results['live_testing'] 
                           if r['test_type'] == 'unknown' and 'similarity' in r]
            
            if legitimate_sims:
                ax4.hist(legitimate_sims, alpha=0.7, label='Legitimate Users', bins=10, color='green')
            if unknown_sims:
                ax4.hist(unknown_sims, alpha=0.7, label='Unknown People', bins=10, color='red')
            
            ax4.axvline(x=0.40, color='black', linestyle='--', label='Threshold (0.40)')
            ax4.set_xlabel('Similarity Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Similarity Score Distribution')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('fyp_hybrid_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Visualizations saved as 'fyp_hybrid_evaluation.png'")
    
    def generate_final_report(self):
        """Generate comprehensive FYP report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE FYP EVALUATION REPORT")
        print("="*60)
        
        # Database Evaluation Results
        if self.results['embedding_evaluation']:
            db_eval = self.results['embedding_evaluation']
            print(f"\n🗄️ DATABASE EMBEDDING EVALUATION:")
            print(f"   Total Embeddings: {db_eval['total_samples']}")
            print(f"   Unique Users: {db_eval['unique_users']}")
            print(f"   Accuracy: {db_eval['accuracy']:.1%}")
            print(f"   Precision: {db_eval['precision']:.3f}")
            print(f"   Recall: {db_eval['recall']:.3f}")
            print(f"   F1-Score: {db_eval['f1_score']:.3f}")
        
        # Live Testing Results
        live_metrics = self.calculate_live_metrics()
        if live_metrics:
            print(f"\n🎥 LIVE IMAGE TESTING:")
            print(f"   Overall Accuracy: {live_metrics['overall_accuracy']:.1%}")
            print(f"   User Recognition: {live_metrics['legitimate_accuracy']:.1%}")
            print(f"   Unknown Rejection: {live_metrics['unknown_rejection_rate']:.1%}")
            print(f"   Total Tests: {live_metrics['total_tests']}")
        
        # Combined Analysis
        if self.results['embedding_evaluation'] and live_metrics:
            avg_accuracy = (self.results['embedding_evaluation']['accuracy'] + 
                           live_metrics['overall_accuracy']) / 2
            print(f"\n📈 COMBINED SYSTEM PERFORMANCE:")
            print(f"   Average Accuracy: {avg_accuracy:.1%}")
            
            # Objective Assessment
            print(f"\n🎯 OBJECTIVE ACHIEVEMENT:")
            print(f"   ✓ Reliable Identification: {avg_accuracy >= 0.80}")
            print(f"   ✓ High Accuracy: {avg_accuracy >= 0.85}")
            print(f"   ✓ Spoofing Resistance: {live_metrics.get('unknown_rejection_rate', 0) >= 0.70}")
        
        # Save detailed report
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'database_evaluation': self.results['embedding_evaluation'],
            'live_testing': {
                'metrics': live_metrics,
                'detailed_results': self.results['live_testing']
            }
        }
        
        with open('hybrid_fyp_evaluation.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Detailed report saved to 'hybrid_fyp_evaluation.json'")
        print("="*60)
        
        return report
    
    def run_full_evaluation(self, test_data_path=None):
        """Run complete hybrid evaluation"""
        print("🚀 Starting Hybrid FYP Evaluation")
        print("Combining Database Embeddings + Live Testing")
        
        # Part 1: Database embedding evaluation
        self.evaluate_embeddings()
        
        # Part 2: Live testing (if test data provided)
        if test_data_path and os.path.exists(test_data_path):
            self.evaluate_live_testing(test_data_path)
        else:
            print("\n⚠️  No test data path provided - skipping live testing")
            print("   Create folder structure: test_data/registered_users/ and test_data/unknown_people/")
        
        # Generate visualizations
        self.create_visualizations()
        
        # Generate final report
        report = self.generate_final_report()
        
        return report

# Usage example
if __name__ == "__main__":
    evaluator = HybridFYPEvaluator()
    
    # Run evaluation
    # If you have test images, put them in: test_data/registered_users/ and test_data/unknown_people/
    report = evaluator.run_full_evaluation("test_data")  # Remove "test_data" if you don't have test images