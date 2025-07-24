import cv2
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FaceDetectionEvaluator:
    def __init__(self):
        self.results = []
        self.setup_models()
        
    def setup_models(self):
        """Initialize all three face detection models"""
        print("Setting up models...")
        
        # 1. OpenCV DNN Model
        try:
            # Download these files if you don't have them:
            # https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector_uint8.pb
            # https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
            self.dnn_net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb', 
                'opencv_face_detector.pbtxt'
            )
            print("✅ OpenCV DNN model loaded")
        except Exception as e:
            print(f"❌ Failed to load DNN model: {e}")
            print("Download the model files from OpenCV GitHub repository")
            self.dnn_net = None
        
        # 2. MTCNN Model
        try:
            from mtcnn import MTCNN
            self.mtcnn_detector = MTCNN()
            print("✅ MTCNN model loaded")
        except ImportError:
            print("❌ MTCNN not installed. Install with: pip install mtcnn")
            self.mtcnn_detector = None
        except Exception as e:
            print(f"❌ Failed to load MTCNN: {e}")
            self.mtcnn_detector = None
        
        # 3. YOLOv8 Model
        try:
            from ultralytics import YOLO
            # Try to load a face-specific model first, fallback to general model
            try:
                self.yolo_model = YOLO('yolov8n-face.pt')
                print("✅ YOLOv8 face model loaded")
            except:
                self.yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLOv8 general model loaded (will detect persons)")
        except ImportError:
            print("❌ Ultralytics not installed. Install with: pip install ultralytics")
            self.yolo_model = None
        except Exception as e:
            print(f"❌ Failed to load YOLOv8: {e}")
            self.yolo_model = None
    
    def detect_faces_dnn(self, image):
        """Face detection using OpenCV DNN"""
        if self.dnn_net is None:
            return [], 0
            
        start_time = time.time()
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        confidence_threshold = 0.5
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append([x1, y1, x2-x1, y2-y1, confidence])
        
        processing_time = time.time() - start_time
        return faces, processing_time
    
    def detect_faces_mtcnn(self, image):
        """Face detection using MTCNN"""
        if self.mtcnn_detector is None:
            return [], 0
            
        start_time = time.time()
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.mtcnn_detector.detect_faces(rgb_image)
        
        faces = []
        for face in result:
            x, y, w, h = face['box']
            confidence = face['confidence']
            faces.append([x, y, w, h, confidence])
        
        processing_time = time.time() - start_time
        return faces, processing_time
    
    def detect_faces_yolo(self, image):
        """Face detection using YOLOv8"""
        if self.yolo_model is None:
            return [], 0
            
        start_time = time.time()
        
        results = self.yolo_model(image, verbose=False)
        
        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0) or face detection
                    class_id = int(box.cls[0])
                    if class_id == 0 or len(self.yolo_model.names) == 1:  # person or face
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        faces.append([int(x1), int(y1), int(x2-x1), int(y2-y1), confidence])
        
        processing_time = time.time() - start_time
        return faces, processing_time
    
    def evaluate_single_image(self, image_path, expected_faces=1):
        """Evaluate all models on a single image"""
        print(f"Evaluating: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        results = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'expected_faces': expected_faces
        }
        
        # Test DNN
        try:
            dnn_faces, dnn_time = self.detect_faces_dnn(image)
            results['dnn_faces_count'] = len(dnn_faces)
            results['dnn_processing_time'] = round(dnn_time, 4)
            results['dnn_accuracy'] = min(len(dnn_faces) / max(expected_faces, 1), 1.0)
            results['dnn_success'] = True
            print(f"  DNN: {len(dnn_faces)} faces, {dnn_time:.4f}s")
        except Exception as e:
            results['dnn_success'] = False
            results['dnn_error'] = str(e)
            print(f"  DNN: Failed - {e}")
        
        # Test MTCNN
        try:
            mtcnn_faces, mtcnn_time = self.detect_faces_mtcnn(image)
            results['mtcnn_faces_count'] = len(mtcnn_faces)
            results['mtcnn_processing_time'] = round(mtcnn_time, 4)
            results['mtcnn_accuracy'] = min(len(mtcnn_faces) / max(expected_faces, 1), 1.0)
            results['mtcnn_success'] = True
            print(f"  MTCNN: {len(mtcnn_faces)} faces, {mtcnn_time:.4f}s")
        except Exception as e:
            results['mtcnn_success'] = False
            results['mtcnn_error'] = str(e)
            print(f"  MTCNN: Failed - {e}")
        
        # Test YOLOv8
        try:
            yolo_faces, yolo_time = self.detect_faces_yolo(image)
            results['yolo_faces_count'] = len(yolo_faces)
            results['yolo_processing_time'] = round(yolo_time, 4)
            results['yolo_accuracy'] = min(len(yolo_faces) / max(expected_faces, 1), 1.0)
            results['yolo_success'] = True
            print(f"  YOLOv8: {len(yolo_faces)} faces, {yolo_time:.4f}s")
        except Exception as e:
            results['yolo_success'] = False
            results['yolo_error'] = str(e)
            print(f"  YOLOv8: Failed - {e}")
        
        return results
    
    def evaluate_dataset(self, test_images_folder="test_images", expected_faces=1):
        """Evaluate all images in the test dataset"""
        print("🔍 Starting face detection evaluation...")
        
        all_results = []
        lighting_conditions = ['normal_lighting', 'low_lighting', 'bright_lighting']
        
        for condition in lighting_conditions:
            condition_path = os.path.join(test_images_folder, condition)
            
            if not os.path.exists(condition_path):
                print(f"⚠️  Folder not found: {condition_path}")
                continue
                
            print(f"\n📁 Testing {condition.replace('_', ' ').title()}...")
            
            image_files = [f for f in os.listdir(condition_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"  No images found in {condition_path}")
                continue
            
            for image_file in sorted(image_files):
                image_path = os.path.join(condition_path, image_file)
                result = self.evaluate_single_image(image_path, expected_faces)
                
                if result:
                    result['lighting_condition'] = condition.replace('_lighting', '')
                    all_results.append(result)
        
        self.results = all_results
        return all_results
    
    def generate_summary_stats(self):
        """Generate summary statistics"""
        if not self.results:
            print("No results to analyze")
            return None
        
        df = pd.DataFrame(self.results)
        summary = {}
        
        models = ['dnn', 'mtcnn', 'yolo']
        conditions = df['lighting_condition'].unique()
        
        print("\n📊 SUMMARY STATISTICS")
        print("=" * 50)
        
        for model in models:
            if f'{model}_success' in df.columns:
                model_data = df[df[f'{model}_success'] == True]
                
                if not model_data.empty:
                    summary[f'{model}_avg_time'] = model_data[f'{model}_processing_time'].mean()
                    summary[f'{model}_std_time'] = model_data[f'{model}_processing_time'].std()
                    summary[f'{model}_avg_accuracy'] = model_data[f'{model}_accuracy'].mean()
                    summary[f'{model}_success_rate'] = len(model_data) / len(df)
                    
                    print(f"\n{model.upper()} Performance:")
                    print(f"  Average Processing Time: {summary[f'{model}_avg_time']:.4f}s")
                    print(f"  Accuracy: {summary[f'{model}_avg_accuracy']:.2%}")
                    print(f"  Success Rate: {summary[f'{model}_success_rate']:.2%}")
                    
                    # Performance by lighting condition
                    for condition in conditions:
                        condition_data = model_data[model_data['lighting_condition'] == condition]
                        if not condition_data.empty:
                            avg_time = condition_data[f'{model}_processing_time'].mean()
                            avg_acc = condition_data[f'{model}_accuracy'].mean()
                            print(f"    {condition.title()}: {avg_time:.4f}s, {avg_acc:.2%} accuracy")
        
        return summary
    
    def save_results(self, filename=None):
        """Save results to Excel file"""
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_detection_results_{timestamp}.xlsx"
        
        df = pd.DataFrame(self.results)
        
        # Create summary statistics
        summary_stats = self.generate_summary_stats()
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Detailed results
                df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Summary statistics
                if summary_stats:
                    summary_df = pd.DataFrame([summary_stats])
                    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Performance by condition
                if not df.empty:
                    condition_summary = []
                    for condition in df['lighting_condition'].unique():
                        condition_data = df[df['lighting_condition'] == condition]
                        for model in ['dnn', 'mtcnn', 'yolo']:
                            if f'{model}_success' in df.columns:
                                model_data = condition_data[condition_data[f'{model}_success'] == True]
                                if not model_data.empty:
                                    condition_summary.append({
                                        'lighting_condition': condition,
                                        'algorithm': model.upper(),
                                        'avg_time': model_data[f'{model}_processing_time'].mean(),
                                        'avg_accuracy': model_data[f'{model}_accuracy'].mean(),
                                        'images_tested': len(model_data)
                                    })
                    
                    if condition_summary:
                        condition_df = pd.DataFrame(condition_summary)
                        condition_df.to_excel(writer, sheet_name='Performance_by_Condition', index=False)
            
            print(f"✅ Results saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None

    def create_visualizations(self, save_path_prefix="face_detection_analysis"):
        """Create performance comparison visualizations as separate images"""
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        # Set up the plot style
        plt.style.use('default')
        
        saved_files = []
        
        # 1. Processing Time Comparison
        plt.figure(figsize=(10, 6))
        time_data = []
        for model in ['dnn', 'mtcnn', 'yolo']:
            if f'{model}_success' in df.columns:
                model_data = df[df[f'{model}_success'] == True]
                for _, row in model_data.iterrows():
                    time_data.append({
                        'Algorithm': model.upper(),
                        'Processing_Time': row[f'{model}_processing_time'],
                        'Condition': row['lighting_condition'].title()
                    })
        
        if time_data:
            time_df = pd.DataFrame(time_data)
            sns.boxplot(data=time_df, x='Algorithm', y='Processing_Time')
            plt.title('Processing Time Comparison', fontsize=14, fontweight='bold')
            plt.ylabel('Time (seconds)')
            plt.xlabel('Algorithm')
            
            filename1 = f"{save_path_prefix}_processing_time.png"
            plt.savefig(filename1, dpi=300, bbox_inches='tight')
            saved_files.append(filename1)
            print(f"✅ Processing time chart saved to: {filename1}")
        
        plt.close()
        
        # 2. Accuracy by Lighting Condition
        plt.figure(figsize=(10, 6))
        accuracy_data = []
        for model in ['dnn', 'mtcnn', 'yolo']:
            if f'{model}_success' in df.columns:
                for condition in df['lighting_condition'].unique():
                    condition_data = df[(df['lighting_condition'] == condition) & 
                                      (df[f'{model}_success'] == True)]
                    if not condition_data.empty:
                        avg_accuracy = condition_data[f'{model}_accuracy'].mean()
                        accuracy_data.append({
                            'Algorithm': model.upper(),
                            'Accuracy': avg_accuracy,
                            'Condition': condition.title()
                        })
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            sns.barplot(data=accuracy_df, x='Condition', y='Accuracy', hue='Algorithm')
            plt.title('Accuracy by Lighting Condition', fontsize=14, fontweight='bold')
            plt.ylabel('Accuracy')
            plt.xlabel('Lighting Condition')
            plt.ylim(0, 1)
            plt.legend(title='Algorithm')
            
            filename2 = f"{save_path_prefix}_accuracy_by_condition.png"
            plt.savefig(filename2, dpi=300, bbox_inches='tight')
            saved_files.append(filename2)
            print(f"✅ Accuracy by condition chart saved to: {filename2}")
        
        plt.close()
        
        # 3. Average Processing Time by Condition
        plt.figure(figsize=(10, 6))
        if time_data:
            avg_time_by_condition = time_df.groupby(['Algorithm', 'Condition'])['Processing_Time'].mean().reset_index()
            sns.barplot(data=avg_time_by_condition, x='Condition', y='Processing_Time', hue='Algorithm')
            plt.title('Average Processing Time by Condition', fontsize=14, fontweight='bold')
            plt.ylabel('Time (seconds)')
            plt.xlabel('Lighting Condition')
            plt.legend(title='Algorithm')
            
            filename3 = f"{save_path_prefix}_time_by_condition.png"
            plt.savefig(filename3, dpi=300, bbox_inches='tight')
            saved_files.append(filename3)
            print(f"✅ Processing time by condition chart saved to: {filename3}")
        
        plt.close()
        
        # 4. Overall Performance Summary
        plt.figure(figsize=(12, 6))
        summary_data = []
        for model in ['dnn', 'mtcnn', 'yolo']:
            if f'{model}_success' in df.columns:
                model_data = df[df[f'{model}_success'] == True]
                if not model_data.empty:
                    summary_data.append({
                        'Algorithm': model.upper(),
                        'Avg_Time': model_data[f'{model}_processing_time'].mean(),
                        'Avg_Accuracy': model_data[f'{model}_accuracy'].mean(),
                        'Success_Rate': len(model_data) / len(df)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            x_pos = range(len(summary_df))
            
            # Create twin axis for accuracy
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            bars1 = ax1.bar([x - 0.2 for x in x_pos], summary_df['Avg_Time'], 
                           width=0.4, label='Avg Time (s)', color='skyblue', alpha=0.7)
            bars2 = ax2.bar([x + 0.2 for x in x_pos], summary_df['Avg_Accuracy'], 
                            width=0.4, label='Accuracy', color='lightcoral', alpha=0.7)
            
            ax1.set_xlabel('Algorithm')
            ax1.set_ylabel('Average Time (seconds)', color='blue')
            ax2.set_ylabel('Accuracy', color='red')
            ax1.set_title('Overall Performance Summary', fontsize=14, fontweight='bold', pad=20)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(summary_df['Algorithm'])
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.001,
                        f'{summary_df.iloc[i]["Avg_Time"]:.3f}s', 
                        ha='center', va='bottom', fontsize=10)
                ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                         f'{summary_df.iloc[i]["Avg_Accuracy"]:.2%}', 
                         ha='center', va='bottom', fontsize=10)
            
            # Create custom legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            filename4 = f"{save_path_prefix}_overall_summary.png"
            plt.savefig(filename4, dpi=300, bbox_inches='tight')
            saved_files.append(filename4)
            print(f"✅ Overall summary chart saved to: {filename4}")
        
        plt.close()
        
        print(f"\n🎉 All visualizations saved! Generated {len(saved_files)} separate images:")
        for file in saved_files:
            print(f"  📈 {file}")
        
        return saved_files

# Main execution function
def run_evaluation():
    """Run the complete face detection evaluation"""
    
    print("🚀 Face Detection Algorithm Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = FaceDetectionEvaluator()
    
    # Check if test images exist
    if not os.path.exists("test_images"):
        print("❌ test_images folder not found!")
        print("Please run the image capture script first to collect test images.")
        return
    
    # Run evaluation
    results = evaluator.evaluate_dataset(expected_faces=1)
    
    if not results:
        print("❌ No results obtained. Check your test images and model setup.")
        return
    
    print(f"\n✅ Evaluation completed! Processed {len(results)} images.")
    
    # Generate summary statistics
    evaluator.generate_summary_stats()
    
    # Save results to Excel
    excel_file = evaluator.save_results()
    
    # Create separate visualizations
    chart_files = evaluator.create_visualizations()
    
    print("\n🎉 Evaluation complete!")
    print("Check the generated files:")
    if excel_file:
        print(f"  📊 Results: {excel_file}")
    print("  📈 Charts generated as separate files (see above)")
    
    return evaluator, results

if __name__ == "__main__":
    # Run the evaluation
    evaluator, results = run_evaluation()