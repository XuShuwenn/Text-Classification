#!/usr/bin/env python3
"""
IMDB文本分类项目主脚本
支持训练、评估和推理功能
"""

import argparse
import sys
import os

def train_models():
    """训练所有模型"""
    print("Starting training for all models...")
    from train import main as train_main
    train_main()

def evaluate_models():
    """评估所有模型"""
    print("Starting evaluation for all models...")
    from evaluate import main as eval_main
    eval_main()

def run_inference(model_type, text=None, interactive=False):
    """运行推理"""
    from scripts.inference import SentimentPredictor
    
    try:
        predictor = SentimentPredictor(model_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using the train command")
        return
    
    if interactive:
        print(f"Sentiment Analysis with {model_type.upper()}")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            text = input("\nEnter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = predictor.predict(text)
            
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
    
    elif text:
        result = predictor.predict(text)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    else:
        # 示例文本
        sample_texts = [
            "This movie is absolutely amazing! I loved every minute of it.",
            "Terrible film, waste of time and money.",
            "The movie was okay, nothing special but not bad either."
        ]
        
        print(f"Running sample predictions with {model_type.upper()}:")
        print("-" * 50)
        
        for text in sample_texts:
            result = predictor.predict(text)
            print(f"Text: {text}")
            print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
            print()

def check_environment():
    """检查环境和依赖"""
    print("Checking environment...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    
    # 检查必要的包
    required_packages = [
        'torch', 'transformers', 'datasets', 'sklearn', 
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing packages:", ', '.join(missing_packages))
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("Environment check passed!")
    return True

def setup_project():
    """设置项目目录"""
    from config import PATHS
    
    print("Setting up project directories...")
    for name, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    print("Project setup completed!")

def main():
    parser = argparse.ArgumentParser(description='IMDB Text Classification Project')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 检查环境命令
    subparsers.add_parser('check', help='Check environment and dependencies')
    
    # 设置项目命令
    subparsers.add_parser('setup', help='Setup project directories')
    
    # 训练命令
    subparsers.add_parser('train', help='Train all models')
    
    # 评估命令
    subparsers.add_parser('evaluate', help='Evaluate all models')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', type=str, default='bert',
                             choices=['textcnn', 'lstm', 'bert'],
                             help='Model to use for inference')
    infer_parser.add_argument('--text', type=str, help='Text to analyze')
    infer_parser.add_argument('--interactive', action='store_true',
                             help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_environment()
    
    elif args.command == 'setup':
        setup_project()
    
    elif args.command == 'train':
        if not check_environment():
            return
        setup_project()
        train_models()
    
    elif args.command == 'evaluate':
        if not check_environment():
            return
        evaluate_models()
    
    elif args.command == 'infer':
        if not check_environment():
            return
        run_inference(args.model, args.text, args.interactive)
    
    else:
        # 默认显示帮助
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py check          # Check environment")
        print("  python main.py setup          # Setup project")
        print("  python main.py train          # Train all models")
        print("  python main.py evaluate       # Evaluate models")
        print("  python main.py infer --model bert --interactive")

if __name__ == "__main__":
    main()
