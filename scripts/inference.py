import torch
import numpy as np
from transformers import AutoTokenizer
import os
import argparse

from config import DEVICE, DATA_CONFIG
from models import get_model
from data_processor import IMDBDataProcessor


class SentimentPredictor:
    """情感分析预测器"""
    
    def __init__(self, model_type, model_path=None):
        self.model_type = model_type
        self.device = DEVICE
        
        # 加载模型
        if model_path is None:
            from config import PATHS
            model_dir = os.path.join(PATHS['model_dir'], model_type)
            model_path = os.path.join(model_dir, 'best_model.pt')
        
        self.model = self._load_model(model_path)
        
        # 初始化处理器
        if model_type == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.processor = IMDBDataProcessor()
            # 这里需要加载词汇表，实际使用时应该保存并加载词汇表
            # 为了演示，我们使用一个简单的词汇表
            self._build_simple_vocab()
    
    def _load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        model_config = checkpoint.get('model_config')
        model = get_model(self.model_type, model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Loaded {self.model_type} model")
        return model
    
    def _build_simple_vocab(self):
        """构建简单词汇表（用于演示）"""
        # 在实际使用中，应该保存并加载训练时的词汇表
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'movie', 'film', 'good', 'bad', 'great', 'terrible', 'amazing', 'awful',
            'love', 'hate', 'like', 'dislike', 'best', 'worst', 'excellent', 'poor'
        ]
        
        self.processor.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, word in enumerate(common_words):
            self.processor.word2idx[word] = i + 2
    
    def preprocess_text(self, text):
        """预处理文本"""
        if self.model_type == 'bert':
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=DATA_CONFIG['max_length'],
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].to(self.device),
                'attention_mask': encoding['attention_mask'].to(self.device)
            }
        else:
            # 传统预处理
            sequence = self.processor.text_to_sequence(text, DATA_CONFIG['max_length'])
            return {
                'input_ids': torch.tensor([sequence], dtype=torch.long).to(self.device)
            }
    
    def predict(self, text):
        """预测单个文本的情感"""
        # 预处理
        inputs = self.preprocess_text(text)
        
        # 预测
        with torch.no_grad():
            if self.model_type == 'bert':
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            else:
                outputs = self.model(inputs['input_ids'])
            
            # 获取概率
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            
            prob_negative = probabilities[0][0].item()
            prob_positive = probabilities[0][1].item()
            pred_label = prediction[0].item()
        
        result = {
            'text': text,
            'prediction': 'Positive' if pred_label == 1 else 'Negative',
            'confidence': max(prob_negative, prob_positive),
            'probabilities': {
                'negative': prob_negative,
                'positive': prob_positive
            }
        }
        
        return result
    
    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Inference')
    parser.add_argument('--model', type=str, default='bert', 
                       choices=['textcnn', 'lstm', 'bert'],
                       help='Model type to use for prediction')
    parser.add_argument('--text', type=str, 
                       help='Text to analyze')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # 创建预测器
    try:
        predictor = SentimentPredictor(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train.py")
        return
    
    if args.interactive:
        # 交互模式
        print(f"Sentiment Analysis with {args.model.upper()}")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            text = input("\nEnter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = predictor.predict(text)
            
            print(f"\nText: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: Negative={result['probabilities']['negative']:.4f}, "
                  f"Positive={result['probabilities']['positive']:.4f}")
    
    elif args.text:
        # 单次预测
        result = predictor.predict(args.text)
        
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Negative={result['probabilities']['negative']:.4f}, "
              f"Positive={result['probabilities']['positive']:.4f}")
    
    else:
        # 示例文本
        sample_texts = [
            "This movie is absolutely amazing! I loved every minute of it.",
            "Terrible film, waste of time and money.",
            "The movie was okay, nothing special but not bad either.",
            "One of the best films I've ever seen. Highly recommended!",
            "I fell asleep halfway through. Very boring."
        ]
        
        print(f"Sentiment Analysis with {args.model.upper()}")
        print("Analyzing sample texts...")
        print("-" * 60)
        
        results = predictor.predict_batch(sample_texts)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Text: {result['text']}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
