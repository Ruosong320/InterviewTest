import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import json
from pathlib import Path
import time
import re
import sys
from typing import Dict, List, Tuple, Optional

# 配置NLTK数据路径
current_script_dir = Path(__file__).resolve().parent
relative_data_path = current_script_dir / 'T1S1Source'
absolute_nltk_data_path = str(relative_data_path)

if absolute_nltk_data_path not in nltk.data.path:
    nltk.data.path.append(absolute_nltk_data_path)


class SentenceClassifier:
    """句子分类器核心类"""
    
    def __init__(self):
        # 疑问词列表
        self.question_words = ['do', 'does', 'did', 'are', 'is', 'am', 'was', 'were', 
                             'have', 'has', 'had', 'can', 'could', 'will', 'would', 
                             'shall', 'should', 'may', 'might', 'must']
        
        self.wh_question_words = ['who', 'what', 'where', 'when', 'why', 'how',
                                'which', 'whose', 'whom']
        
        # 强烈情感词
        self.strong_emotion_words = ['wow', 'awesome', 'amazing', 'terrible', 
                                   'horrible', 'fantastic', 'incredible', 
                                   'unbelievable', 'excellent', 'perfect', 
                                   'wonderful', 'great', 'brilliant']
        
        # 英语常见单词模式（用于验证单词合法性）
        self.english_word_pattern = re.compile(r'^[A-Za-z\-\'\.\,\!\?\;]+$')
        
        # 常见英文缩写
        self.common_abbreviations = ["mr.", "mrs.", "ms.", "dr.", "st.", "ave.", "blvd.", 
                                    "etc.", "e.g.", "i.e.", "vs.", "jr.", "sr.", "prof."]
    
    def is_valid_word(self, word: str) -> bool:
        """检查单词是否合法"""
        if word in ['.', ',', '!', '?', ';', ':', "'", '"', '(', ')', '[', ']', '{', '}']:
            return True
            
        if word.lower() in self.common_abbreviations:
            return True
            
        if self.english_word_pattern.match(word):
            return True
            
        if any(char.isdigit() for char in word):
            return True
            
        return False
    
    def is_complete_sentence(self, text: str, tokens: List[str]) -> bool:
        """判断是否是一个完整的句子"""
        if len(tokens) < 2:
            if len(tokens) == 1:
                single_word = tokens[0].lower()
                allowed_single_words = ['hello', 'hi', 'stop', 'go', 'wait', 'help', 
                                       'yes', 'no', 'ok', 'wow', 'oh', 'hey', 'thanks',
                                       'please', 'sorry', 'goodbye']
                if single_word in allowed_single_words:
                    return True
            return False
        
        invalid_word_count = sum(1 for token in tokens if not self.is_valid_word(token))
        if invalid_word_count / len(tokens) > 0.3:
            return False
        
        try:
            tagged = nltk.pos_tag(tokens)
            has_verb = any(tag.startswith('VB') for _, tag in tagged)
            has_noun_or_pronoun = any(tag in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$'] for _, tag in tagged)
            
            if not has_verb and not has_noun_or_pronoun:
                return False
                
        except Exception:
            pass
        
        return True
    
    def classify_single(self, text: str) -> Dict:
        """分类单个句子"""
        text = text.strip()
        if not text:
            return {"text": text, "type": "其他", "confidence": 1.0}
        
        # 标点符号处理
        has_question_mark = text.endswith('?')
        has_exclamation = text.endswith('!')
        
        # 移除句末标点用于进一步分析
        if text[-1] in '.?!':
            clean_text = text[:-1].strip()
        else:
            clean_text = text
        
        if not clean_text:
            return {"text": text, "type": "其他", "confidence": 1.0}
        
        # 分词和词性标注
        try:
            tokens = word_tokenize(clean_text)
            if not tokens:
                return {"text": text, "type": "其他", "confidence": 1.0}
            
            # 检查是否是完整句子
            if not self.is_complete_sentence(text, tokens):
                return {"text": text, "type": "其他", "confidence": 0.7}
            
            tagged = nltk.pos_tag(tokens)
            first_word = tokens[0].lower()
            first_tag = tagged[0][1] if tagged else ''
            
        except Exception:
            return {"text": text, "type": "其他", "confidence": 0.0}
        
        # 感叹句判断
        if first_word in ['what', 'how'] and has_exclamation:
            if len(tokens) >= 2:
                if first_word == 'what':
                    second_word = tokens[1].lower()
                    if second_word in ['a', 'an']:
                        return {"text": text, "type": "感叹句", "confidence": 0.95}
                    else:
                        second_tag = tagged[1][1] if len(tagged) > 1 else ''
                        if second_tag in ['JJ', 'JJR', 'JJS']:
                            return {"text": text, "type": "感叹句", "confidence": 0.9}
                elif first_word == 'how':
                    second_tag = tagged[1][1] if len(tagged) > 1 else ''
                    if second_tag in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                        return {"text": text, "type": "感叹句", "confidence": 0.9}
        
        # 强烈情感词 + 感叹号
        if has_exclamation:
            text_lower = text.lower()
            if any(word in text_lower for word in self.strong_emotion_words):
                return {"text": text, "type": "感叹句", "confidence": 0.85}
            
            exclamatory_words = ['oh', 'ah', 'wow', 'ooh', 'aah', 'ugh', 'phew', 'yikes', 'yay']
            if any(word in text_lower.split() for word in exclamatory_words):
                return {"text": text, "type": "感叹句", "confidence": 0.8}
        
        # 疑问句判断
        if has_question_mark:
            if first_word == 'what' and has_exclamation:
                pass  # 已经在感叹句部分处理过了
            else:
                return {"text": text, "type": "疑问句", "confidence": 0.9}
        
        if first_word in self.question_words or first_word in self.wh_question_words:
            if first_word in ['what', 'how'] and has_exclamation:
                pass  # 已经在感叹句部分处理过了
            else:
                return {"text": text, "type": "疑问句", "confidence": 0.85}
        
        # 倒装语序检查
        if len(tokens) > 1 and first_word in self.question_words:
            second_tag = tagged[1][1] if len(tagged) > 1 else ''
            if second_tag in ['PRP', 'NN', 'NNP', 'NNS']:
                return {"text": text, "type": "疑问句", "confidence": 0.8}
        
        # 祈使句判断
        imperative_indicators = [
            (first_tag == 'VB' and first_word not in ['be', 'am', 'is', 'are', 'was', 'were']),
            (first_word == 'do' and len(tokens) > 1 and tokens[1].lower() == 'not'),
            (first_word == "let's" or (first_word == 'let' and len(tokens) > 1 and tokens[1].lower() == 'us')),
            (first_word == 'please' and len(tokens) > 1 and tagged[1][1] == 'VB'),
        ]
        
        if any(imperative_indicators):
            return {"text": text, "type": "祈使句", "confidence": 0.8}
        
        # 默认分类为陈述句
        return {"text": text, "type": "陈述句", "confidence": 0.7}
    
    def classify_paragraph(self, paragraph: str) -> List[Dict]:
        """段落分类"""
        sentences = sent_tokenize(paragraph.strip())
        results = []
        for sentence in sentences:
            if sentence.strip():
                result = self.classify_single(sentence.strip())
                results.append(result)
        return results
    
    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """批量分类"""
        return [self.classify_single(text) for text in texts]


# API接口函数
def create_classifier() -> SentenceClassifier:
    """创建分类器实例"""
    return SentenceClassifier()


def classify_text(text: str) -> Dict:
    """
    快速分类文本
    
    参数:
        text (str): 输入的英文文本
        
    返回:
        dict: 包含分类结果的字典
        
    示例:
        >>> result = classify_text("What a beautiful day!")
        >>> print(result)
        {'text': 'What a beautiful day!', 'type': '感叹句', 'confidence': 0.95}
    """
    classifier = SentenceClassifier()
    return classifier.classify_single(text)


def classify_paragraph(paragraph: str) -> List[Dict]:
    """
    分类段落中的多个句子
    
    参数:
        paragraph (str): 输入的英文段落
        
    返回:
        list: 包含每个句子分类结果的列表
        
    示例:
        >>> results = classify_paragraph("Hello! How are you? I'm fine.")
        >>> for result in results:
        >>>     print(result)
    """
    classifier = SentenceClassifier()
    return classifier.classify_paragraph(paragraph)


def classify_batch(texts: List[str]) -> List[Dict]:
    """
    批量分类多个文本
    
    参数:
        texts (list): 英文文本列表
        
    返回:
        list: 包含每个文本分类结果的列表
        
    示例:
        >>> results = classify_batch(["Hello", "What is your name?", "Close the door."])
        >>> for result in results:
        >>>     print(result)
    """
    classifier = SentenceClassifier()
    return classifier.classify_batch(texts)


# 主模块导出
__all__ = [
    'SentenceClassifier',
    'create_classifier',
    'classify_text',
    'classify_paragraph',
    'classify_batch'
]


if __name__ == "__main__":
    # 示例用法
    classifier = create_classifier()
    
    # 示例1: 单句分类
    sentence = "What a beautiful day!"
    result = classifier.classify_single(sentence)
    print(f"单句分类结果: {result}")
    
    # 示例2: 段落分类
    paragraph = "Hello! How are you? I'm fine. Thanks!"
    results = classifier.classify_paragraph(paragraph)
    print(f"段落分类结果: {results}")
    
    # 示例3: 批量分类
    texts = [
        "Close the window.",
        "Do you like coffee?",
        "What a beautiful day!",
        "She reads books."
    ]
    batch_results = classifier.classify_batch(texts)
    print(f"批量分类结果: {batch_results}")