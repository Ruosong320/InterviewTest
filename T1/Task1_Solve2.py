import sys
import json
import time
from pathlib import Path
import re
import numpy as np

# 机器学习相关导入
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 1. 特征工程 (Feature Engineering)
# ==========================================

class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """
    提取人工设计的语言学特征：
    1. 标点符号 (?, !)
    2. 句子长度
    3. Wh-words (疑问词) 出现情况
    4. 动词位置特征
    """
    def __init__(self):
        self.wh_words = ['what', 'where', 'when', 'who', 'why', 'how', 'which', 
                        'do', 'does', 'did', 'are', 'is', 'am', 'was', 'were',
                        'have', 'has', 'had', 'can', 'could', 'will', 'would',
                        'shall', 'should', 'may', 'might', 'must']
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text = str(text).strip()
            if not text:
                features.append([0, 0, 0, 0, 0, 0])
                continue
                
            text_lower = text.lower()
            
            # 特征 1: 是否包含问号
            has_question = 1 if '?' in text else 0
            
            # 特征 2: 是否包含感叹号
            has_exclaim = 1 if '!' in text else 0
            
            # 特征 3: 句子长度 (字符数)
            length = len(text)
            
            # 特征 4: 是否以疑问词开头
            first_word = text_lower.split()[0] if text_lower.split() else ""
            starts_wh = 1 if first_word in self.wh_words else 0
            
            # 特征 5: 是否包含强烈情感词
            strong_emotion_words = ['wow', 'awesome', 'amazing', 'terrible', 
                                   'horrible', 'fantastic', 'incredible', 
                                   'unbelievable', 'excellent', 'perfect', 
                                   'wonderful', 'great', 'brilliant']
            has_emotion = 1 if any(word in text_lower for word in strong_emotion_words) else 0
            
            # 特征 6: 是否以动词原形开头（简单判断）
            starts_with_vb = 1 if first_word in ['close', 'open', 'read', 'write', 'study', 'work',
                                                 'listen', 'speak', 'wait', 'stop', 'go', 'come',
                                                 'leave', 'stay', 'take', 'bring', 'give', 'show',
                                                 'tell', 'ask', 'help', 'call', 'please', 'let'] else 0
            
            features.append([has_question, has_exclaim, length, starts_wh, has_emotion, starts_with_vb])
        
        return np.array(features)

# ==========================================
# 2. 改进的句子拆分器
# ==========================================

class SentenceSplitter:
    """智能句子拆分器，处理各种复杂情况"""
    
    def __init__(self, method='advanced'):
        """
        初始化句子拆分器
        
        参数:
            method (str): 拆分方法，可选值:
                - 'advanced': 使用高级规则+缩写处理（默认）
                - 'simple': 简单标点拆分
        """
        self.method = method
        
        # 常见缩写列表，防止错误拆分
        self.abbreviations = {
            'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'rev.', 'hon.', 'st.', 'ave.',
            'jr.', 'sr.', 'ph.d.', 'm.d.', 'b.a.', 'm.a.', 'd.d.s.', 'd.v.m.',
            'u.s.', 'u.k.', 'e.g.', 'i.e.', 'etc.', 'vs.', 'et al.', 'p.s.',
            'a.m.', 'p.m.', 'b.c.', 'a.d.', 'inc.', 'corp.', 'ltd.', 'co.',
            'no.', 'vol.', 'ch.', 'fig.', 'eq.', 'sec.', 'para.', 'app.',
            'ref.', 'ex.', 'dept.', 'univ.', 'assn.', 'bldg.', 'mt.', 'ft.',
            'lb.', 'oz.', 'pt.', 'qt.', 'gal.', 'in.', 'ft.', 'yd.', 'mi.'
        }
        
        # 特殊情况处理：连字符开头的句子
        self.special_prefixes = ['-', '•', '*', '◦', '›', '»', '▸', '→', '✓']
        
        # 特殊情况处理：引用和对话
        self.quotation_prefixes = ['"', "'", '「', '」', '『', '』']
        
        # 连接词列表，用于判断句子是否需要连接
        self.connectors = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor', 'because',
                          'although', 'since', 'unless', 'while', 'though']
    
    def split_simple(self, text):
        """简单标点拆分（原方法）"""
        # 改进的简单拆分：保留标点
        sentences = []
        current_sentence = []
        chars = list(text)
        
        i = 0
        while i < len(chars):
            char = chars[i]
            current_sentence.append(char)
            
            # 检查是否是句子结束标点
            if char in ['.', '!', '?']:
                # 检查是否在缩写中
                if self._is_in_abbreviation(text, i):
                    i += 1
                    continue
                    
                # 检查是否在数字中
                if self._is_in_number(text, i):
                    i += 1
                    continue
                
                # 检查下一个字符是否可能不是新句子的开始
                if i + 1 < len(chars):
                    next_char = chars[i + 1]
                    # 如果下一个字符不是空格或大写字母，可能不是句子结束
                    if next_char.islower() and next_char.isalpha():
                        i += 1
                        continue
                
                # 创建句子
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = []
            
            i += 1
        
        # 处理最后一个句子
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    def _is_in_abbreviation(self, text, pos):
        """检查位置pos是否在缩写中"""
        # 向前查找单词开始
        start = pos
        while start > 0 and text[start-1].isalpha():
            start -= 1
        
        # 向后查找单词结束
        end = pos
        while end < len(text) - 1 and text[end+1].isalpha():
            end += 1
        
        word = text[start:end+1].lower()
        return word in self.abbreviations
    
    def _is_in_number(self, text, pos):
        """检查位置pos是否在数字中"""
        # 检查前后字符是否为数字
        if pos > 0 and text[pos-1].isdigit():
            return True
        if pos < len(text) - 1 and text[pos+1].isdigit():
            return True
        return False
    
    def split_advanced(self, text):
        """高级句子拆分，考虑更多语言特征"""
        # 首先处理特殊前缀
        if text.startswith(tuple(self.special_prefixes)):
            return self._split_special_prefix(text)
        
        # 处理引用/对话
        if text.startswith(tuple(self.quotation_prefixes)):
            return self._split_quotation(text)
        
        # 使用状态机进行句子拆分
        sentences = []
        current_sentence = []
        in_quotes = False
        quote_char = None
        in_parentheses = 0
        
        i = 0
        while i < len(text):
            char = text[i]
            current_sentence.append(char)
            
            # 处理引号
            if char in ['"', "'", '`']:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            # 处理括号
            elif char == '(':
                in_parentheses += 1
            elif char == ')':
                in_parentheses -= 1
                if in_parentheses < 0:
                    in_parentheses = 0
            
            # 检查句子结束
            elif char in ['.', '!', '?', '。', '！', '？']:
                # 如果不在引号或括号中，且不是缩写/数字
                if not in_quotes and in_parentheses == 0:
                    if self._is_sentence_end(text, i):
                        # 创建句子
                        sentence = ''.join(current_sentence).strip()
                        if sentence and len(sentence) > 1:  # 避免只有标点的句子
                            sentences.append(sentence)
                        current_sentence = []
            
            i += 1
        
        # 处理最后一个句子
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence and len(sentence) > 1:
                sentences.append(sentence)
        
        # 如果没找到句子，返回整个文本作为一个句子
        if not sentences:
            return [text.strip()]
        
        return sentences
    
    def _is_sentence_end(self, text, pos):
        """判断位置pos是否是句子结束"""
        # 排除缩写
        if self._is_in_abbreviation(text, pos):
            return False
        
        # 排除数字中的点
        if self._is_in_number(text, pos):
            return False
        
        # 检查是否在URL或电子邮件中
        if pos > 0 and text[pos-1] in [':', '/', '@']:
            return False
        
        # 检查下一个字符
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            # 如果下一个字符是空格、换行、结束符或大写字母，可能是句子结束
            if next_char in [' ', '\n', '\t', '\r', '"', "'", ')', ']', '}']:
                return True
            elif next_char.isupper():
                return True
            # 如果当前是问号或感叹号，即使后面是小写也结束句子
            elif text[pos] in ['!', '?', '！', '？']:
                return True
        
        return True
    
    def _split_special_prefix(self, text):
        """处理以特殊前缀开头的文本"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        sentences = []
        
        for line in lines:
            # 移除特殊前缀
            cleaned_line = line
            for prefix in self.special_prefixes:
                if cleaned_line.startswith(prefix):
                    cleaned_line = cleaned_line[len(prefix):].strip()
            
            # 如果还有内容，进一步拆分
            if cleaned_line:
                # 递归调用拆分方法（使用simple方法避免递归）
                sub_sentences = self.split_simple(cleaned_line)
                sentences.extend(sub_sentences)
            elif line:  # 如果只有前缀，保留原始行
                sentences.append(line)
        
        return sentences
    
    def _split_quotation(self, text):
        """处理引用/对话文本"""
        # 简单处理：按引号拆分
        parts = []
        current_part = []
        in_quote = False
        
        for char in text:
            current_part.append(char)
            
            if char in ['"', "'"]:
                if in_quote:
                    # 结束引用
                    part = ''.join(current_part).strip()
                    if part:
                        parts.append(part)
                    current_part = []
                    in_quote = False
                else:
                    in_quote = True
            elif not in_quote and char in ['.', '!', '?']:
                # 不在引号中，可能是句子结束
                part = ''.join(current_part).strip()
                if part:
                    parts.append(part)
                current_part = []
        
        # 处理最后一部分
        if current_part:
            part = ''.join(current_part).strip()
            if part:
                parts.append(part)
        
        # 如果拆分失败，回退到简单拆分
        if not parts:
            return self.split_simple(text)
        
        return parts
    
    def split_paragraph(self, paragraph):
        """主拆分函数"""
        paragraph = paragraph.strip()
        if not paragraph:
            return []
        
        # 根据选择的方法进行拆分
        if self.method == 'advanced':
            return self.split_advanced(paragraph)
        else:
            return self.split_simple(paragraph)

# ==========================================
# 3. 核心模型类 (The Core Logic)
# ==========================================

class SentenceClassifier:
    """基于传统机器学习的句子分类器"""
    
    def __init__(self, split_method='advanced'):
        self.pipeline = None
        # 标签顺序必须与训练时的数字标签对应
        self.labels = ["陈述句", "疑问句", "感叹句", "祈使句", "其他"]
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.reverse_label_map = {i: label for label, i in self.label_map.items()}
        
        # 初始化句子拆分器
        self.splitter = SentenceSplitter(method=split_method)
        
        # 初始化模型
        self.build_and_train_model()
    
    def build_and_train_model(self):
        """构建并训练模型"""
        # 构建特征管道
        combined_features = FeatureUnion([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3), 
                min_df=1,
                max_features=5000,
                stop_words='english'
            )),
            ("manual", Pipeline([
                ('extractor', HandcraftedFeatures()),
                ('scaler', MinMaxScaler())
            ]))
        ])

        # 构建完整管道
        self.pipeline = Pipeline([
            ("features", combined_features),
            ("classifier", LinearSVC(
                dual='auto', 
                random_state=42,
                C=1.0,
                max_iter=1000
            )) 
        ])
        
        # 训练数据
        X_train, y_train = self.get_training_data()
        self.pipeline.fit(X_train, y_train)
    
    def get_training_data(self):
        """获取训练数据"""
        # 这里使用内置的训练数据
        X_train = [
            # 陈述句
            "Today is a sunny day.", "I like coding Python.", "He went to the market.",
            "She works in a hospital.", "The book is on the table.", "We are going to the park.",
            "They have finished their homework.", "It is raining outside.", "My name is John.",
            "The cat is sleeping.", "He reads books every day.", "The sun rises in the east.",
            
            # 疑问句
            "What is your name?", "How do you do?", "Where is the library?",
            "When will you arrive?", "Why did you leave?", "Who is that person?",
            "Can you help me?", "Do you like coffee?", "Have you finished your work?",
            "Is this your book?", "Are you coming to the party?", "Will you marry me?",
            
            # 感叹句
            "What a beautiful view!", "So amazing!", "I can't believe it!",
            "How wonderful!", "What a surprise!", "That's fantastic!",
            "Wow, that's incredible!", "Oh my God!", "What a perfect day!",
            "How terrible!", "What a brilliant idea!", "Unbelievable!",
            
            # 祈使句
            "Open the door.", "Sit down please.", "Do not touch that.",
            "Close the window.", "Please be quiet.", "Take your time.",
            "Let's go to the park.", "Stop right now!", "Wait for me.",
            "Please pass the salt.", "Do your homework.", "Clean your room.",
            
            # 其他（不完整或异常句子）
            "Hello", "Hi there", "OK",
            "123456", "asdfghjkl", "The",
            "And then", "Because of", "In the"
        ]
        
        # 标签：0-陈述句, 1-疑问句, 2-感叹句, 3-祈使句, 4-其他
        y_train = (
            [0] * 12 +  # 陈述句
            [1] * 12 +  # 疑问句
            [2] * 12 +  # 感叹句
            [3] * 12 +  # 祈使句
            [4] * 9     # 其他
        )
        
        return X_train, y_train
    
    def preprocess_text(self, text):
        """预处理文本"""
        text = text.strip()
        if not text:
            return ""
        
        # 确保句子以标点结尾
        if text[-1] not in ['.', '!', '?']:
            text += '.'
        
        return text
    
    def classify_single(self, text):
        """分类单个句子"""
        text = text.strip()
        if not text:
            return {"text": text, "type": "其他", "confidence": 1.0}
        
        text = self.preprocess_text(text)
        try:
            prediction_idx = self.pipeline.predict([text])[0]
            return {
                "text": text,
                "type": self.reverse_label_map[prediction_idx],
                "confidence": 0.7  # 机器学习模型通常不提供置信度，这里给一个固定值
            }
        except Exception as e:
            return {"text": text, "type": "其他", "confidence": 0.0}
    
    def classify_paragraph(self, paragraph):
        """段落分类 - 使用改进的句子拆分"""
        sentences = self.splitter.split_paragraph(paragraph)
        results = []
        for sentence in sentences:
            if sentence.strip():
                result = self.classify_single(sentence.strip())
                results.append(result)
        return results
    
    def classify_batch(self, texts):
        """批量分类"""
        return [self.classify_single(text) for text in texts]


# ==========================================
# 4. API接口函数
# ==========================================

def create_classifier(split_method='advanced') -> SentenceClassifier:
    """创建分类器实例
    
    参数:
        split_method (str): 句子拆分方法，可选:
            - 'advanced': 使用高级规则（推荐）
            - 'simple': 简单标点拆分
    """
    return SentenceClassifier(split_method=split_method)


def classify_text(text: str):
    """
    快速分类文本
    
    参数:
        text (str): 输入的英文文本
        
    返回:
        dict: 包含分类结果的字典
        
    示例:
        >>> result = classify_text("What a beautiful day!")
        >>> print(result)
        {'text': 'What a beautiful day!', 'type': '感叹句', 'confidence': 0.7}
    """
    classifier = SentenceClassifier()
    return classifier.classify_single(text)


def classify_paragraph(paragraph: str, split_method='advanced'):
    """
    分类段落中的多个句子
    
    参数:
        paragraph (str): 输入的英文段落
        split_method (str): 句子拆分方法
        
    返回:
        list: 包含每个句子分类结果的列表
        
    示例:
        >>> results = classify_paragraph("Hello! How are you? I'm fine.", split_method='advanced')
        >>> for result in results:
        >>>     print(result)
    """
    classifier = SentenceClassifier(split_method=split_method)
    return classifier.classify_paragraph(paragraph)


def classify_batch(texts):
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


# ==========================================
# 5. 测试和评估函数
# ==========================================

def test_split_methods():
    """测试不同拆分方法的效果"""
    
    test_paragraphs = [
        # 测试1: 包含缩写的段落
        "Dr. Smith works at the Univ. of California. He has a Ph.D. in Computer Science. Mr. Johnson is his colleague.",
        
        # 测试2: 包含项目符号的段落
        "- First, open the document. - Then, read the instructions. - Finally, submit your work.",
        
        # 测试3: 复杂标点使用
        "Hello! How are you? I'm fine... Wait, what? That's amazing!",
        
        # 测试4: 包含数字和缩写的段落
        "The meeting is at 2:30 p.m. in Bldg. 5. Please bring the vol. 3 report. See you then!",
        
        # 测试5: 长段落
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It is used to apply machine learning algorithms to text and speech. For example, we can use NLP to create systems like speech recognition, document summarization, machine translation, spam detection, etc."
    ]
    
    splitter = SentenceSplitter()
    
    print("=" * 80)
    print("句子拆分方法测试")
    print("=" * 80)
    
    for i, paragraph in enumerate(test_paragraphs, 1):
        print(f"\n测试段落 {i}:")
        print(f"原文: {paragraph[:100]}...")
        
        for method in ['simple', 'advanced']:
            splitter.method = method
            sentences = splitter.split_paragraph(paragraph)
            print(f"\n{method.upper()} 方法拆分结果 ({len(sentences)} 句):")
            for j, sent in enumerate(sentences, 1):
                print(f"  {j}. {sent[:80]}{'...' if len(sent) > 80 else ''}")


# ==========================================
# 6. 主模块导出
# ==========================================

__all__ = [
    'SentenceClassifier',
    'SentenceSplitter',
    'create_classifier',
    'classify_text',
    'classify_paragraph',
    'classify_batch',
    'test_split_methods'
]


if __name__ == "__main__":
    # 测试不同拆分方法
    test_split_methods()
    
    # 示例用法
    print("\n" + "=" * 80)
    print("分类器示例用法")
    print("=" * 80)
    
    classifier = create_classifier(split_method='advanced')
    
    # 示例1: 单句分类
    sentence = "What a beautiful day!"
    result = classifier.classify_single(sentence)
    print(f"\n单句分类结果: {result}")
    
    # 示例2: 段落分类
    paragraph = "Hello! How are you? I'm fine. Thanks! Dr. Smith will arrive at 3:00 p.m."
    results = classifier.classify_paragraph(paragraph)
    print(f"\n段落分类结果:")
    for i, result in enumerate(results, 1):
        print(f"  句子{i}: {result['text'][:50]}... -> {result['type']}")
    
    # 示例3: 批量分类
    texts = [
        "Close the window.",
        "Do you like coffee?",
        "What a beautiful day!",
        "She reads books."
    ]
    batch_results = classifier.classify_batch(texts)
    print(f"\n批量分类结果:")
    for i, result in enumerate(batch_results, 1):
        print(f"  文本{i}: {result['type']}")