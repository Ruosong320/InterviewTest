import sys
import json
import time
from pathlib import Path
import re
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import argparse

# 机器学习相关导入
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 配置和初始化
# ==========================================

# 统一测试数据文件路径
current_script_dir = Path(__file__).resolve().parent
TEST_DATA_FILE = current_script_dir / 'test_data.json'

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
# 2. 核心模型类 (The Core Logic)
# ==========================================

class SentenceClassifierML:
    """基于传统机器学习的句子分类器"""
    
    def __init__(self):
        self.pipeline = None
        # 标签顺序必须与训练时的数字标签对应
        self.labels = ["陈述句", "疑问句", "感叹句", "祈使句", "其他"]
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.reverse_label_map = {i: label for label, i in self.label_map.items()}
        
        # 初始化模型
        self.build_and_train_model()
    
    def build_and_train_model(self):
        """构建并训练模型"""
        print("正在构建并训练传统机器学习模型...")
        
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
        print("模型训练完成。")
    
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
        if not text or not text.strip():
            return "其他"
        
        text = self.preprocess_text(text)
        try:
            prediction_idx = self.pipeline.predict([text])[0]
            return self.reverse_label_map[prediction_idx]
        except Exception as e:
            print(f"分类错误: {e}")
            return "其他"
    
    def improved_sentence_splitter(self, text):
        """句子分割器"""
        sentences = []
        current = ""
        i = 0
        n = len(text)
        
        while i < n:
            current += text[i]
            
            # 检查是否是句子结束标点
            if text[i] in '.!?':
                # 检查是否是缩写（简化版）
                if i > 0 and i < n - 1:
                    # 常见缩写模式
                    if text[i] == '.' and text[i-1].isalpha() and text[i+1].isalpha():
                        # 可能是缩写，如 "Dr. Smith"
                        i += 1
                        continue
                    elif text[i] == '.' and i > 1 and text[i-1] == '.':
                        # 可能是省略号 "..."
                        if i > 2 and text[i-2] == '.':
                            # 完整的省略号
                            pass
                
                # 检查是否是数字中的小数点
                if text[i] == '.' and i > 0 and text[i-1].isdigit():
                    if i < n - 1 and text[i+1].isdigit():
                        # 小数，如 "3.14"
                        i += 1
                        continue
                
                # 检查下一字符是否是引号或括号
                if i < n - 1:
                    next_char = text[i+1]
                    if next_char in '\'")\]}':
                        # 标点在引号或括号内，继续
                        i += 1
                        continue
                
                # 保存句子
                sentences.append(current.strip())
                current = ""
            
            i += 1
        
        # 添加最后一个句子（如果没有以标点结尾）
        if current.strip():
            sentences.append(current.strip())
        
        return sentences

    def classify_paragraph(self, paragraph):
        """分类段落 - 使用改进的分割器"""
        sentences = self.improved_sentence_splitter(paragraph)
        
        results = []
        for sentence in sentences:
            if sentence.strip():
                result = self.classify_single(sentence)
                results.append((sentence.strip(), result))
        
        return results
    
    def evaluate_on_test_data(self, test_file_path):
        """在测试数据上评估模型性能"""
        if not Path(test_file_path).exists():
            print(f"错误: 找不到测试文件 '{test_file_path}'")
            return
        
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            return
        
        if not test_data:
            print("警告: 测试数据为空")
            return
        
        # 准备数据
        X_test = [item['sentence'] for item in test_data]
        y_true_labels = [self.label_map.get(item['label'], 4) for item in test_data]  # 默认为"其他"
        
        print(f"开始评估，测试数据量: {len(X_test)} 条")
        print("-" * 60)
        
        start_time = time.time()
        
        # 批量预测
        predictions = []
        correct_count = 0
        
        for i, (text, true_label) in enumerate(zip(X_test, y_true_labels), 1):
            try:
                pred_label = self.classify_single(text)
                pred_idx = self.label_map.get(pred_label, 4)
                
                if pred_idx == true_label:
                    correct_count += 1
                
                predictions.append(pred_idx)
                
                # 每100条显示一次进度
                if i % 100 == 0:
                    print(f"已处理 {i}/{len(X_test)} 条")
            except Exception as e:
                print(f"处理第 {i} 条数据时出错: {e}")
                predictions.append(4)  # 默认为"其他"
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        accuracy = correct_count / len(X_test)
        
        print("\n" + "=" * 60)
        print("评估结果:")
        print("=" * 60)
        print(f"总测试数据: {len(X_test)} 条")
        print(f"正确分类: {correct_count} 条")
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每条: {elapsed_time/len(X_test)*1000:.2f} 毫秒")
        print(f"QPS: {len(X_test)/elapsed_time:.2f} 条/秒")
        
        return accuracy

# ==========================================
# 3. GUI 界面 (与之前保持相同)
# ==========================================

class SentenceClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("英文句式分类器 (机器学习版)")
        self.root.geometry("550x650")
        
        # 创建分类器实例
        self.classifier = SentenceClassifierML()
        
        self.setup_ui()
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="英文句式分类器 (机器学习版)", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # 输入标签
        input_label = ttk.Label(main_frame, text="输入英文句子或段落:")
        input_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        # 输入文本框
        self.input_text = scrolledtext.ScrolledText(main_frame, width=70, height=10, wrap=tk.WORD)
        self.input_text.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        self.input_text.insert("1.0", "请输入英文句子或段落...")
        
        # 示例按钮
        example_frame = ttk.Frame(main_frame)
        example_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(example_frame, text="示例:").pack(side=tk.LEFT, padx=(0, 10))
        
        examples = [
            "Close the window.",
            "Do you like coffee?",
            "What a beautiful day!",
            "She reads books. He writes stories."
        ]
        for example in examples:
            btn = ttk.Button(example_frame, text=example[:15]+"..." if len(example) > 15 else example, 
                           command=lambda e=example: self.insert_example(e))
            btn.pack(side=tk.LEFT, padx=2)
        
        # 分类按钮
        classify_btn = ttk.Button(main_frame, text="分类", command=self.classify_text, width=20)
        classify_btn.grid(row=4, column=0, columnspan=2, pady=(0, 15))
        
        # 结果区域
        result_frame = ttk.LabelFrame(main_frame, text="分类结果", padding="10")
        result_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 结果文本框
        self.result_text = scrolledtext.ScrolledText(result_frame, width=65, height=15, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 批量测试按钮
        test_btn = ttk.Button(main_frame, text="批量测试", command=self.run_batch_test, width=20)
        test_btn.grid(row=6, column=0, pady=(10, 0), padx=(0, 5))
        
        # 清除按钮
        clear_btn = ttk.Button(main_frame, text="清除", command=self.clear_text, width=20)
        clear_btn.grid(row=6, column=1, pady=(10, 0), padx=(5, 0))
        
        # 状态栏
        self.status_label = ttk.Label(main_frame, text="就绪", relief=tk.SUNKEN)
        self.status_label.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def insert_example(self, example):
        """插入示例句子"""
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", example)
    
    def classify_text(self):
        """分类文本（单句或段落）"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text or text == "请输入英文句子或段落...":
            messagebox.showwarning("警告", "请输入英文文本")
            return
        
        self.status_label.config(text="正在分类...")
        self.root.update()
        
        try:
            start_time = time.time()
            
            # 判断是否可能是段落（包含多个句子）
            sentences = self.sent_tokenize(text)
            
            if len(sentences) == 1:
                # 单个句子
                result = self.classifier.classify_single(text)
                elapsed_time = (time.time() - start_time) * 1000
                
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert("1.0", f"输入: {text}\n\n")
                self.result_text.insert(tk.END, f"分类结果: {result}\n")
                self.result_text.insert(tk.END, f"处理时间: {elapsed_time:.2f} 毫秒\n")
            else:
                # 多个句子
                results = self.classifier.classify_paragraph(text)
                elapsed_time = (time.time() - start_time) * 1000
                
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert("1.0", f"输入段落（{len(sentences)}个句子）:\n")
                self.result_text.insert(tk.END, f"{text}\n\n")
                self.result_text.insert(tk.END, "="*50 + "\n")
                
                for i, (sentence, result) in enumerate(results, 1):
                    self.result_text.insert(tk.END, f"句子 {i}: {sentence}\n")
                    self.result_text.insert(tk.END, f"    句型: {result}\n\n")
                
                self.result_text.insert(tk.END, f"总处理时间: {elapsed_time:.2f} 毫秒\n")
            
            self.status_label.config(text="分类完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"分类时发生错误: {str(e)}")
            self.status_label.config(text="分类失败")
    
    def sent_tokenize(self, text):
        """简单的句子分割"""
        # 使用正则表达式分割句子
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return sentences
    
    def run_batch_test(self):
        """运行批量测试"""
        if not TEST_DATA_FILE.exists():
            messagebox.showwarning("警告", f"测试文件 {TEST_DATA_FILE} 不存在\n请先运行 generate_test_data.py 生成测试数据")
            return
        
        # 创建批量测试窗口
        batch_window = tk.Toplevel(self.root)
        batch_window.title("批量测试")
        batch_window.geometry("500x400")
        
        # 主框架
        main_frame = ttk.Frame(batch_window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="批量测试", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # 加载测试数据
        try:
            with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            total_count = len(test_data)
            
            # 测试信息
            info_label = ttk.Label(main_frame, text=f"测试数据: {total_count} 条")
            info_label.grid(row=1, column=0, pady=(0, 10))
            
            # 进度条
            progress = ttk.Progressbar(main_frame, length=300, mode='indeterminate')
            progress.grid(row=2, column=0, pady=(0, 10))
            
            # 结果文本框
            result_text = scrolledtext.ScrolledText(main_frame, width=50, height=15, wrap=tk.WORD)
            result_text.grid(row=3, column=0, pady=(0, 10))
            
            # 开始测试按钮
            def start_test():
                progress.start()
                result_text.delete("1.0", tk.END)
                result_text.insert("1.0", "开始测试...\n")
                batch_window.update()
                
                start_time = time.time()
                correct_count = 0
                
                for i, item in enumerate(test_data, 1):
                    sentence = item["sentence"]
                    expected = item["label"]
                    
                    actual = self.classifier.classify_single(sentence)
                    
                    if actual == expected:
                        correct_count += 1
                    
                    # 每100条更新一次进度
                    if i % 100 == 0:
                        result_text.insert(tk.END, f"已处理 {i}/{total_count} 条...\n")
                        batch_window.update()
                
                elapsed_time = time.time() - start_time
                accuracy = correct_count / total_count * 100
                
                progress.stop()
                
                # 显示结果
                result_text.insert(tk.END, "\n" + "="*50 + "\n")
                result_text.insert(tk.END, f"测试完成！\n")
                result_text.insert(tk.END, f"总数据量: {total_count} 条\n")
                result_text.insert(tk.END, f"正确分类: {correct_count} 条\n")
                result_text.insert(tk.END, f"正确率: {accuracy:.2f}%\n")
                result_text.insert(tk.END, f"总耗时: {elapsed_time:.2f} 秒\n")
                result_text.insert(tk.END, f"平均每条: {elapsed_time/total_count*1000:.2f} 毫秒\n")
            
            test_btn = ttk.Button(main_frame, text="开始测试", command=start_test, width=20)
            test_btn.grid(row=4, column=0)
            
        except Exception as e:
            error_label = ttk.Label(main_frame, text=f"加载测试数据失败: {str(e)}", foreground="red")
            error_label.grid(row=1, column=0, pady=20)
    
    def clear_text(self):
        """清除文本框内容"""
        self.input_text.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)
        self.status_label.config(text="已清除")

# ==========================================
# 4. 命令行接口
# ==========================================

def command_line_interface():
    """命令行接口"""
    classifier = SentenceClassifierML()
    
    print("=" * 60)
    print("英文句式分类器 - 机器学习版 (命令行模式)")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'test' 运行批量测试")
    print("=" * 60)
    
    while True:
        try:
            text = input("\n请输入英文句子或段落: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if text.lower() == 'test':
                run_batch_test_from_cli()
                continue
            
            if not text:
                continue
            
            start_time = time.time()
            
            # 使用正则表达式判断是否为段落
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            if len(sentences) > 1:
                # 段落
                results = classifier.classify_paragraph(text)
                elapsed_time = (time.time() - start_time) * 1000
                
                print(f"\n输入段落（{len(results)}个句子）:")
                print("=" * 50)
                
                for i, (sentence, result) in enumerate(results, 1):
                    print(f"句子 {i}: {sentence}")
                    print(f"    句型: {result}")
                    print()
                
                print(f"总处理时间: {elapsed_time:.2f} 毫秒")
            else:
                # 单句
                result = classifier.classify_single(text)
                elapsed_time = (time.time() - start_time) * 1000
                print(f"分类结果: {result}")
                print(f"处理时间: {elapsed_time:.2f} 毫秒")
            
        except KeyboardInterrupt:
            print("\n程序已退出")
            break
        except Exception as e:
            print(f"错误: {e}")

def run_batch_test_from_cli():
    """命令行批量测试"""
    if not TEST_DATA_FILE.exists():
        print(f"错误：测试文件 {TEST_DATA_FILE} 不存在")
        print("请先运行 generate_test_data.py 生成测试数据")
        return
    
    try:
        classifier = SentenceClassifierML()
        classifier.evaluate_on_test_data(TEST_DATA_FILE)
    except Exception as e:
        print(f"批量测试失败: {e}")

# ==========================================
# 5. API接口
# ==========================================

def create_classifier():
    """创建分类器实例（API接口）"""
    return SentenceClassifierML()

class ClassificationAPI:
    """分类器API接口类"""
    
    def __init__(self):
        self.classifier = SentenceClassifierML()
    
    def classify_text(self, text):
        """
        分类文本API接口
        
        参数:
            text (str): 输入的英文文本，可以是单个句子或段落
        
        返回:
            dict: 包含分类结果的字典
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "输入文本不能为空",
                "result": None
            }
        
        try:
            start_time = time.time()
            
            # 使用正则表达式分割句子
            sentences = [s.strip() for s in re.split(r'[.!?]+', text.strip()) if s.strip()]
            
            if len(sentences) == 1:
                # 单个句子
                result = self.classifier.classify_single(text.strip())
                elapsed_time = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "input_type": "single_sentence",
                    "input": text.strip(),
                    "result": result,
                    "processing_time_ms": elapsed_time
                }
            else:
                # 多个句子
                results = self.classifier.classify_paragraph(text.strip())
                elapsed_time = (time.time() - start_time) * 1000
                
                # 格式化结果
                formatted_results = []
                for i, (sentence, classification) in enumerate(results, 1):
                    formatted_results.append({
                        "sentence_index": i,
                        "sentence": sentence,
                        "classification": classification
                    })
                
                return {
                    "success": True,
                    "input_type": "paragraph",
                    "input": text.strip(),
                    "sentence_count": len(sentences),
                    "results": formatted_results,
                    "processing_time_ms": elapsed_time
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input": text.strip(),
                "result": None
            }
    
    def batch_classify(self, texts):
        """
        批量分类API接口
        
        参数:
            texts (list): 英文文本列表
        
        返回:
            dict: 包含批量分类结果的字典
        """
        if not texts or not isinstance(texts, list):
            return {
                "success": False,
                "error": "输入必须是文本列表",
                "results": None
            }
        
        try:
            start_time = time.time()
            results = []
            
            for i, text in enumerate(texts, 1):
                if not text or not text.strip():
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "文本为空",
                        "input": text,
                        "result": None
                    })
                    continue
                
                try:
                    classification_result = self.classifier.classify_single(text.strip())
                    results.append({
                        "index": i,
                        "success": True,
                        "input": text.strip(),
                        "result": classification_result
                    })
                except Exception as e:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": str(e),
                        "input": text.strip(),
                        "result": None
                    })
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "total_count": len(texts),
                "success_count": sum(1 for r in results if r["success"]),
                "failed_count": sum(1 for r in results if not r["success"]),
                "results": results,
                "total_processing_time_ms": elapsed_time,
                "average_time_per_item_ms": elapsed_time / len(texts) if texts else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": None
            }
    
    def get_statistics(self):
        """
        获取分类器统计信息
        
        返回:
            dict: 分类器统计信息
        """
        return {
            "classifier_type": "Traditional ML English Sentence Classifier",
            "version": "1.0.0",
            "algorithm": "LinearSVC + TF-IDF + Handcrafted Features",
            "supported_categories": ["陈述句", "疑问句", "感叹句", "祈使句", "其他"],
            "features": [
                "单句分类",
                "段落分类",
                "批量处理",
                "TF-IDF特征",
                "手工特征提取"
            ],
            "training_samples": 57,  # 训练样本数量
            "feature_dimensions": 5006,  # TF-IDF (5000) + 手工特征(6)
        }

# 创建默认API实例
_default_api = None

def get_classifier_api():
    """获取分类器API实例（单例模式）"""
    global _default_api
    if _default_api is None:
        _default_api = ClassificationAPI()
    return _default_api

def classify_text(text):
    """
    快速分类文本（简化API）
    
    参数:
        text (str): 输入的英文文本
    
    返回:
        str or list: 单个句子的分类结果或段落分类的结果列表
    
    示例:
        >>> classify_text("Close the window.")
        '祈使句'
        >>> classify_text("Hello! How are you?")
        [('Hello!', '感叹句'), ('How are you?', '疑问句')]
    """
    api = get_classifier_api()
    result = api.classify_text(text)
    
    if result["success"]:
        if result["input_type"] == "single_sentence":
            return result["result"]
        else:
            # 对于段落，返回简化的结果列表
            return [(item["sentence"], item["classification"]) for item in result["results"]]
    else:
        raise ValueError(result["error"])

# ==========================================
# 6. 主函数
# ==========================================

def main():
    """主函数：根据参数决定运行模式"""
    parser = argparse.ArgumentParser(description='英文句式分类器')
    parser.add_argument('mode', nargs='?', default='gui', choices=['gui', 'cli', 'test'],
                       help='运行模式: gui (图形界面), cli (命令行), test (批量测试)')
    parser.add_argument('--text', type=str, help='直接分类的文本')
    
    args = parser.parse_args()
    
    if args.text:
        # 直接分类文本
        classifier = SentenceClassifierML()
        result = classifier.classify_single(args.text)
        print(f"分类结果: {result}")
    elif args.mode == "cli":
        command_line_interface()
    elif args.mode == "gui":
        run_gui()
    elif args.mode == "test":
        run_batch_test_from_cli()

def run_gui():
    """运行GUI界面"""
    root = tk.Tk()
    app = SentenceClassifierGUI(root)
    
    # 使窗口可调整大小
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    root.mainloop()

if __name__ == "__main__":
    main()