import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys
import re

# 配置NLTK数据路径
current_script_dir = Path(__file__).resolve().parent
relative_data_path = current_script_dir / 'T1S1Source'
absolute_nltk_data_path = str(relative_data_path)

if absolute_nltk_data_path not in nltk.data.path:
    nltk.data.path.append(absolute_nltk_data_path)

# 统一测试数据文件路径
TEST_DATA_FILE = current_script_dir / 'test_data.json'

class SentenceClassifier:
    """句子分类器"""
    
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
    
    def is_valid_word(self, word):
        """检查单词是否合法"""
        # 跳过常见标点符号
        if word in ['.', ',', '!', '?', ';', ':', "'", '"', '(', ')', '[', ']', '{', '}']:
            return True
            
        # 检查是否是常见缩写
        if word.lower() in self.common_abbreviations:
            return True
            
        # 检查是否符合英文单词模式（字母、连字符、撇号）
        if self.english_word_pattern.match(word):
            return True
            
        # 检查是否包含数字（可能是年份、时间等）
        if any(char.isdigit() for char in word):
            return True
            
        return False
    
    def is_complete_sentence(self, text, tokens):
        """判断是否是一个完整的句子"""
        # 1. 检查是否有足够的单词
        if len(tokens) < 2:
            # 单个单词可能是感叹词、命令等
            if len(tokens) == 1:
                single_word = tokens[0].lower()
                # 允许的单个单词情况
                allowed_single_words = ['hello', 'hi', 'stop', 'go', 'wait', 'help', 
                                       'yes', 'no', 'ok', 'wow', 'oh', 'hey', 'thanks',
                                       'please', 'sorry', 'goodbye']
                if single_word in allowed_single_words:
                    return True
            return False
        
        # 2. 检查是否有异常单词
        invalid_word_count = sum(1 for token in tokens if not self.is_valid_word(token))
        if invalid_word_count / len(tokens) > 0.3:  # 超过30%的单词异常
            return False
        
        # 3. 检查是否有谓语动词
        try:
            tagged = nltk.pos_tag(tokens)
            # 检查是否有动词（VB, VBD, VBG, VBN, VBP, VBZ）
            has_verb = any(tag.startswith('VB') for _, tag in tagged)
            
            # 检查是否有名词或代词（确保句子有主语或宾语）
            has_noun_or_pronoun = any(tag in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$'] for _, tag in tagged)
            
            # 完整句子通常需要有动词，以及名词/代词（除非是祈使句）
            # 对于感叹句（如"What a beautiful day!"），可能没有动词
            # 所以放宽这个要求
            if not has_verb and not has_noun_or_pronoun:
                return False
                
        except Exception:
            # 如果词性标注失败，则依赖其他规则
            pass
        
        return True
    
    def classify(self, text):
        """
        基于规则和NLTK词性标注判断英文句式。
        分类：疑问句、感叹句、陈述句、祈使句。
        """
        text = text.strip()
        if not text:
            return "Other (其他)"
        
        # 1. 标点符号处理
        has_question_mark = text.endswith('?')
        has_exclamation = text.endswith('!')
        
        # 移除句末标点用于进一步分析
        if text[-1] in '.?!':
            clean_text = text[:-1].strip()
        else:
            clean_text = text
        
        if not clean_text:
            return "Other (其他)"
        
        # 2. 分词和词性标注
        try:
            tokens = word_tokenize(clean_text)
            if not tokens:
                return "Other (其他)"
            
            # 检查是否是完整句子
            if not self.is_complete_sentence(text, tokens):
                return "Other (其他)"
            
            tagged = nltk.pos_tag(tokens)
        except Exception:
            return "Other (其他)"
        
        first_word = tokens[0].lower()
        first_tag = tagged[0][1] if tagged else ''
        
        # 3. 感叹句判断（提前到疑问句之前，因为What开头的句子可能是感叹句）
        # 特征1：以What/How开头的感叹句
        if first_word in ['what', 'how'] and has_exclamation:
            # 检查后续结构
            if len(tokens) >= 2:
                if first_word == 'what':
                    # What + a/an + 形容词 + 名词结构（如"What a beautiful day!")
                    # 或者 What + 形容词 + 名词结构（如"What beautiful weather!")
                    second_word = tokens[1].lower()
                    if second_word in ['a', 'an']:
                        # What a/an + 形容词 + 名词
                        return "Exclamatory (感叹句)"
                    else:
                        # 检查第二个词是否是形容词
                        second_tag = tagged[1][1] if len(tagged) > 1 else ''
                        if second_tag in ['JJ', 'JJR', 'JJS']:  # 形容词
                            return "Exclamatory (感叹句)"
                elif first_word == 'how':
                    # How + 形容词/副词结构（如"How beautiful!", "How quickly!")
                    second_tag = tagged[1][1] if len(tagged) > 1 else ''
                    if second_tag in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:  # 形容词或副词
                        return "Exclamatory (感叹句)"
        
        # 特征2：强烈情感+感叹号
        if has_exclamation:
            # 检查文本中是否包含强烈情感词
            text_lower = text.lower()
            if any(word in text_lower for word in self.strong_emotion_words):
                return "Exclamatory (感叹句)"
            # 检查是否有明显的感叹词
            exclamatory_words = ['oh', 'ah', 'wow', 'ooh', 'aah', 'ugh', 'phew', 'yikes', 'yay']
            if any(word in text_lower.split() for word in exclamatory_words):
                return "Exclamatory (感叹句)"
        
        # 4. 疑问句判断（最高优先级）
        # 情况1：以问号结尾
        if has_question_mark:
            # 排除"What a ...!"被误判为疑问句的情况
            if first_word == 'what' and has_exclamation:
                # 已经在感叹句部分处理过了
                pass
            else:
                return "Interrogative (疑问句)"
        
        # 情况2：以疑问词开头
        if first_word in self.question_words or first_word in self.wh_question_words:
            # 排除感叹句的情况
            if first_word in ['what', 'how'] and has_exclamation:
                # 已经在感叹句部分处理过了
                pass
            else:
                return "Interrogative (疑问句)"
        
        # 情况3：倒装语序（助动词在主语前）
        if len(tokens) > 1 and first_word in self.question_words:
            second_tag = tagged[1][1] if len(tagged) > 1 else ''
            if second_tag in ['PRP', 'NN', 'NNP', 'NNS']:  # 代词或名词
                return "Interrogative (疑问句)"
        
        # 5. 祈使句判断
        # 特征1：以动词原形开头（不包括be动词）
        imperative_indicators = [
            # 动词原形特征
            (first_tag == 'VB' and first_word not in ['be', 'am', 'is', 'are', 'was', 'were']),
            # 否定祈使句
            (first_word == 'do' and len(tokens) > 1 and tokens[1].lower() == 'not'),
            # Let's 开头的祈使句
            (first_word == "let's" or (first_word == 'let' and len(tokens) > 1 and tokens[1].lower() == 'us')),
            # Please 开头的祈使句（通常）
            (first_word == 'please' and len(tokens) > 1 and tagged[1][1] == 'VB'),
        ]
        
        if any(imperative_indicators):
            return "Imperative (祈使句)"
        
        # 6. 默认分类为陈述句
        return "Declarative (陈述句)"
    
    def simplify_label(self, label):
        """简化分类标签"""
        if "疑问句" in label:
            return "疑问句"
        elif "感叹句" in label:
            return "感叹句"
        elif "祈使句" in label:
            return "祈使句"
        elif "陈述句" in label:
            return "陈述句"
        else:
            return "其他"
    
    def classify_single(self, text):
        """单个句子分类接口"""
        result = self.classify(text)
        return self.simplify_label(result)
    
    def classify_paragraph(self, paragraph):
        """段落分类：分割成多个句子并分别分类"""
        sentences = sent_tokenize(paragraph.strip())
        results = []
        for sentence in sentences:
            if sentence.strip():  # 跳过空句子
                result = self.classify_single(sentence)
                results.append((sentence.strip(), result))
        return results

# ========== GUI 界面 ==========
class SentenceClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("英文句式分类器")
        self.root.geometry("550x650")
        
        # 创建分类器实例
        self.classifier = SentenceClassifier()
        
        self.setup_ui()
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="英文句式分类器", font=("Arial", 16, "bold"))
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
            sentences = sent_tokenize(text)
            
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

# ========== 命令行接口 ==========
def command_line_interface():
    """命令行接口"""
    print("=" * 60)
    print("英文句式分类器 - 命令行模式")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'test' 运行批量测试")
    print("=" * 60)
    
    classifier = SentenceClassifier()
    
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
            
            # 判断是否可能是段落
            sentences = sent_tokenize(text)
            
            if len(sentences) == 1:
                result = classifier.classify_single(text)
                elapsed_time = (time.time() - start_time) * 1000
                print(f"分类结果: {result}")
                print(f"处理时间: {elapsed_time:.2f} 毫秒")
            else:
                results = classifier.classify_paragraph(text)
                elapsed_time = (time.time() - start_time) * 1000
                
                print(f"\n输入段落（{len(sentences)}个句子）:")
                print("=" * 50)
                
                for i, (sentence, result) in enumerate(results, 1):
                    print(f"句子 {i}: {sentence}")
                    print(f"    句型: {result}")
                    print()
                
                print(f"总处理时间: {elapsed_time:.2f} 毫秒")
            
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
        with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        total_count = len(test_data)
        classifier = SentenceClassifier()
        
        print(f"\n开始批量测试，数据量: {total_count} 条")
        print("-" * 60)
        
        start_time = time.time()
        correct_count = 0
        
        for i, item in enumerate(test_data, 1):
            sentence = item["sentence"]
            expected = item["label"]
            
            actual = classifier.classify_single(sentence)
            
            if actual == expected:
                correct_count += 1
            
            # 每100条显示一次进度
            if i % 100 == 0:
                accuracy_so_far = correct_count / i * 100
                print(f"已处理 {i}/{total_count} 条，当前正确率: {accuracy_so_far:.2f}%")
        
        elapsed_time = time.time() - start_time
        accuracy = correct_count / total_count * 100
        
        print("-" * 60)
        print(f"测试完成！")
        print(f"总数据量: {total_count} 条")
        print(f"正确分类: {correct_count} 条")
        print(f"正确率: {accuracy:.2f}%")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每条: {elapsed_time/total_count*1000:.2f} 毫秒")
        
    except Exception as e:
        print(f"批量测试失败: {e}")

# ========== API 接口 ==========
def create_classifier():
    """创建分类器实例（API接口）"""
    return SentenceClassifier()

class ClassificationAPI:
    """分类器API接口类"""
    
    def __init__(self):
        self.classifier = SentenceClassifier()
    
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
            
            # 判断是否可能是段落（包含多个句子）
            sentences = sent_tokenize(text.strip())
            
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
            "classifier_type": "Rule-based English Sentence Classifier",
            "version": "1.0.0",
            "supported_categories": ["疑问句", "感叹句", "祈使句", "陈述句", "其他"],
            "features": [
                "单句分类",
                "段落分类",
                "批量处理",
                "异常检测",
                "中英文标签"
            ],
            "question_words_count": len(self.classifier.question_words) + len(self.classifier.wh_question_words),
            "emotion_words_count": len(self.classifier.strong_emotion_words)
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

# ========== 主函数 ==========
def main():
    """主函数：根据参数决定运行模式"""
    # 检查NLTK数据是否可用
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK数据加载成功")
    except LookupError:
        print("警告: NLTK数据未找到，请确保已正确下载")
        print(f"数据路径: {absolute_nltk_data_path}")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "cli":
            # 命令行模式
            command_line_interface()
        elif mode == "gui":
            # GUI模式（默认）
            run_gui()
        elif mode == "test":
            # 批量测试模式
            run_batch_test_from_cli()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: gui, cli, test")
    else:
        # 默认运行GUI模式
        run_gui()

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