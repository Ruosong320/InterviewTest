import nltk
from nltk.tokenize import word_tokenize
import json
from pathlib import Path
import time
from datetime import datetime

# 配置NLTK数据路径
current_script_dir = Path(__file__).resolve().parent
relative_data_path = current_script_dir / 'T1S1Source'
absolute_nltk_data_path = str(relative_data_path)

if absolute_nltk_data_path not in nltk.data.path:
    nltk.data.path.append(absolute_nltk_data_path)

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
                                   'unbelievable', 'excellent', 'perfect']
    
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
            tagged = nltk.pos_tag(tokens)
        except Exception:
            return "Other (其他)"
        
        first_word = tokens[0].lower()
        first_tag = tagged[0][1] if tagged else ''
        
        # 3. 疑问句判断（最高优先级）
        # 情况1：以问号结尾
        if has_question_mark:
            return "Interrogative (疑问句)"
        
        # 情况2：以疑问词开头
        if first_word in self.question_words or first_word in self.wh_question_words:
            return "Interrogative (疑问句)"
        
        # 情况3：倒装语序（助动词在主语前）
        if len(tokens) > 1 and first_word in self.question_words:
            second_tag = tagged[1][1] if len(tagged) > 1 else ''
            if second_tag in ['PRP', 'NN', 'NNP', 'NNS']:  # 代词或名词
                return "Interrogative (疑问句)"
        
        # 4. 祈使句判断
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
        
        # 5. 感叹句判断
        # 特征1：以What/How开头的感叹句
        if first_word in ['what', 'how'] and has_exclamation:
            # 检查后续结构
            if len(tokens) > 2:
                second_word = tokens[1].lower()
                if (first_word == 'what' and second_word in ['a', 'an']) or \
                   (first_word == 'how' and tagged[1][1] in ['JJ', 'RB']):  # 形容词或副词
                    return "Exclamatory (感叹句)"
        
        # 特征2：强烈情感+感叹号
        if has_exclamation and any(word in clean_text.lower() for word in self.strong_emotion_words):
            return "Exclamatory (感叹句)"
        
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

def load_test_data(filename="test_data.json"):
    """加载测试数据"""
    file_path = Path(filename)
    if not file_path.exists():
        print(f"错误：测试文件 {filename} 不存在")
        print("请先运行 generate_test_data.py 生成测试数据")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [(item["sentence"], item["label"]) for item in data]

def run_test(test_data, classifier, sample_size=None):
    """运行测试"""
    if sample_size and sample_size < len(test_data):
        import random
        test_data = random.sample(test_data, sample_size)
    
    print(f"开始测试，数据量: {len(test_data)} 条")
    print("-" * 60)
    
    start_time = time.time()
    
    results = []
    correct_count = 0
    
    for i, (sentence, expected_label) in enumerate(test_data, 1):
        # 分类
        classified_label = classifier.classify(sentence)
        simplified_label = classifier.simplify_label(classified_label)
        
        # 检查是否正确
        is_correct = (simplified_label == expected_label)
        if is_correct:
            correct_count += 1
        
        results.append({
            "sentence": sentence,
            "expected": expected_label,
            "actual": simplified_label,
            "is_correct": is_correct
        })
        
        # 每100条显示一次进度
        if i % 100 == 0:
            accuracy_so_far = correct_count / i * 100
            print(f"已处理 {i}/{len(test_data)} 条，当前正确率: {accuracy_so_far:.2f}%")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 计算统计信息
    accuracy = correct_count / len(test_data) * 100
    
    print("-" * 60)
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每条处理时间: {elapsed_time/len(test_data)*1000:.2f} 毫秒")
    print(f"正确率: {accuracy:.2f}% ({correct_count}/{len(test_data)})")
    
    return results, accuracy, elapsed_time


def main():
    print("\n" + "=" * 60)
    
    # 1. 创建分类器
    classifier = SentenceClassifier()
    
    # 2. 加载测试数据
    test_data = load_test_data()
    if not test_data:
        return
    
    print(f"成功加载 {len(test_data)} 条测试数据")
    
    # 3. 运行测试
    results, accuracy, elapsed_time = run_test(test_data, classifier)
    
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 检查测试数据文件是否存在
    test_file = Path("test_data.json")
    if test_file.exists():
        # 运行完整测试
        main()
       