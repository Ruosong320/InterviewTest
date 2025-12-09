import json
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import time

# 添加当前目录到Python路径，以便导入sentence_classifier模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from Task1_Solve1 import SentenceClassifier, classify_text
#from Task1_Solve2 import SentenceClassifier, classify_text


def load_test_data(file_path: Path) -> List[Dict]:
    """加载测试数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return []


def test_single_sentences(test_data: List[Dict]) -> Tuple[int, int, float]:
    """测试单句子分类"""
    print("\n" + "="*60)
    print("开始测试单句子分类...")
    
    classifier = SentenceClassifier()
    correct_count = 0
    total_count = 0
    
    for i, item in enumerate(test_data, 1):
        text = item.get("text", "")
        expected_labels = item.get("expected_labels", [])
        
        if not text or not expected_labels:
            continue
        
        # 单句子分类
        result = classifier.classify_single(text)
        predicted_type = result.get("type", "其他")
        
        # 单句子的expected_labels应该只有一个元素
        if expected_labels and predicted_type == expected_labels[0]:
            correct_count += 1
        else:
            # 显示错误分类的句子
            if i <= 20:  # 只显示前20个错误分类
                print(f"错误分类 {i}: '{text}'")
                print(f"  预期: {expected_labels[0]}, 实际: {predicted_type}")
        
        total_count += 1
        
        # 显示进度
        if i % 100 == 0:
            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            print(f"已处理 {i}/{len(test_data)} 条，当前正确率: {accuracy:.2f}%")
    
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"单句子分类测试完成: {correct_count}/{total_count} 正确，正确率: {accuracy:.2f}%")
    
    return correct_count, total_count, accuracy


def test_paragraphs(test_data: List[Dict]) -> Tuple[int, int, float]:
    """测试段落分类"""
    print("\n" + "="*60)
    print("开始测试段落分类...")
    
    classifier = SentenceClassifier()
    correct_sentences = 0
    total_sentences = 0
    
    for i, paragraph_item in enumerate(test_data, 1):
        text = paragraph_item.get("text", "")
        expected_labels = paragraph_item.get("expected_labels", [])
        
        if not text or not expected_labels:
            continue
        
        # 段落分类
        results = classifier.classify_paragraph(text)
        
        # 比较每个句子的分类结果
        for j, (result, expected_label) in enumerate(zip(results, expected_labels)):
            predicted_type = result.get("type", "其他")
            
            if predicted_type == expected_label:
                correct_sentences += 1
            else:
                # 显示错误分类的句子
                if i <= 10 and j < 3:  # 限制显示数量
                    print(f"段落 {i}, 句子 {j+1} 错误: '{result.get('text', '')[:50]}...'")
                    print(f"  预期: {expected_label}, 实际: {predicted_type}")
            
            total_sentences += 1
        
        # 显示进度
        if i % 20 == 0:
            accuracy = correct_sentences / total_sentences * 100 if total_sentences > 0 else 0
            print(f"已处理 {i}/{len(test_data)} 个段落，当前正确率: {accuracy:.2f}%")
    
    accuracy = correct_sentences / total_sentences * 100 if total_sentences > 0 else 0
    print(f"段落分类测试完成: {correct_sentences}/{total_sentences} 个句子正确，正确率: {accuracy:.2f}%")
    
    return correct_sentences, total_sentences, accuracy


def test_dialogues(test_data: List[Dict]) -> Tuple[int, int, float]:
    """测试对话分类"""
    print("\n" + "="*60)
    print("开始测试对话分类...")
    
    classifier = SentenceClassifier()
    correct_sentences = 0
    total_sentences = 0
    
    for i, dialogue_item in enumerate(test_data, 1):
        text = dialogue_item.get("text", "")
        expected_labels = dialogue_item.get("expected_labels", [])
        
        if not text or not expected_labels:
            continue
        
        # 对话数据包含说话人标签，我们需要先分割
        # 格式: "Speaker1: sentence1. Speaker2: sentence2."
        # 我们需要提取出每个句子，去掉说话人标签
        
        # 简单的分割逻辑：按说话人标签分割
        import re
        # 正则表达式匹配 "Speaker: " 模式
        pattern = r'([A-Za-z]+):\s*'
        
        # 分割文本，保留分割部分
        parts = re.split(pattern, text)
        
        # parts格式: ['', 'Speaker1', 'sentence1. ', 'Speaker2', 'sentence2.']
        # 我们需要提取奇数索引的句子部分（1, 3, 5...）
        sentences = []
        for idx in range(1, len(parts), 2):
            if idx + 1 < len(parts):
                speaker = parts[idx]
                sentence = parts[idx + 1].strip()
                if sentence:
                    sentences.append(sentence)
        
        # 对每个句子进行分类
        for j, (sentence, expected_label) in enumerate(zip(sentences, expected_labels)):
            if j >= len(expected_labels):
                break
                
            result = classifier.classify_single(sentence)
            predicted_type = result.get("type", "其他")
            
            if predicted_type == expected_label:
                correct_sentences += 1
            else:
                # 显示错误分类的句子
                if i <= 5 and j < 2:  # 限制显示数量
                    print(f"对话 {i}, 句子 {j+1} 错误: '{sentence[:50]}...'")
                    print(f"  预期: {expected_label}, 实际: {predicted_type}")
            
            total_sentences += 1
        
        # 显示进度
        if i % 10 == 0:
            accuracy = correct_sentences / total_sentences * 100 if total_sentences > 0 else 0
            print(f"已处理 {i}/{len(test_data)} 个对话，当前正确率: {accuracy:.2f}%")
    
    accuracy = correct_sentences / total_sentences * 100 if total_sentences > 0 else 0
    print(f"对话分类测试完成: {correct_sentences}/{total_sentences} 个句子正确，正确率: {accuracy:.2f}%")
    
    return correct_sentences, total_sentences, accuracy


def test_function_apis():
    """测试函数API"""
    print("\n" + "="*60)
    print("测试函数API...")
    
    test_cases = [
        ("Close the window.", "祈使句"),
        ("What time is it?", "疑问句"),
        ("I love reading books.", "陈述句"),
        ("What a beautiful day!", "感叹句"),
        ("Hello!", "感叹句"),
    ]
    
    correct_count = 0
    
    for text, expected_type in test_cases:
        result = classify_text(text)
        predicted_type = result.get("type", "其他")
        
        if predicted_type == expected_type:
            correct_count += 1
            print(f"✓ '{text[:30]}...' -> {predicted_type}")
        else:
            print(f"✗ '{text[:30]}...' -> 预期: {expected_type}, 实际: {predicted_type}")
    
    accuracy = correct_count / len(test_cases) * 100
    print(f"函数API测试: {correct_count}/{len(test_cases)} 正确，正确率: {accuracy:.2f}%")


def main():
    """主测试函数"""
    print("英文句式分类器测试")
    print("="*60)
    
    # 获取test_data文件夹路径
    test_data_dir = current_dir / "test_data"
    
    # 测试文件路径
    single_data_file = test_data_dir / "test_data_single.json"
    full_data_file = test_data_dir / "test_data_full.json"
    
    # 检查文件是否存在
    if not single_data_file.exists() or not full_data_file.exists():
        print("错误: 测试数据文件不存在!")
        print(f"请确保以下文件存在:")
        print(f"  {single_data_file}")
        print(f"  {full_data_file}")
        print("请先运行 gen_test_dataset.py 生成测试数据")
        return
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 测试单句子数据
    print(f"加载单句子测试数据: {single_data_file}")
    single_test_data = load_test_data(single_data_file)
    if not single_test_data:
        print("单句子测试数据为空!")
        return
    
    single_correct, single_total, single_accuracy = test_single_sentences(single_test_data)
    
    # 2. 测试完整数据
    print(f"\n加载完整测试数据: {full_data_file}")
    full_test_data = load_test_data(full_data_file)
    if not full_test_data:
        print("完整测试数据为空!")
        return
    
    # 从完整数据中提取不同类型的数据
    single_data = full_test_data.get("single_sentences", [])
    paragraph_data = full_test_data.get("paragraphs", [])
    dialogue_data = full_test_data.get("dialogues", [])
    
    print(f"完整数据包含: {len(single_data)} 个单句子, {len(paragraph_data)} 个段落, {len(dialogue_data)} 个对话")
    
    # 测试不同类型的分类
    all_results = []
    
    # 测试完整数据中的单句子
    if single_data:
        print("\n" + "="*60)
        print("测试完整数据中的单句子...")
        classifier = SentenceClassifier()
        correct = 0
        total = 0
        
        for i, item in enumerate(single_data, 1):
            text = item.get("text", "")
            expected_labels = item.get("expected_labels", [])
            
            if not text or not expected_labels:
                continue
            
            result = classifier.classify_single(text)
            predicted_type = result.get("type", "其他")
            
            if predicted_type == expected_labels[0]:
                correct += 1
            total += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        all_results.append(("完整数据-单句子", correct, total, accuracy))
        print(f"完整数据中的单句子: {correct}/{total} 正确，正确率: {accuracy:.2f}%")
    
    # 测试段落数据
    if paragraph_data:
        _, _, para_accuracy = test_paragraphs(paragraph_data)
        # 我们已经在test_paragraphs函数中打印了结果，这里只需要收集数据
        # 重新计算以获得统计数据
        classifier = SentenceClassifier()
        para_correct = 0
        para_total = 0
        
        for paragraph_item in paragraph_data:
            text = paragraph_item.get("text", "")
            expected_labels = paragraph_item.get("expected_labels", [])
            
            if not text or not expected_labels:
                continue
            
            results = classifier.classify_paragraph(text)
            
            for result, expected_label in zip(results, expected_labels):
                predicted_type = result.get("type", "其他")
                if predicted_type == expected_label:
                    para_correct += 1
                para_total += 1
        
        para_accuracy = para_correct / para_total * 100 if para_total > 0 else 0
        all_results.append(("段落数据", para_correct, para_total, para_accuracy))
    
    # 测试对话数据
    if dialogue_data:
        _, _, dialog_accuracy = test_dialogues(dialogue_data)
        # 重新计算以获得统计数据
        classifier = SentenceClassifier()
        dialog_correct = 0
        dialog_total = 0
        
        for dialogue_item in dialogue_data:
            text = dialogue_item.get("text", "")
            expected_labels = dialogue_item.get("expected_labels", [])
            
            if not text or not expected_labels:
                continue
            
            # 分割对话
            import re
            pattern = r'([A-Za-z]+):\s*'
            parts = re.split(pattern, text)
            
            sentences = []
            for idx in range(1, len(parts), 2):
                if idx + 1 < len(parts):
                    sentence = parts[idx + 1].strip()
                    if sentence:
                        sentences.append(sentence)
            
            # 对每个句子进行分类
            for j, (sentence, expected_label) in enumerate(zip(sentences, expected_labels)):
                if j >= len(expected_labels):
                    break
                    
                result = classifier.classify_single(sentence)
                predicted_type = result.get("type", "其他")
                
                if predicted_type == expected_label:
                    dialog_correct += 1
                dialog_total += 1
        
        dialog_accuracy = dialog_correct / dialog_total * 100 if dialog_total > 0 else 0
        all_results.append(("对话数据", dialog_correct, dialog_total, dialog_accuracy))
    
    # 3. 测试函数API
    test_function_apis()
    
    # 计算总时间
    total_time = time.time() - start_time
    
    # 打印汇总报告
    print("\n" + "="*60)
    print("测试汇总报告")
    print("="*60)
    
    # 计算总体统计数据
    total_correct = sum(result[1] for result in all_results)
    total_sentences = sum(result[2] for result in all_results)
    overall_accuracy = total_correct / total_sentences * 100 if total_sentences > 0 else 0
    
    print(f"{'测试类型':<20} {'正确数':<10} {'总数':<10} {'正确率':<10}")
    print("-" * 50)
    
    for test_type, correct, total, accuracy in all_results:
        print(f"{test_type:<20} {correct:<10} {total:<10} {accuracy:.2f}%")
    
    print("-" * 50)
    print(f"{'总计':<20} {total_correct:<10} {total_sentences:<10} {overall_accuracy:.2f}%")
    
    # 单句子数据测试结果
    print(f"\n单句子数据文件测试:")
    print(f"  正确数: {single_correct}/{single_total}")
    print(f"  正确率: {single_accuracy:.2f}%")
    
    print(f"\n测试完成，总耗时: {total_time:.2f} 秒")
    print(f"平均每句处理时间: {total_time/total_sentences*1000:.2f} 毫秒" if total_sentences > 0 else "")


if __name__ == "__main__":
    main()