import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Union
import os

def generate_imperative_sentences(count=250) -> List[Tuple[str, str]]:
    """生成祈使句"""
    sentences = []
    
    verbs = ["close", "open", "clean", "wash", "read", "write", "study", "work",
             "listen", "speak", "wait", "stop", "go", "come", "leave", "stay",
             "take", "bring", "give", "show", "tell", "ask", "help", "call"]
    
    objects = ["the window", "the door", "your room", "the car", "the book",
               "your homework", "to me", "your hands", "the dishes", "the floor",
               "your mind", "the phone", "the computer", "the light", "the TV",
               "the report", "the document", "the package", "the letter"]
    
    adverbs = ["", " please", " quickly", " carefully", " quietly", " immediately",
               " now", " right now", " at once", " before leaving"]
    
    for _ in range(count):
        verb = random.choice(verbs)
        obj = random.choice(objects)
        adverb = random.choice(adverbs)
        sentence = f"{verb.capitalize()} {obj}{adverb}."
        if random.random() < 0.2:  # 20%的句子用感叹号
            sentence = sentence[:-1] + "!"
        sentences.append((sentence, "祈使句"))
    
    return sentences

def generate_interrogative_sentences(count=250) -> List[Tuple[str, str]]:
    """生成疑问句"""
    sentences = []
    
    question_words = ["Do", "Does", "Did", "Are", "Is", "Was", "Were",
                      "Have", "Has", "Had", "Can", "Could", "Will", "Would",
                      "Shall", "Should", "May", "Might", "Must"]
    
    wh_words = ["What", "When", "Where", "Why", "How", "Who", "Whom", "Which"]
    
    subjects = ["you", "he", "she", "it", "we", "they", "the teacher",
                "your friend", "your brother", "the company", "the government"]
    
    verbs = ["like", "know", "understand", "remember", "think", "believe",
             "want", "need", "have", "do", "go", "come", "work", "study"]
    
    objects = ["the movie", "the answer", "the problem", "the way", "the time",
               "the place", "the reason", "the solution", "to the party",
               "about this", "with that", "for dinner"]
    
    for _ in range(count):
        if random.random() < 0.6:  # 60%用一般疑问词
            q_word = random.choice(question_words)
            subject = random.choice(subjects)
            verb = random.choice(verbs)
            obj = random.choice(objects) if random.random() < 0.7 else ""
            sentence = f"{q_word} {subject} {verb} {obj}?".strip()
        else:  # 40%用特殊疑问词
            q_word = random.choice(wh_words)
            aux = random.choice(["do", "does", "did", "is", "are", "was", "were", "have", "has"]) if random.random() < 0.7 else ""
            subject = random.choice(subjects) if random.random() < 0.8 else ""
            verb = random.choice(verbs) if aux else ""
            obj = random.choice(objects) if random.random() < 0.5 else ""
            
            parts = [q_word, aux, subject, verb, obj]
            sentence = " ".join(filter(None, parts)) + "?"
        
        # 5%的句子故意用句号而不是问号（语法错误）
        if random.random() < 0.05:
            sentence = sentence[:-1] + "."
        
        sentences.append((sentence, "疑问句"))
    
    return sentences

def generate_declarative_sentences(count=250) -> List[Tuple[str, str]]:
    """生成陈述句"""
    sentences = []
    
    subjects = ["I", "You", "He", "She", "It", "We", "They", 
                "The cat", "The dog", "The student", "The teacher",
                "My friend", "Our company", "The government", "The weather"]
    
    verbs_present = ["like", "love", "hate", "know", "understand", "think",
                     "believe", "want", "need", "have", "do", "go", "come",
                     "work", "study", "read", "write", "speak", "listen"]
    
    verbs_past = ["liked", "loved", "hated", "knew", "understood", "thought",
                  "believed", "wanted", "needed", "had", "did", "went", "came",
                  "worked", "studied", "read", "wrote", "spoke", "listened"]
    
    objects = ["books", "movies", "music", "sports", "food", "coffee", "tea",
               "the idea", "the plan", "the project", "the solution", "the answer",
               "in the morning", "at night", "every day", "with friends",
               "at school", "at work", "at home"]
    
    adverbs = ["", " always", " usually", " often", " sometimes", " rarely",
               " never", " probably", " definitely", " certainly"]
    
    for _ in range(count):
        subject = random.choice(subjects)
        
        if subject in ["I", "You", "We", "They"] or " " in subject:
            verb_base = random.choice(verbs_present)
            verb = verb_base if subject == "I" else verb_base + "s" if subject in ["He", "She", "It"] else verb_base
        else:
            if " " in subject:  # 复数主语
                verb = random.choice(verbs_present)
            else:
                verb = random.choice(verbs_present) + "s" if random.random() < 0.7 else random.choice(verbs_present)
        
        if random.random() < 0.3:  # 30%用过去时
            verb = random.choice(verbs_past)
        
        obj = random.choice(objects)
        adverb = random.choice(adverbs)
        
        sentence = f"{subject} {verb} {obj}{adverb}."
        
        # 复杂陈述句
        if random.random() < 0.1:
            conjunctions = ["because", "although", "since", "while", "if", "when"]
            conjunction = random.choice(conjunctions)
            second_subject = random.choice(subjects)
            second_verb = random.choice(verbs_present)
            second_obj = random.choice(objects)
            sentence = f"{sentence[:-1]} {conjunction} {second_subject} {second_verb} {second_obj}."
        
        sentences.append((sentence, "陈述句"))
    
    return sentences

def generate_exclamatory_sentences(count=250) -> List[Tuple[str, str]]:
    """生成感叹句"""
    sentences = []
    
    what_phrases = ["What a", "What an", "What"]
    what_nouns = ["beautiful day", "wonderful surprise", "great idea", 
                  "amazing performance", "terrible mistake", "horrible accident",
                  "fantastic movie", "brilliant solution", "stupid decision"]
    
    how_adjectives = ["How beautiful", "How wonderful", "How amazing", 
                      "How terrible", "How fantastic", "How brilliant",
                      "How stupid", "How clever", "How kind", "How rude"]
    
    emotional_phrases = ["I can't believe", "Oh my God", "Wow", "Awesome",
                         "Incredible", "Unbelievable", "Fantastic", "Terrible"]
    
    for _ in range(count):
        if random.random() < 0.4:  # 40% What型感叹句
            phrase = random.choice(what_phrases)
            noun = random.choice(what_nouns)
            sentence = f"{phrase} {noun}!"
        elif random.random() < 0.7:  # 30% How型感叹句
            adjective = random.choice(how_adjectives)
            if random.random() < 0.5:
                subject = random.choice(["she is", "he is", "it is", "they are", "you are"])
                sentence = f"{adjective} {subject}!"
            else:
                sentence = f"{adjective}!"
        else:  # 30% 情感型感叹句
            phrase = random.choice(emotional_phrases)
            if random.random() < 0.5:
                continuation = ["this is", "that was", "it is", "you are"]
                cont = random.choice(continuation)
                adj = random.choice(["amazing", "incredible", "unbelievable", "fantastic"])
                sentence = f"{phrase} {cont} {adj}!"
            else:
                sentence = f"{phrase}!"
        
        # 10%的句子故意用句号而不是感叹号
        if random.random() < 0.1:
            sentence = sentence[:-1] + "."
        
        sentences.append((sentence, "感叹句"))
    
    return sentences

def generate_single_sentences(total=1000) -> List[Dict[str, Union[str, List[str]]]]:
    """生成单句子测试数据"""
    print(f"生成 {total} 条单句子测试数据...")
    
    # 计算各类别的数量（大致平均分布）
    each_count = total // 4
    remaining = total % 4
    
    imperative = generate_imperative_sentences(each_count + (1 if remaining > 0 else 0))
    interrogative = generate_interrogative_sentences(each_count + (1 if remaining > 1 else 0))
    declarative = generate_declarative_sentences(each_count + (1 if remaining > 2 else 0))
    exclamatory = generate_exclamatory_sentences(each_count)
    
    # 合并所有数据
    all_data = imperative + interrogative + declarative + exclamatory
    
    # 打乱顺序
    random.shuffle(all_data)
    
    # 转换为标准格式
    data_to_save = []
    for sentence, label in all_data:
        data_to_save.append({
            "text": sentence,
            "type": "single",
            "expected_labels": [label],
            "sentence_count": 1
        })
    
    return data_to_save

def generate_paragraphs(paragraph_count=200, min_sentences=2, max_sentences=5) -> List[Dict[str, Union[str, List[str]]]]:
    """生成多句子段落测试数据"""
    print(f"生成 {paragraph_count} 个段落测试数据...")
    
    # 先生成大量的单句子
    single_sentences = []
    single_sentences.extend(generate_imperative_sentences(100))
    single_sentences.extend(generate_interrogative_sentences(100))
    single_sentences.extend(generate_declarative_sentences(100))
    single_sentences.extend(generate_exclamatory_sentences(100))
    
    paragraphs = []
    
    for i in range(paragraph_count):
        # 随机决定段落中的句子数量
        sentence_count = random.randint(min_sentences, max_sentences)
        
        # 随机选择句子
        selected_sentences = random.sample(single_sentences, sentence_count)
        
        # 构建段落文本和标签列表
        paragraph_text = ""
        expected_labels = []
        
        for j, (sentence, label) in enumerate(selected_sentences):
            # 确保句子以合适的标点结束
            if not sentence[-1] in ['.', '!', '?']:
                sentence += '.'
            
            # 将句子添加到段落中，并在句子间添加空格
            if j > 0:
                paragraph_text += " "
            paragraph_text += sentence
            
            expected_labels.append(label)
        
        paragraphs.append({
            "text": paragraph_text,
            "type": "paragraph",
            "expected_labels": expected_labels,
            "sentence_count": sentence_count
        })
    
    return paragraphs

def generate_mixed_dialogue(dialogue_count=100, min_exchanges=2, max_exchanges=4) -> List[Dict[str, Union[str, List[str]]]]:
    """生成对话式混合段落"""
    print(f"生成 {dialogue_count} 个对话式段落测试数据...")
    
    # 所有类型的句子
    all_sentences = []
    all_sentences.extend(generate_imperative_sentences(100))
    all_sentences.extend(generate_interrogative_sentences(100))
    all_sentences.extend(generate_declarative_sentences(100))
    all_sentences.extend(generate_exclamatory_sentences(100))
    
    dialogues = []
    speakers = ["John", "Mary", "Tom", "Alice", "Teacher", "Student", "Boss", "Employee"]
    
    for i in range(dialogue_count):
        # 随机决定对话轮次
        exchange_count = random.randint(min_exchanges, max_exchanges)
        
        dialogue_text = ""
        expected_labels = []
        
        for j in range(exchange_count):
            # 随机选择说话者和句子
            speaker = random.choice(speakers)
            sentence, label = random.choice(all_sentences)
            
            # 确保句子以合适的标点结束
            if not sentence[-1] in ['.', '!', '?']:
                sentence += '.'
            
            # 构建对话行
            if j > 0:
                dialogue_text += " "
            dialogue_text += f"{speaker}: {sentence}"
            
            expected_labels.append(label)
        
        dialogues.append({
            "text": dialogue_text,
            "type": "dialogue",
            "expected_labels": expected_labels,
            "sentence_count": exchange_count
        })
    
    return dialogues

def generate_test_data(total=1000, include_paragraphs=True, include_dialogues=True) -> Dict[str, List]:
    """生成测试数据集"""
    print(f"开始生成测试数据...")
    
    # 确保有足够的数据
    base_count = total // 3 if include_paragraphs or include_dialogues else total
    
    # 生成单句子数据
    single_data = generate_single_sentences(base_count)
    
    all_data = {
        "single_sentences": single_data,
        "paragraphs": [],
        "dialogues": []
    }
    
    # 生成段落数据（如果需要）
    if include_paragraphs:
        paragraph_count = total // 4
        all_data["paragraphs"] = generate_paragraphs(paragraph_count)
    
    # 生成对话数据（如果需要）
    if include_dialogues:
        dialogue_count = total // 4
        all_data["dialogues"] = generate_mixed_dialogue(dialogue_count)
    
    # 获取当前脚本所在目录，并创建test_data子目录
    current_dir = Path(__file__).parent
    test_data_dir = current_dir / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # 保存到文件
    output_files = {}
    
    # 保存单句子数据
    single_output = test_data_dir / "test_data_single.json"
    with open(single_output, 'w', encoding='utf-8') as f:
        json.dump(single_data, f, ensure_ascii=False, indent=2)
    output_files["single"] = single_output
    
    # 保存完整数据
    full_output = test_data_dir / "test_data_full.json"
    with open(full_output, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    output_files["full"] = full_output
    
    # 打印统计信息
    print(f"\n测试数据生成完成!")
    print(f"单句子数据: {len(single_data)} 条")
    print(f"段落数据: {len(all_data['paragraphs'])} 个")
    print(f"对话数据: {len(all_data['dialogues'])} 个")
    
    # 计算句子总数
    total_sentences = len(single_data)
    for paragraph in all_data["paragraphs"]:
        total_sentences += paragraph["sentence_count"]
    for dialogue in all_data["dialogues"]:
        total_sentences += dialogue["sentence_count"]
    
    print(f"总句子数: {total_sentences}")
    print(f"输出文件:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")
    
    return all_data

if __name__ == "__main__":
    # 生成完整测试数据
    print("="*50)
    test_data = generate_test_data(
        total=1000,
        include_paragraphs=True,
        include_dialogues=True
    )