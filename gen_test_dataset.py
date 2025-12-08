import random
import json
from pathlib import Path

def generate_imperative_sentences(count=250):
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

def generate_interrogative_sentences(count=250):
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

def generate_declarative_sentences(count=250):
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

def generate_exclamatory_sentences(count=250):
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

def generate_test_data(total=1000):
    """生成测试数据"""
    print(f"生成 {total} 条测试数据...")
    
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
    
    # 保存到文件
    output_file = Path("test_data.json")
    
    # 保存为JSON格式
    data_to_save = [{"sentence": s, "label": l} for s, l in all_data]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据已保存到: {output_file}")
    
    # 统计信息
    counts = {
        "祈使句": len(imperative),
        "疑问句": len(interrogative),
        "陈述句": len(declarative),
        "感叹句": len(exclamatory)
    }
    
    print("\n数据分布统计:")
    for label, count in counts.items():
        print(f"  {label}: {count} 条")
    
    return all_data


if __name__ == "__main__":
    # 生成1000条测试数据
    generate_test_data(1000)
    