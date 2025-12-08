from Task1_Solve1 import classify_text

# 测试数据
test_sentences = [
    "Close the window.",
    "Do you like coffee?",
    "What a beautiful day!",
    "She reads books.",
    "Please sit down.",
    "What a perfect afternoon for a picnic!",
    "Where are you from?",
    "Let's go to the park."
]

print("英文句式分类器 - 简单API测试")
print("=" * 50)

# 测试单个句子
for sentence in test_sentences:
    result = classify_text(sentence)
    print(f"'{sentence}'")
    print(f"  分类: {result}")
    print()

# 测试段落
paragraph = "Hello! How are you? I'm fine, thank you. What a lovely day!"
print(f"段落测试: {paragraph}")
print("-" * 50)

results = classify_text(paragraph)
for i, (sentence, classification) in enumerate(results, 1):
    print(f"句子 {i}: {sentence}")
    print(f"    分类: {classification}")
    print()