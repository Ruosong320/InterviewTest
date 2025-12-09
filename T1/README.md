# Task1
三种不同方法实现的英文句子属性分类判断器

## 方法1：nltk
【原理】使用 NLTK 的分词和分句功能处理输入的英文文本。对每个句子进行词性标注（POS Tagging），识别出句子中每个单词的语法身份（如动词 VB、名词 NN、代词 PRP 等）。之后根据句子成分组成和标点等信息判定句子属性。可以通过API在其他程序中调用类并创建对象实现分析，实现难度大，分类精确度依赖于对句式的拆分，但是速度最快，适合快速判断的业务场景。
【实现及示例】提供API接口，可被其他python脚本调用：（可运行APItest.py查看运行test_data文件夹下测试文件正确率）
    from Task1_Solve1 import SentenceClassifier  # 分类器类
    classifier = SentenceClassifier()
    
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


## 方法2：传统算法与机器学习
【原理】采用 LinearSVC（线性支持向量分类器）作为核心模型。采用 FeatureUnion，将两种互补的特征工程方法结合起来，形成一个强大的特征向量。使用 Scikit-learn 的 Pipeline 和 FeatureUnion 工具构建一个完整的特征提取和分类流程。
统计特征 (TF-IDF)：使用 TfidfVectorizer 提取文本的词频-逆文档频率（Term Frequency-Inverse Document Frequency）特征，捕捉句子中词汇的重要性。
人工特征 (Handcrafted Features)：自定义 HandcraftedFeatures 转换器，提取语言学信号，如问号、感叹号、句子长度、是否以疑问词（What、Do 等）或动词原形开头。这些特征使用 MinMaxScaler 进行归一化。
可以通过API在其他程序中调用类并创建对象实现分析，实现难度中等，分类精确度经过轻量训练后较高，但是速度较慢，适合需要较为精确判断或数据量不大的场景。
【实现及示例】提供与方案一命名方式一样的API接口，可被其他python脚本调用：（可修改头文件引用后运行APItest.py查看运行test_data文件夹下测试文件正确率）
