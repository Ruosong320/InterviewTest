import nltk
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加到nltk路径
nltk.data.path.append(current_dir)

# 下载到当前目录的T1S1Source文件夹
nltk.download('punkt', download_dir=current_dir)
nltk.download('averaged_perceptron_tagger', download_dir=current_dir)

print(f"下载完成！数据已保存到：{current_dir}")