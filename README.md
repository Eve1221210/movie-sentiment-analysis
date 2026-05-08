# movie-sentiment-analysis
# 🎬 电影评论情感分析
AI 初学者的第一个NLP项目：用朴素贝叶斯模型判断电影评论的正负情感。

## 📌 项目目标
- 学习文本预处理流程（清洗、分词、去停用词）
- 理解词袋模型（Bag of Words）的基本原理
- 用朴素贝叶斯实现文本二分类（正面/负面评论）
- 掌握模型评估的基本指标（准确率、混淆矩阵）

## 🛠️ 技术栈
- Python 3.x
- `scikit-learn`：模型训练与评估
- `nltk` / `re`：文本预处理
- `pandas`：数据处理

## 📊 数据集
使用公开的 IMDB 电影评论小样本数据集，包含5000条带标签的正负评论。

## 🚀 运行步骤
1.  安装依赖：
    ```bash
    pip install pandas scikit-learn nltk
    ```
2.  下载nltk停用词（首次运行）：
    ```python
    import nltk
    nltk.download('stopwords')
    ```
3.  运行主脚本：
    ```bash
    python sentiment_analysis.py
    ```

## 📝 学习笔记
- 文本清洗是NLP的基础：去除标点、小写转换、去停用词能大幅提升模型效果
- 词袋模型虽然简单，但对短文本分类非常有效
- 朴素贝叶斯模型训练速度快，很适合作为入门级文本分类模型

## 📈 模型效果
测试集准确率：约85%
