
---

## 四、项目核心代码框架
```python
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 加载数据（示例，可替换为真实数据集）
# 这里用一个小样本做演示，实际可以下载公开IMDB数据集
data = pd.DataFrame({
    'review': [
        "This movie is amazing! I love every part of it.",
        "Terrible waste of time, the plot makes no sense.",
        "Great acting and beautiful cinematography, highly recommend!",
        "Boring and predictable, I fell asleep halfway.",
        "The best movie I've seen this year!",
        "Worst movie ever, don't watch it."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]  # 1=正面，0=负面
})

# 2. 文本预处理
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 转小写
    text = text.lower()
    # 去除标点和特殊字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词并去除停用词
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['clean_review'] = data['review'].apply(clean_text)

# 3. 特征工程：词袋模型
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['clean_review'])
y = data['sentiment']

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. 训练朴素贝叶斯模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. 模型评估
y_pred = model.predict(X_test)
print(f"模型准确率: {accuracy_score(y_test, y_pred):.2f}")
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 7. 测试新评论
def predict_sentiment(text):
    clean_text_input = clean_text(text)
    text_vector = vectorizer.transform([clean_text_input])
    prediction = model.predict(text_vector)
    return "正面评论" if prediction[0] == 1 else "负面评论"

# 示例
print("\n测试新评论:")
print(predict_sentiment("This film is fantastic, I can't wait to watch it again!"))
print(predict_sentiment("Awful movie, I regret watching it."))
