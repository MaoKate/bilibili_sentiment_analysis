import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# 下载nltk必需的数据
nltk.download('punkt')
nltk.download('stopwords')

# 加载情感分类数据集
def load_dataset(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label, text = line.strip().split('\t')
            texts.append(text)
            labels.append(label)
    return texts, labels

# 数据预处理
def preprocess(text):
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # 将词转换为小写形式
    lowercase_tokens = [token.lower() for token in filtered_tokens]
    return ' '.join(lowercase_tokens)

# 加载数据集
texts, labels = load_dataset('刘畊宏弹幕.csv')

# 数据预处理
preprocessed_texts = [preprocess(text) for text in texts]

# 特征提取
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(preprocessed_texts)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 训练分类器
classifier = LinearSVC()
classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = classifier.predict(X_test)

# 输出预测结果
for text, true_label, pred_label in zip(X_test, y_test, y_pred):
    print(f'Text: {text}\nTrue label: {true_label}\nPredicted label: {pred_label}\n')

