import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_data = load_iris()
# 该函数返回一个Bunch对象，它直接继承自Dict类，与字典类似，由键值对组成。
# 可以使用bunch.keys(),bunch.values(),bunch.items()等方法。
print(type(iris_data))
# data里面是花萼长度、花萼宽度、花瓣长度、花瓣宽度的测量数据，格式为 NumPy数组
print(iris_data['data'])  # 花的样本数据
print("花的样本数量：{}".format(iris_data['data'].shape))
print("花的前5个样本数据：{}".format(iris_data['data'][:5]))

# 0 代表 setosa， 1 代表 versicolor，2 代表 virginica
print(iris_data['target'])  # 类别
print(iris_data['target_names'])  # 花的品种

# 构造训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split( iris_data['data'], iris_data['target'], random_state=0)
print("训练样本数据的大小：{}".format(X_train.shape))
print("训练样本标签的大小：{}".format(y_train.shape))
print("测试样本数据的大小：{}".format(X_test.shape))
print("测试样本标签的大小：{}".format(y_test.shape))

# 构造KNN模型
knn = KNeighborsClassifier(n_neighbors=1)
# knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 评估模型
print("模型精度：{:.2f}".format(np.mean(y_pred == y_test)))
print("模型精度：{:.2f}".format(knn.score(X_test, y_test)))

# 做出预测
X_new = np.array([[1.1, 5.9, 1.4, 2.2]])
prediction = knn.predict(X_new)
print("预测的目标类别是：{}".format(prediction))
print("预测的目标类别花名是：{}".format(iris_data['target_names'][prediction]))