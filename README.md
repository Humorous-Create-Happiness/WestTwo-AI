## python-AI-第一轮

### 机器学习（sklearn）对鸢尾花数据集的分类

#### 第一步：sklearn库的预置：

  1.删除原有的numpy库，重新下载numpy+mkl库;

  2.下载Scipy库；

  3.下载matplotlib库；

  4.下载scikit_learn库；



#### 第二步：了解sklearn库中各个函数并使用

```python
#使用内置的数据集
from sklearn.datasets import load_iris
iris_data = load_iris()
# 该函数返回一个设置成Bunch对象的鸢尾花数据集，直接继承自Dict类，与字典类似，由键值对组成。
# 可以使用bunch.keys(),bunch.values(),bunch.items()等方法。
print(iris_data['data'])  # 花的样本数据,包括花萼长度、花萼宽度、花瓣长度、花瓣宽度的测量数据
# 百度后知道0 代表 setosa， 1 代表 versicolor，2 代表 virginica
print(iris_data['target'])  # 类别:0,1,2
print(iris_data['target_names'])  # 花的品种:setosa..

```





```py
#数据集函数库的导入
from sklearn.model_selection import train_test_split

# 构造训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split( iris_data['data'], iris_data['target'], random_state=0)
print("训练样本数据的大小：{}".format(X_train.shape))
print("训练样本标签的大小：{}".format(y_train.shape))
print("测试样本数据的大小：{}".format(X_test.shape))
print("测试样本标签的大小：{}".format(y_test.shape))
```

#### 第三步：深入了解knn模型

KNN的全称是K Nearest Neighbors，意思是K个最近的邻居，从这个名字我们就能看出一些KNN算法的蛛丝马迹了。K个最近邻居，毫无疑问，K的取值肯定是至关重要的有关细节的东西有两个：K值选取和点距离的计算。

2.1距离计算
	要度量空间中点距离的话，有好几种度量方式，比如常见的曼哈顿距离计算，欧式距离计算等等。不过通常KNN算法中使用的是欧式距离，即简单粗暴地将预测点与所有点距离进行计算，然后保存并排序，选出前面K个值看看哪些类别比较多。但其实也可以通过一些数据结构来辅助，比如最大堆。

2.2 K值选择
	那么该如何确定K取多少值好呢？答案是通过交叉验证（将样本数据按照一定比例，拆分出训练用的数据和验证用的数据，比如6：4拆分出部分训练数据和验证数据），从选取一个较小的K值开始，不断增加K的值，然后计算验证集合的方差，最终找到一个比较合适的K值。

​	所以选择K点的时候可以选择一个较大的临界K点，当它继续增大或减小的时候，错误率都会上升。具体如何得出K最佳值的代码，下一节的代码实例中会介绍。

2.3 KNN特点
	KNN是一种非参的，惰性的算法模型。什么是非参，什么是惰性呢？

​	非参的意思并不是说这个算法不需要参数，而是意味着这个模型不会对数据做出任何的假设，与之相对的是线性回归（我们总会假设线性回归是一条直线）。也就是说KNN建立的模型结构是根据数据来决定的，这也比较符合现实的情况，毕竟在现实中的情况往往与理论上的假设是不相符的。

​	惰性又是什么意思呢？想想看，同样是分类算法，逻辑回归需要先对数据进行大量训练（tranning），最后才会得到一个算法模型。而KNN算法却不需要，它没有明确的训练数据的过程，或者说这个过程很快。

​	KNN算法的优势和劣势
了解KNN算法的优势和劣势，可以帮助我们在选择学习算法的时候做出更加明智的决定。那我们就来看看KNN算法都有哪些优势以及其缺陷所在！

​	KNN算法优点

简单易用，相比其他算法，KNN算是比较简洁明了的算法。即使没有很高的数学基础也能搞清楚它的原理。
模型训练时间快，上面说到KNN算法是惰性的，这里也就不再过多讲述。
预测效果好。
对异常值不敏感
	KNN算法缺点

对内存要求较高，因为该算法存储了所有训练数据
预测阶段可能很慢
对不相关的功能和数据规模敏感

#### 第四步 构造并训练模型

```pyth
# 构造KNN模型
knn = KNeighborsClassifier(n_neighbors=1)当k=1时，训练准确度较高
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
```

