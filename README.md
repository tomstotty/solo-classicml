# solo-classicml

从零实现的经典机器学习库，仅用 Python 标准库、不联网。

- 入口：`python classicml.py <子命令>`
- 所有随机来源必须由显式随机种子决定；相同种子与输入必须产生逐字节相同的输出。
- 模型参数以 JSON 序列化，浮点数按固定小数位格式化。

## 测试

    python -m unittest discover

## 子命令示例

### train-linear

拟合 `LinearRegression`，从标准输入读取训练参数，把拟合结果写到标准输出：

    $ echo '{"X":[[1.0],[2.0],[3.0],[4.0]],"y":[3.0,5.0,7.0,9.0],"lr":0.1,"l2":0.0,"max_iter":1000,"tol":0.0000000001}' | python classicml.py train-linear
    {"class":"LinearRegression","lr":0.1000000000,"l2":0.0000000000,"max_iter":1000,"tol":0.0000000001,"w":[2.0000000011],"b":0.9999999968}

### train-logistic

拟合二分类 `LogisticRegression`（`y` 只能取 0/1）：

    $ echo '{"X":[[0.0],[1.0],[2.0],[3.0]],"y":[0,0,1,1],"lr":0.5,"l2":0.0,"max_iter":1000,"tol":0.0000000001}' | python classicml.py train-logistic
    {"class":"LogisticRegression","lr":0.5000000000,"l2":0.0000000000,"max_iter":1000,"tol":0.0000000001,"w":[6.0181991064],"b":-8.7789263253}

### predict-knn

用 `KNeighborsClassifier` 对查询点 `Q` 预测类别：

    $ echo '{"X":[[0.0],[1.0],[2.0],[3.0]],"y":[0,0,1,1],"n_neighbors":1,"Q":[[0.4],[2.6]]}' | python classicml.py predict-knn
    {"class":"KNeighborsClassifier","predictions":[0,1]}
