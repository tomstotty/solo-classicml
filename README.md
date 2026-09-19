# solo-classicml

从零实现的经典机器学习库，仅用 Python 标准库、不联网。

- 入口：`python classicml.py <子命令>`
- 所有随机来源必须由显式随机种子决定；相同种子与输入必须产生逐字节相同的输出。
- 模型参数以 JSON 序列化，浮点数按固定小数位格式化。

## 测试

    python -m unittest discover
