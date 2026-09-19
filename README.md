# solo-depot

按 SKU 管理的库存仓储命令行工具，数据保存在工作目录下的本地文件中。

- 仅使用 Python 标准库，不联网。
- 入口：`python depot.py <子命令>`
- 库存数量为整数，任何操作都不允许出现负库存。

## 测试

    python -m unittest discover
