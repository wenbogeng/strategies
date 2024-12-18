### **多因子投资组合回测**

C++ 程序会模拟 ipynb 对 DataFrame 进行处理，并在过程中打印出 DataFrame 的内容，与 ipynb 打印的内容相对应

**数据结构:**

- **StockData:** 表示 DataFrame 中的单个数据点，可以存储双精度值或字符串
- **DataRow:** 一组 `StockData` 对象，代表 DataFrame 中的单行
- **DataFrame:** 一组 `DataRow` 对象，以及列名和每个列的数据类型（字符串或双精度）信息

**辅助函数:**

该代码库包含用于以下的各种辅助函数：

- 字符串操作（根据分隔符分割字符串）
- 错误处理（检查文件是否可以打开）
- 数学运算（比较函数、计算标准差等）

**数据处理:**

- 从 CSV 文件读取数据，固定数据集保证 ipynb 和 C++ 代码输出的一致性
- 创建和操作 DataFrames，用于存储和管理表格数据
- 选择和转换特定列
- 根据特定列对数据进行分组

**数据预处理:**

- `winsorize`数据（处理异常值）
- `standardize`数据（归一化处理）
- 为分类数据创建虚拟变量
- **行业和市值中性化:**
    - 调整因子值以消除行业和市值偏差
- **指标计算:**
    - 将日期字符串转换为数字表示形式
    - 实现 `pandas.qcut` 和 `pandas.merge`的功能，根据分位数对数据进行分箱
- **多因子投资组合回测:**
    - 计算投资组合收益
    - 计算投资组合风险指标
    - 分析投资组合表现

**使用说明:**

1. C++ 11 或更高版本 环境 Linux Ubuntu 9.4.0-1ubuntu1~20.04.1
2. 包含的第三方库：
    - 线性代数运算库 `Eigen`
3. 已包含 Makefile 
    - 运行 make clean; make
        - 编译生成 main 可执行文件
4. 运行./main 执行c++程序，读取 my_dataframe.csv，并输出结果