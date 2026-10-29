# Streaming direct graph-filter evaluation

本仓库提供固定低阶图滤波的直接递推、全用户评分、Top20排名、推荐指标及独立进程计时程序。
仅包含项目代码、运行所需配置和本说明；原始交互数据与外部依赖源码另行取得。


## 项目结构

```text
streaming-graph-filter/
├── main.py                   # 全用户评分与评价
├── benchmark.py              # 完整进程计时和内存测量
├── prepare_data.py           # 数据转换入口
├── src/
│   ├── evaluate.py           # 直接递推、参考路线与推荐指标
│   ├── data.py               # 数据读取
│   └── preprocessing.py      # 数据转换辅助函数
├── configs/                  # 系数、输入摘要和参考版本
├── scripts/
│   ├── download_reference.py # 获取固定版本的外部参考依赖
│   └── verify_inputs.py      # 核对输入与依赖摘要
├── requirements.txt
└── README.md
```

## 计算方法

对归一化交互算子B，隐式计算Pz = Bᵀ(Bz)，逐阶传播并即时累加多项式响应。
核心函数为`src/evaluate.py`中的`streaming_direct`。
`streaming_direct`是直接求值路线，`author_snapshot`是固定版本Mem-GF发布预测器的对照路线。
两条路线在该版本下定义的预测函数不同，不能将对照解释为同一预测器的无损替换。

## 环境与安装

使用Python 3.12和CPU PyTorch。建议先创建并激活虚拟环境，再从仓库根执行：

```text
python -m pip install -r requirements.txt
python scripts/download_reference.py
python main.py --author-source external/memgf --self-check --threads 1
```

获取工具只下载固定提交的五个小型源码文件，并核对摘要；这些外部文件不随本包分发。
已有相同版本时，可改用`python scripts/download_reference.py --source 原源码目录`进行离线获取。

## 输入准备

评分入口读取形状相同的二元CSR训练/测试NPZ。训练矩阵至少包含一条用户交互；每用户须至少有20个未交互候选商品。
项目处理版文本的预期摘要位于`configs/inputs.json`与`configs/raw_inputs.json`。对应文本目录结构为：

```text
data/lightgcn_text/yelp2018/train.txt
data/lightgcn_text/yelp2018/test.txt
data/lightgcn_text/amazon-book/train.txt
data/lightgcn_text/amazon-book/test.txt
```

取得摘要一致的文本后，运行：

```text
python prepare_data.py --raw-root data/lightgcn_text --output data/processed
```

该入口保留完整官方训练与测试划分，不使用额外验证集划分。输入须与清单一致；一般二元CSR输入可直接交给评分入口。

## 全用户评分与评价

```text
python main.py --author-source external/memgf --train data/processed/yelp2018/R_train_full_official.npz --test data/processed/yelp2018/R_test.npz --config yelp_f5 --route streaming_direct --batch-size 64 --threads 4 --output runs/yelp_direct
```

将`--route`改为`author_snapshot`可运行发布版对照。
Amazon-book使用`--config amazon_f2`并替换两个输入路径。
质量入口另支持`yelp_f2`与`amazon_f5`；完整系数见`configs/coefficients.json`。
程序覆盖全部用户，屏蔽训练交互，以商品编号升序打破分数平局，输出有序Top20、逐用户四项指标及汇总。
所有输出目录须为新目录，已有目录不会被覆盖。

## 独立进程计时与内存测量

```text
python benchmark.py --author-source external/memgf --train data/processed/yelp2018/R_train_full_official.npz --test data/processed/yelp2018/R_test.npz --config yelp_f5 --route streaming_direct --batch-size 64 --threads 4 --output runs/yelp_direct_performance
```

计时覆盖子进程启动至退出，包括加载、评分、核验、排名、评价及输出；内存记录完整进程采样峰值，Windows另读取生存期峰值。
这是可移植重跑入口，与早期固定实验的计时边界不同，所得时长应在相同入口与配置下比较。
计时入口支持`yelp_f5`和`amazon_f2`两种默认配置；`--help`可查看全部选项。

## 代码来源

项目提供直接求值及配套评价、转换和测量程序；直接多项式递推属于经典数值方法。
外部参考实现Mem-GF由Jin-Duk Park和Won-Yong Shin提出：
https://github.com/jindeok/Mem-GF

参考提交：`ef70fc4826543f84d6ca5ba2a8e8c5f0becd626e`。
原依赖的归属和使用条款继续适用；本包不重新分发其原文件，也不替其授予许可。
