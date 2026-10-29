"""krylov_recsys 数值核心包。

模块边界（计划书 §5.1, §5.3）：
- data:       清洗、划分、缓存与泄漏检查
- operators:  稀疏归一化与 matvec/matmat（矩阵无显式构造）
- filters:    多项式、Chebyshev、投影求值
- krylov:     Arnoldi、Lanczos、变块秩处理
- stopping:   停止指标、上限与原因编码
- evaluate:   屏蔽、Top-K、Recall 与 NDCG
- benchmark:  阶段计时、内存与运行记录
"""
