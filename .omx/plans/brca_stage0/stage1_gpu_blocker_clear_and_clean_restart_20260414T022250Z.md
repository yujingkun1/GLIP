# Stage 1 执行记录：清理 GPU 阻塞并重新启动统一 baseline

## 背景
此前 batch_size=2 与 batch_size=1 的两次 Visium baseline 重试都在 GPU OOM 失败，但 fresh evidence 显示主要原因并非当前 BRCA 任务本身，而是两个无关的旧任务训练进程长期占用 GPU0/GPU1。

## 已执行操作
- 核对 PID 775862 / 775899 为旧的 C-Regis Stage5 训练
- 终止上述两个旧进程，释放当前 BRCA 任务所需 GPU 资源
- 重新以 **统一配置** 启动新的 clean baseline run：
  - model=UNI2-h
  - batch_size=2
  - split=sample-level LOO
  - data=SPA119-154

## 原则
旧的 batch2/batch1 partial runs 仅保留为失败证据，不作为最终 baseline 结果来源。
最终 Stage1 Visium baseline 以新的 clean run 为准。
