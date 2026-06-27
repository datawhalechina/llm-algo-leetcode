# 18. Triton Block 模型 | Triton Block Model

**难度：** Medium | **标签：** `Triton`, `Block Model`, `Kernel` | **目标人群：** 准备进入 Chapter 3 的学习者

这一页把 block、warp 和数据切块的直觉，接到 Triton 的编程方式上。重点是按块组织计算。

## 前置关系

- Chapter 3 会直接进入 Triton 的 kernel 写法
- 先理解 block/program 的映射，代码才好看懂

## 你应该先建立的直觉

### 1. Triton 不是把 CUDA 语法换个名字

Triton 关注的不是逐行写低层细节，而是：
- 怎么把计算切成合理的块
- 怎么让每个程序实例处理一部分数据
- 怎么把 memory access 和计算布局得更顺

### 2. block/program 映射是核心思路

你可以先把 Triton 理解成：
- 一个 program 对应一个数据块
- 多个 program 共同覆盖整个张量
- 每个 program 只关心自己负责的局部区域

这就是它比“手写很多重复循环”更适合 kernel 开发的原因。

### 3. 分块方式决定性能上限

如果块切得不好，就会出现：
- 访问不连续
- 共享数据复用差
- 负载不均衡

所以 Triton 的重点不是“写法简洁”，而是“分块是否适合硬件”。

## 一个最常见的理解路径

```text
大张量 -> 切成多个块 -> 每个 program 负责一块 -> 程序实例并行执行 -> 完成整体计算
```

这条线比记住 Triton API 更重要。  
后面 Chapter 3 的融合算子和优化实现，都会沿着这条线展开。

## 常见误区

- 把 Triton 当成高级版 PyTorch 语法
- 只看代码简洁，不看 block 映射是否合理
- 以为 program 数量越多越好
- 忽略块内数据布局和访问连续性

## 这一页学完后，你应该能回答

- Triton 的 block/program 思路是什么
- 为什么分块是 Triton 的核心
- 为什么分块方式会影响性能
- 为什么 Chapter 3 的 Triton 代码不能只看表面语法

## 和后续章节的联系

- **Chapter 3: Triton 基础**  
  你会看到更具体的 block 映射和 kernel 书写方式

- **Chapter 3: 融合算子实现**  
  你会看到如何用 Triton 组织多个算子

- **Chapter 3: CUDA 对照理解**  
  你会看到 Triton 的高层映射和 CUDA 的低层执行是如何对应的

## 小结

这一页的作用很简单：
- 先让你知道 Triton 为什么强调 block
- 再让你知道 program 映射在做什么
- 最后让你知道为什么 Chapter 3 要先理解分块思路

如果你能把“大张量切块”和“程序实例映射”联系起来，这一页的目标基本达成。

## 进入 Part 3 前的提示

这一页是进入 Part 3 Triton 主线的直接前置。建议你在看 3.1 之前先把这一页和 `1D` 里的 `15 / 16 / 17 / 19` 一起过一遍，这样后面看到 `program_id`、`mask`、`stride` 和 `block` 时会更顺。
