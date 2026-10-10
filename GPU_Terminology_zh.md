# GPU 性能术语速查表（新手向）

本文档配合 `flash_attn_kernel.cu` 项目，解释本项目中用到的所有 GPU 性能术语。
用「工厂车间」比喻贯穿全文。

---

## 一、硬件结构（"舞台"）

### GPU / SM（Streaming Multiprocessor，流式多处理器）
GPU 由几十个 SM 组成，SM 是真正干活的"车间"。
本项目显卡 RTX 4070 SUPER 有 **46 个 SM**。

### Warp（线程束）
GPU 以 **32 个线程为一个 warp** 一起执行同一条指令（SIMT）。
warp 是调度和执行的**基本单位**，相当于一个"工人"。

### Warp Scheduler（warp 调度器 / "工头"）
每个 SM 有 **4 个调度器**，各自管理一批 warp。
调度器**每周期最多给一个 warp 发射一条指令**。

### 执行单元（Execution Units，"机器"）
| 单元 | 作用 | 数量/SM |
|------|------|---------|
| FP32 Core | 普通加减乘 | 128 |
| Tensor Core（张量核） | 矩阵乘加速 | 4 |
| SFU | exp/sin 等特殊函数 | 32 |
| LSU | 读/写内存 | 4 |

---

## 二、核心性能概念

### Issue（发射指令）
调度器把一条指令交给 warp 执行。调度器最多每周期发射 1 条。

### Latency（延迟）
从"发出指令"到"结果可用"的周期数。类比：外卖从下单到送达的时间。
| 操作 | 延迟(周期) |
|------|-----------|
| 普通加法 | ~4 |
| 张量核矩阵乘 | ~20 |
| 读共享内存 | ~30 |
| 读显存 DRAM | ~500 |

### Throughput（吞吐量）
单位时间完成的活，衡量"整体快不快"（如 TFLOPS）。延迟 vs 吞吐：一件活多久 vs 每秒多少活。

### Stall（停顿 / 阻塞）
warp 在等结果时无法继续执行的状态。工人站着等，就叫 stall。

### Occupancy（占用率）
每个 SM 上**实际驻留的 warp 数 ÷ 最大能放的 warp 数**。
**注意：occupancy 不是手动指定的旋钮，而是硬件资源限制决定的结果**（见第五节）。

### 延迟受限 vs 吞吐受限（latency-bound vs throughput-bound）
- 吞吐受限：机器满负荷，问题是"机器不够快"。
- 延迟受限：机器大量空转，问题是"工人在等，没人派活"。
本项目的 kernel 属于**延迟受限**（张量核只用 ~54%）。

### 依赖链（Dependency Chain）
一条指令必须等上一条的结果才能算，形成串行等待。链越长越难并行。

### ILP（Instruction Level Parallelism，指令级并行）
"能同时独立进行的活有多少"。独立链越多，越能交错发指令藏延迟。

### Block / Thread（块 / 线程）
- **Thread**：最小执行单元。
- **Block**：一组线程，共享 shared memory，可做 `__syncthreads()` 同步。
- **Grid**：所有 block 的总和。kernel 启动时你指定 gridDim（多少 block）和 blockDim（每 block 多少线程）。

---

## 三、内存层级（"材料放哪"）

| 层级 | 容量(每SM) | 延迟 | 类比 |
|------|-----------|------|------|
| 寄存器 Register | 65536 个 | 最快 | 工人手上的工具 |
| 共享内存 Shared Memory | 100KB | ~30周期 | 车间公共货架 |
| L1/L2 缓存 | L2 约 48MB | ~200周期 | 工厂仓库 |
| 全局内存/显存 DRAM | 12GB | ~500周期 | 城外总仓 |

### Register（寄存器）
最快、最稀缺。**每个线程用多少寄存器，直接决定一个 SM 能放多少线程**。

### Shared Memory（共享内存）
block 内线程共享，分 32 个 bank（货架道）。

### Bank Conflict（bank 冲突）
多个线程同时读同一个 bank 就得排队，读写慢 N 倍。
本项目 `ldm=128`（32 的倍数）导致 8 路冲突，改 `136`（padding）后归零，提速 2 倍。

### Global Memory / DRAM（全局内存/显存）
最慢最大。本项目 K/V 每 head 仅 18.9MB，被 L2 缓存兜住，DRAM 几乎空闲（1.5%）。

---

## 四、Stall 类型（ncu 实测，单位=每发一条指令平均 stall 周期）

| 类型 | 含义 | 大白话 |
|------|------|--------|
| long_scoreboard | 等全局内存读回 | 等城外材料 |
| short_scoreboard | 等共享内存读回 | 等货架材料 |
| math_pipe_throttle | 等张量核空出来 | 机器被占用 |
| wait | 等寄存器依赖 | 等上一步结果 |
| barrier | 等 `__syncthreads` 到齐 | 等人齐开工 |
| not_selected | 调度器选了别的 warp | 工头喊了别人 |

**Scoreboard（记分板）**：硬件跟踪"哪个结果还没回来"的机制；结果未回，warp 被挂起（stall）。

本项目实测（v6）：math_pipe_throttle 27% + short_scoreboard 22% + wait 21% + barrier 15% + long_scoreboard 7%。

---

## 五、为什么 Occupancy 不能"手动拉满"

### 三个硬件硬限制
1. **每 SM 最多 1536 线程（48 warp）**。
2. **寄存器文件 65536 个/线程**：若每线程用 R 个寄存器，最多放 `65536 / R` 个线程。
3. **共享内存 100KB/SM**：若每 block 用 S，最多放 `100KB / S` 个 block。

### 本项目 v6 的限制计算
- 寄存器 128/线程 → 最多 `65536/128 = 512` 线程 = 16 warp。
- 共享内存 49KB/block → 最多 2 个 block（2×49=98KB）。
- 所以 2 block × 256 线程 = **512 线程 = 16 warp，就是硬上限**。

### 为什么寄存器用这么多？
核心是**在线 softmax 的累加器（acc）必须常驻寄存器**：
每个 warp 负责 16 行 query × 128 维输出，累加器共 `16×128=2048` 个 fp32，
除以 32 线程 = **每线程 64 个寄存器**，这是算法设计决定的、难以压缩的底数。

### 与 CPU 应用层线程的本质区别
- CPU：可以 spawn 10000 线程（OS 时间片轮转，允许超卖）。
- GPU：**不允许超卖**。资源不够，多余的 block 就**排队不启动**，等资源释放才上。
  所以"拉满"不是你想拉就能拉，而是被寄存器/共享内存卡死的。

### 手动旋钮有哪些（但都有限制）
- `__launch_bounds__(N)` / `-maxrregcount=R`：**强制编译器把寄存器压到 R**，
  代价是**超出部分 spill（溢出）到本地内存**（很慢）。本项目用 `-maxrregcount=128`
  从 137 压到 128；再往下压就得把累加器 spill，得不偿失。

---

## 六、优化技巧术语

### 张量核 / MMA / WMMA
- MMA = Matrix Multiply Accumulate（矩阵乘加），注意力的 Q@Kᵀ、P@V 就是它。
- 张量核 = 专门算 MMA 的硬件。
- WMMA = CUDA 调用张量核的高级 API（`wmma::mma_sync`）。

### Fusion（融合）
多步运算合并成一个 kernel，省掉中间读写。FlashAttention 就是把
QKᵀ+softmax+PV 融合成"分块+在线 softmax"，不写 S×S 大矩阵。

### Online Softmax（在线 softmax）
用 running max / running sum 增量更新，分块时也能算出精确 softmax。

### cp.async（异步拷贝）
内存加载"发出即返回"，后台搬，warp 先干别的。

### Software Pipelining（软件流水线）
把下一块料的准备塞进这一块料的加工空档。本项目尚未做，是逼近 cuDNN 的关键。

### Warp Specialization（warp 专业化）
生产者 warp 专职搬运、消费者 warp 专职计算。
本项目实测**无效**（因为全局内存 stall 仅 7%，搬运不是瓶颈）。

### Padding（共享内存 padding）
把 stride 从 128 加到 136，错开 bank，消除 bank conflict。

---

## 一句话总结

> GPU 是工厂：SM=车间，warp=工人，调度器=工头，张量核=特种机器。
> 工人等待 = stall；车间工人太少 = occupancy 低；机器空转 = 延迟受限。
> occupancy 由寄存器+共享内存硬性决定，不能像 CPU 线程那样随意拉满。
