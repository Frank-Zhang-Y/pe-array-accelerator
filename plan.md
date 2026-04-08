# Custom Vitis IP Design – PE Array-Based Neural Network Accelerator

## 1. GitHub Repository

Repository link:
https://github.com/username/pe-array-accelerator

This repository contains the full project including RTL design, simulation, and documentation.
This `plan.md` provides the high-level design outline.

---

## 2. Project Team

* Student 1: Temira Koenig
* Student 2: Zihao Zhang

---

## 3. IP Definition

### 3.1 Functionality

This IP is a neural-network-style accelerator based on a Processing Element (PE) array.

It accelerates core operations used in:

* Fully Connected (FC)
* Convolution (Conv)
* Matrix Multiplication (GEMM)

---

### 3.2 Core Computation (Simplified)

The core operation is a **dot product**:

```text
y = x0*w0 + x1*w1 + x2*w2 + ... + xN*wN
```

This is implemented using repeated MAC operations:

```text
acc = acc + (x * w)
```

---

### 3.3 Higher-Level Operations

#### Fully Connected (FC)

```text
y = W * x + bias
```

#### Matrix Multiplication (GEMM)

```text
C[i][j] = sum over k: A[i][k] * B[k][j]
```

#### Convolution (Conv)

```text
y = sum(input_window * kernel)
```

(All of the above are internally mapped to dot-product operations)

---

### 3.4 Pseudocode

```python
acc = 0
for i in range(N):
    acc += x[i] * w[i]

out = relu(acc + bias)
```

---

### 3.5 Why Hardware Acceleration (Detailed)

The target operations (dot product, FC, GEMM, Conv) are highly suitable for hardware acceleration due to their **regular structure, high parallelism, and streaming-friendly dataflow**.

---

### 1. Parallelism (PE Array)

The core computation:

```text
y = x0*w0 + x1*w1 + ... + xN*wN
```

can be parallelized.

* Software (CPU):

  ```text
  for i in range(N):
      acc += x[i] * w[i]   # sequential
  ```

* Hardware (PE array):

  ```text
  p0 = x0*w0
  p1 = x1*w1
  p2 = x2*w2
  p3 = x3*w3
  ...
  sum = p0 + p1 + p2 + p3
  ```

👉 Multiple multiplications happen **in the same cycle**, significantly improving throughput.

---

### 2. Pipelining (Key Speedup Mechanism)

The computation is divided into pipeline stages:

```text
Stage 1: load data (x, w)
Stage 2: multiplication (x * w)
Stage 3: partial sum (adder tree or MAC)
Stage 4: post-processing (bias + ReLU)
```

Once pipelined:

```text
Cycle 1: Stage1 (data load)
Cycle 2: Stage1 + Stage2
Cycle 3: Stage1 + Stage2 + Stage3
Cycle 4: Stage1 + Stage2 + Stage3 + Stage4
Cycle 5+: one result per cycle
```

👉 After pipeline fill, the system achieves:

```text
Throughput ≈ 1 output per cycle
```

Even though latency is multiple cycles, **throughput is maximized**.

---

### 3. Streaming Dataflow

The design uses AXI-Stream and FIFO-based buffering:

```text
Input Stream → FIFO → PE Array → Output Stream
```

Key advantages:

* No need to store full matrices in memory
* Continuous data processing (no idle cycles)
* Natural support for large workloads

👉 Data is processed **as it arrives**, enabling high utilization.

---

### 4. Data Reuse (Critical Optimization)

In neural network workloads:

* The same weights are reused across multiple inputs
* The same input data may be reused across multiple outputs

With local buffers:

```text
Weight Buffer → reused across many MAC operations
```

👉 Reduces external memory access (which is the real bottleneck)

---

### 5. Reduced Memory Bandwidth Pressure

CPU/GPU style:

```text
load x → compute → store → reload → compute ...
```

Hardware accelerator:

```text
load once → reuse many times inside PE array
```

👉 Significantly reduces bandwidth requirements and improves efficiency.

---

### 6. MAC-Dominated Workload

Neural network computation is dominated by:

```text
acc += a * b
```

Hardware can implement this directly as:

* Dedicated multipliers (DSP blocks)
* Deep pipelines
* Parallel PE arrays

👉 Achieves much higher efficiency than general-purpose processors.

---

### 7. Dataflow-Driven Execution

Unlike CPUs (instruction-driven), this design is:

```text
data-driven (streaming)
```

* Computation starts when data arrives
* No instruction fetch/decode overhead
* High hardware utilization

---

### Summary

Hardware acceleration improves performance through:

* Parallel computation (PE array)
* Deep pipelining (high throughput)
* Streaming dataflow (continuous processing)
* Data reuse (reduced memory traffic)

These together enable:

```text
High throughput + Low latency + High energy efficiency
```


## 4. IP Architecture

### 4.1 Top-Level Architecture

```text
AXI-Lite Slave (control)
        ↓
Control Registers
        ↓
Control FSM
        ↓
AXI-Stream Slave (input)
        ↓
Async FIFO (core_clk → compute_clk)
        ↓
Input Buffer
        ↓
PE Array
        ↓
Post Processing (bias + ReLU)
        ↓
Output Buffer
        ↓
Async FIFO (compute_clk → core_clk)
        ↓
AXI-Stream Master (output)
```

---

### 4.2 Clock Domains

* core_clk domain:

  * AXI interfaces
  * control logic

* compute_clk domain:

  * PE array
  * data processing

Clock crossing handled by async FIFO.

---

## 5. Module Description

### AXI-Lite Slave

* Configuration interface
* Controls start, mode, size

---

### Control FSM

* Controls execution flow
* Issues start/done signals

---

### AXI-Stream Slave

* Receives input data
* Uses valid/ready handshake

---

### Async FIFO (Input)

* Handles clock domain crossing
* Buffers input data

---

### Input Buffer

* Packs data into vectors
* Feeds PE array

---

### PE Array

* Core compute engine
* Each PE performs:

```text
acc += a * b
```

Supports:

* 1D array (baseline)
* 2D systolic array (advanced)

---

### Weight Buffer

* Stores weights locally
* Enables reuse

---

### Post Processing

* bias addition
* ReLU activation
* optional scaling

---

### Output Buffer

* Buffers results
* Matches output rate

---

### Async FIFO (Output)

* compute → core clock crossing

---

### AXI-Stream Master

* Sends output data
* Uses valid/ready protocol

---

## 6. Interface Strategy

* AXI-Lite → control
* AXI-Stream → data
* FIFO → buffering
* Async FIFO → clock crossing

---

## 7. Design Strategy

* Modular design
* Streaming pipeline
* Scalable PE array
* Separate control and data paths

---

## 8. Future Work

* 2D systolic array
* Convolution sliding window
* Sparse support
* Quantization (INT8 / BF16)

---

## 9. Expected Outcome

* Working Vitis IP
* Verified RTL
* Scalable accelerator design

---
