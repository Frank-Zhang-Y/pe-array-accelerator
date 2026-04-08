# Custom Vitis IP Design – PE Array-Based Neural Network Accelerator

## 1. GitHub Repository

Repository link:
https://github.com/username/pe-array-accelerator

This repository contains the full project including RTL design, simulation, and documentation.
The current file (`plan.md`) serves as the high-level design outline for the IP.

---

## 2. Project Team

* Student 1: [Your Name]
* Student 2: [Teammate Name]

---

## 3. IP Definition

### 3.1 Functionality

The proposed IP is a **neural-network-style accelerator** based on a Processing Element (PE) array.
It is designed to accelerate **dot-product and matrix multiplication operations**, which are the core computations in:

* Fully Connected (FC) layers
* Convolution (Conv) layers (after transformation)
* General Matrix Multiplication (GEMM)

---

### 3.2 Mathematical Operations

The core computation performed by the IP is:

[
y = \sum_{i=0}^{N-1} x_i \cdot w_i
]

This represents a **dot product** between an input vector and a weight vector.

More generally, the IP supports:

#### Fully Connected:

[
\mathbf{y} = W \cdot \mathbf{x} + \mathbf{b}
]

#### Matrix Multiplication:

[
C = A \cdot B
]

#### Convolution (after transformation):

[
y = \sum (input \times kernel)
]

---

### 3.3 Pseudocode Representation

```python
# dot product (core operation)
acc = 0
for i in range(N):
    acc += x[i] * w[i]

# post-processing
out = relu(acc + bias)
```

---

### 3.4 Why Hardware Acceleration

These operations are well-suited for hardware acceleration because:

* High computational intensity (MAC operations dominate workload)
* Strong data parallelism
* Regular memory access patterns
* Repetitive structure (ideal for PE arrays)

---

## 4. IP Architecture

### 4.1 Top-Level Architecture

The IP adopts a **streaming dataflow architecture with dual clock domains**:

```text
AXI-Lite Slave (control)
        ↓
Control Registers
        ↓
Control FSM
        ↓
AXI-Stream Slave (input data)
        ↓
Async FIFO (CDC: core_clk → compute_clk)
        ↓
Input Buffer / Loader
        ↓
PE Array (compute core)
        ↓
Post Processing (bias + ReLU)
        ↓
Output Buffer
        ↓
Async FIFO (CDC: compute_clk → core_clk)
        ↓
AXI-Stream Master (output data)
```

---

### 4.2 Clock Domains

* **core_clk domain**

  * AXI-Lite interface
  * AXI-Stream input/output
  * Control FSM

* **compute_clk domain**

  * Input buffer
  * PE array
  * Post-processing

Clock domain crossing is handled using **asynchronous FIFOs**.

---

## 5. Module Description

### 5.1 AXI-Lite Slave

* Provides configuration registers:

  * start
  * mode (FC / Conv)
  * input size
  * output size
* Allows host processor to control the IP

---

### 5.2 Control FSM

* Manages execution flow:

  * start / stop
  * data loading
  * computation trigger
  * completion signaling

---

### 5.3 AXI-Stream Slave (Input)

* Receives input data stream
* Supports valid/ready handshake
* Input may include:

  * feature data
  * weight data (or preloaded separately)

---

### 5.4 Async FIFO (Input Side)

* Handles clock domain crossing
* Buffers incoming data
* Decouples AXI stream from compute pipeline

---

### 5.5 Input Buffer

* Reorganizes input data
* Packs data into vectors (e.g., x0, x1, x2, x3)
* Feeds PE array efficiently

---

### 5.6 PE Array

* Core computation unit
* Composed of multiple PEs

Each PE performs:

```text
acc = acc + a * b
```

Two design options:

* Baseline: 1D PE array (parallel multiply + reduction)
* Advanced: 2D systolic array (dataflow-driven MAC)

---

### 5.7 Weight Buffer

* Stores weights locally
* Enables data reuse
* Reduces memory bandwidth

---

### 5.8 Post Processing

* Bias addition
* Activation (ReLU)
* Optional scaling (quantization support)

---

### 5.9 Output Buffer

* Temporarily stores results
* Handles mismatch between compute and output rates

---

### 5.10 Async FIFO (Output Side)

* Transfers data from compute domain to core domain
* Ensures safe CDC

---

### 5.11 AXI-Stream Master (Output)

* Streams computed results to external system
* Uses valid/ready handshake

---

## 6. Interface Strategy

* **AXI-Lite** → control path
* **AXI-Stream** → high-throughput data path
* **FIFO-based buffering** → decouples modules
* **Async FIFO** → clock domain crossing

---

## 7. Design Strategy

* Modular design (each block independently verifiable)
* Streaming architecture for high throughput
* Parameterizable PE array (scalable)
* Separation of control path and data path

---

## 8. Future Extensions

* 2D systolic array implementation
* Convolution mode with sliding window generator
* Sparse computation support
* Quantization (INT8 / BF16)
* FPGA synthesis and performance evaluation

---

## 9. Expected Outcome

* Fully functional custom Vitis IP
* Verified RTL implementation
* Scalable accelerator architecture
* Ready for integration into SoC or FPGA system

---
