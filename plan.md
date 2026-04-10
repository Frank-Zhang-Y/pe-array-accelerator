# Initial Plan — Fixed-Point NN Accelerator

## Project Team

- Temira Koenig  
- Zihao Zhang

---

## Note:
We may still increase the scope of this project. This is what we would like to build as of now, but we may change some things.

---

## 1. IP Definition

### Overview

This project implements a custom Vitis hardware IP that accelerates inference for a single fully connected neural network layer using fixed-point arithmetic.

The design starts from a baseline single-MAC architecture and is extended with a more structured compute core based on a small Processing Element (PE) array. This allows the accelerator to exploit parallelism while keeping the overall functionality unchanged.

In addition, the post-processing stage is slightly extended to better reflect a practical inference pipeline.

The IP computes:

$$
y_i = \max\left(0, \sum_{j=0}^{N-1} W[i][j] \cdot x[j] + b[i]\right)
$$

with an additional optional scaling/clipping step for fixed-point output.

---

### Functionality

The IP performs the following steps:

1. Receives an input vector x (streamed sequentially)
2. Stores the input internally
3. Computes matrix-vector multiplication using a PE array
4. Adds bias
5. Applies ReLU activation
6. Applies optional scaling/clipping for fixed-point output
7. Outputs the result vector y

---

### Mathematical Operations

For each output element:

1. Multiply input elements by weights  
2. Accumulate partial sums  
3. Add bias  
4. Apply ReLU activation  
5. Apply scaling and clipping  


Equivalent pseudocode:

```python
for i in range(M):
    acc = 0
    for j in range(N):
        acc += W[i][j] * x[j]
    acc += b[i]
    acc = max(0, acc)
    y[i] = clip_and_scale(acc)

## Why This Is Suitable for Hardware Acceleration

- High arithmetic intensity: many multiply-accumulate operations
- Deterministic computation pattern: fixed loops over i and j
- Data reuse: input vector reused across multiple outputs
- Parallelism potential: MAC operations can be parallelized using PE arrays
- Common workload: dense layers are fundamental in ML inference

These properties make the computation well-suited for a custom hardware IP.

These characteristics also guide the architectural design of the accelerator. 
To efficiently exploit parallelism and data reuse, the computation is mapped onto a structured hardware architecture as described below.

---

# IP Architecture

## High-Level Design

The design is modular and consists of the following components:

- Input Buffer
- Weight/Bias Memory
- PE Array (Compute Core)
- Bias + ReLU + Scaling Unit
- Control FSM
- Output Buffer

---

## Module Descriptions

### 1. Input Buffer

- Stores the input vector x
- Receives streamed input data
- Provides indexed access during computation
- Enables reuse of input data across multiple computations

---

### 2. Weight/Bias Memory

- Stores weight matrix W and bias vector b
- Provides values to the compute core
- Can be implemented using internal memory (BRAM-style arrays)

---

### 3. PE Array (Compute Core)

- Performs parallel multiply-accumulate operations
- Consists of multiple Processing Elements (PEs)
- Each PE computes: acc += W[i][j] * x[j]
- Input data is broadcast to multiple PEs
- Enables parallel computation across multiple output elements
- Supports parameterization (e.g., number of PEs)

---

### 4. Bias + ReLU + Scaling Unit

- Adds bias to accumulated result
- Applies activation: y = max(0, acc)
- Applies optional scaling and clipping for fixed-point output
- Produces final output value

---

### 5. Control FSM

- Controls execution flow
- Handles:
  - input loading
  - iteration over i and j
  - PE control (enable / accumulate / reset)
  - output generation
- Maintains counters and control signals

---

### 6. Output Buffer

- Stores computed output values
- Streams output vector to the host system

---

## Interface with Host (PS)

The IP will be structured as a Vitis-compatible custom IP with the following interfaces:

### AXI4-Stream (Data Path)

- Input vector x streamed into the accelerator
- Output vector y streamed out

### AXI4-Lite (Control)

- Control registers:
  - start
  - done
  - busy
  - optional configuration (dimensions, etc.)

---

## Dataflow

Input Stream → Input Buffer → PE Array → Post-Processing → Output Buffer → Output Stream
                               ↑
                       Weight/Bias Memory
                               ↑
                         Control FSM

In this design:

- Input activations are loaded once and reused across multiple computations
- The same input value is broadcast to multiple PEs
- This enables parallel execution and reduces memory access overhead

---

## Architectural Choice

The baseline design uses a time-multiplexed (serial) MAC engine.  
In this project, we extend it to a small PE-array-based architecture.

### Justification

- Improves parallelism compared to a single MAC
- Maintains manageable design complexity
- Enables data reuse through input buffering and broadcast
- Reflects a more realistic accelerator-style design
- Still fits within project constraints

---

## Modularity and Partitioning

The design is explicitly partitioned into separate modules to:

- Enable independent unit testing
- Support incremental development
- Allow clear debugging boundaries
- Provide a path to future extensions (e.g., larger PE arrays)

---

## Use of Existing AMD IP

Because this project is implemented as a custom Vitis IP, existing AMD/Xilinx IP blocks may be used where appropriate to simplify system integration.

This includes:

- AXI4-Stream interface modules
- AXI4-Lite control interface
- FIFOs and buffering support

However, the core functionality — including the PE array, dataflow, control logic, and post-processing — will be implemented as custom RTL.

---

## Design Extension

Compared to the baseline single-MAC design, this project introduces two key extensions:

1. PE Array-Based Compute  
   - Replaces the single MAC with a small array of processing elements  
   - Enables parallel computation of multiple output elements  

2. Enhanced Post-Processing  
   - Extends bias and ReLU with optional scaling/clipping  
   - Better reflects practical fixed-point inference pipelines  

---

## Summary

This project defines a modular, fixed-point neural network inference accelerator implemented as a custom Vitis IP.

The design enhances a baseline implementation by introducing:

- A PE-array-based compute structure
- Data reuse through buffering and broadcast
- A more complete post-processing pipeline

This makes the design closer to a realistic neural network accelerator while remaining feasible within the project scope.
