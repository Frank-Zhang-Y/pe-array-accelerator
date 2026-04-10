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

This project implements a **custom Vitis hardware IP** that accelerates inference for a single fully connected neural network layer using fixed-point arithmetic.

The design starts from a baseline single-MAC architecture and is extended with a more structured compute core based on a small Processing Element (PE) array. This allows the accelerator to exploit parallelism while keeping the overall functionality unchanged.

In addition, the post-processing stage is slightly extended to better reflect a practical inference pipeline.

The IP computes:

y_i = max(0, sum W[i][j] * x[j] + b[i])

with an additional optional scaling/clipping step for fixed-point output.

---

### Functionality

1. Receives an input vector x (streamed)
2. Stores the input internally
3. Computes matrix-vector multiplication using a PE array
4. Adds bias
5. Applies ReLU activation
6. Applies optional scaling/clipping
7. Outputs result y

---

### Mathematical Operations

for i in range(M):
    acc = 0
    for j in range(N):
        acc += W[i][j] * x[j]
    acc += b[i]
    acc = max(0, acc)
    y[i] = clip_and_scale(acc)

---

## Why This Is Suitable for Hardware Acceleration

- High arithmetic intensity
- Deterministic computation
- Data reuse
- Parallelism via PE array
- Common ML workload

---

# IP Architecture

## High-Level Design

- Input Buffer
- Weight/Bias Memory
- PE Array
- Bias + ReLU + Scaling
- Control FSM
- Output Buffer

---

## Dataflow

Input → Buffer → PE Array → Post-process → Output

---

## Summary

PE array + data reuse + scaling improve architecture while keeping scope manageable.
