# 🛠️ Edge AI for Bearing Fault Diagnosis

> **Efficient bearing fault diagnosis using compact 1D-CNNs, structured pruning, INT8 quantization, cross-dataset generalization, and MMD-based domain adaptation.**

## 📌 Overview

This project investigates **vibration-based bearing fault diagnosis using Deep Learning**, with a particular focus on **efficient deployment for Edge AI and TinyML applications**.

The project goes beyond achieving high classification accuracy on a single dataset. It investigates the complete machine learning pipeline, including:

* Leakage-aware data splitting
* Compact 1D-CNN design
* Structured neural network pruning
* L2 vs. Taylor filter importance comparison
* FP16 and INT8 quantization
* Accuracy–size trade-offs for Edge deployment
* Cross-dataset generalization
* Domain shift analysis
* MMD-based unsupervised domain adaptation

The goal is to develop models that are not only accurate but also **compact, efficient, and critically evaluated for real-world generalization**.

---

# 🎯 Objectives

The main objectives of this project are:

1. Develop a compact 1D-CNN for bearing fault diagnosis from vibration signals.
2. Prevent data leakage using file-level grouped data splitting.
3. Reduce model complexity using structured filter pruning.
4. Compare L2 magnitude-based and Taylor-based pruning strategies.
5. Apply FP16 and INT8 quantization for efficient deployment.
6. Identify optimal accuracy–compression trade-offs.
7. Evaluate model generalization across different bearing datasets.
8. Investigate domain shift and dataset-specific shortcut features.
9. Explore MMD-based domain adaptation for cross-domain transfer.

---

# 📊 Datasets

The project investigates bearing fault diagnosis and cross-dataset transfer using:

* **CWRU — Case Western Reserve University Bearing Dataset**
* **Paderborn University Bearing Dataset**

The fault diagnosis task includes bearing conditions such as:

* 🟢 Normal
* 🔵 Ball Fault
* 🟠 Inner Race Fault
* 🔴 Outer Race Fault

---

# 🧠 Model Architecture

A compact **1D Convolutional Neural Network (1D-CNN)** is used to process raw vibration signals.

The architecture consists of:

```text
Input Vibration Signal
        ↓
Conv1D + Batch Normalization + MaxPooling
        ↓
Conv1D + Batch Normalization + MaxPooling
        ↓
Conv1D
        ↓
Global Average Pooling
        ↓
Dense + Dropout
        ↓
Softmax Classification
```

The architecture was designed with efficiency in mind, making it suitable for later optimization and deployment on resource-constrained devices.

---

# 🔒 Leakage-Aware Evaluation

A major focus of this project is preventing **data leakage**.

Instead of randomly splitting individual signal windows, the dataset is split at the **source-file level**.

This ensures that windows originating from the same vibration recording cannot appear simultaneously in training and testing datasets.

```text
Source Files
     ↓
Grouped File-Level Split
     ↓
Training Files       Test Files
     ↓                   ↓
Signal Windows      Signal Windows
```

This provides a more reliable evaluation than a random window-level split.

---

# ✂️ Structured Pruning

To reduce model complexity, this project implements **structured filter pruning**.

Unlike unstructured pruning, structured pruning removes complete convolutional filters, resulting in a physically smaller neural network.

Two filter importance criteria are investigated:

## L2 Magnitude-Based Pruning

Filters with smaller L2 norms are considered less important and can be removed.

## Taylor-Based Pruning

Filter importance is estimated using a first-order Taylor approximation of the effect of removing a feature.

The methods are compared under matched pruning ratios.

---

# ⚡ Quantization

The optimized models are converted to TensorFlow Lite and evaluated using:

* FP32
* FP16
* INT8

INT8 quantization is particularly relevant for embedded and Edge AI deployment.

---

# 🏆 Best Compression Result

The strongest Edge AI candidate obtained in the original compression experiments is:

| Metric         |            Result |
| -------------- | ----------------: |
| Model          | Pruned 50% + INT8 |
| Accuracy       |        **97.03%** |
| Macro-F1       |        **0.9743** |
| Model Size     |       **17.1 KB** |
| Size Reduction |          **~76%** |

This demonstrates that significant model compression can be achieved while maintaining strong intra-dataset classification performance.

---

# 📈 Compression Pipeline

```text
FP32 Compact CNN
        ↓
Structured Filter Pruning
        ↓
Model Reconstruction
        ↓
Fine-Tuning
        ↓
FP16 / INT8 Quantization
        ↓
TensorFlow Lite Model
        ↓
Edge AI Deployment Candidate
```

---

# 🌍 Cross-Dataset Generalization

High accuracy on a single dataset does not necessarily imply robust real-world generalization.

This project therefore evaluates cross-dataset transfer between:

```text
CWRU  →  Paderborn

Paderborn  →  CWRU
```

The experiments reveal significant domain shift between the two datasets.

The model performs strongly within its original dataset but experiences substantial degradation when transferred to a different experimental domain.

This highlights the limitations of evaluating bearing diagnosis systems exclusively using intra-dataset accuracy.

---

# 🔬 Shortcut Feature Investigation

The cross-dataset experiments suggest that the model may partially rely on **dataset-specific characteristics**, such as signal amplitude or energy distributions.

This raises an important research question:

> Is the model learning universal physical signatures of bearing faults, or is it exploiting dataset-specific shortcuts?

The project investigates this question through cross-dataset analysis and domain adaptation experiments.

---

# 🌐 MMD-Based Domain Adaptation

To reduce the distribution gap between source and target domains, the project explores **Maximum Mean Discrepancy (MMD)** based domain adaptation.

The training objective is:

$$
L = L_{classification} + \lambda_{MMD}L_{MMD}
$$

where:

* \(L_{classification}\) represents source-domain classification loss.
* \(L_{MMD}\) measures the discrepancy between source and target feature distributions.
* \(\lambda_{MMD}\) controls the importance of domain alignment.

A source-only control experiment is performed using:

```text
λ_MMD = 0
```

and compared against MMD-based adaptation.

Current experiments show that domain alignment must be carefully tuned and that MMD-based feature alignment does not automatically guarantee improved target-domain classification.

---

# 🧪 Experimental Research Pipeline

```text
High Intra-Dataset Accuracy
            ↓
Structured Pruning
            ↓
INT8 Quantization
            ↓
Compact Edge AI Model
            ↓
Cross-Dataset Evaluation
            ↓
Generalization Failure
            ↓
Domain Shift Investigation
            ↓
Shortcut Feature Hypothesis
            ↓
MMD Domain Adaptation
            ↓
Control Experiment
            ↓
Systematic λ Evaluation
```

---




# 🚀 Key Takeaways

This project demonstrates that:

* High intra-dataset accuracy does not guarantee cross-dataset generalization.
* Compact neural networks can achieve strong performance for bearing fault diagnosis.
* Structured pruning and INT8 quantization can significantly reduce model size.
* L2 and Taylor pruning strategies can behave differently depending on the architecture and experimental configuration.
* Domain shift remains a major challenge in vibration-based fault diagnosis.
* Simple feature alignment using MMD may not be sufficient to overcome significant cross-domain differences.

---

# 🔮 Future Work

Future research directions include:

* Systematic MMD hyperparameter optimization.
* Multiple random seed experiments and statistical reporting.
* Improved domain adaptation techniques.
* Class-conditional domain alignment.
* Feature normalization strategies.
* Domain-adversarial learning.
* Real microcontroller deployment and benchmarking.
* Flash, RAM, latency, and energy measurements on embedded hardware.

---

# 🛠️ Technologies

* Python
* TensorFlow / Keras
* TensorFlow Lite
* NumPy
* Scikit-learn
* Structured Neural Network Pruning
* INT8 Quantization
* Domain Adaptation
* Maximum Mean Discrepancy (MMD)

---

# 📄 Research Focus

**Edge AI · TinyML · Deep Learning · Condition Monitoring · Predictive Maintenance · Time-Series Classification · Model Compression · Domain Adaptation**
