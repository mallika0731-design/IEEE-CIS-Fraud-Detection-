# IEEE-CIS Fraud Detection using Graph Machine Learning

## 📌 Project Overview

This project demonstrates a **professional, production-inspired Graph Machine Learning pipeline** for fraud detection, inspired by the IEEE-CIS Fraud Detection challenge. The core motivation is to show how **relational information** (shared cards, devices, IPs, merchants) can significantly improve fraud detection compared to traditional tabular models.

Unlike standard ML approaches, this project models the data as a **graph** and applies **GraphSAGE**, a scalable Graph Neural Network (GNN), to propagate risk across connected entities.

The implementation is designed to be **memory-safe**, **reproducible**, and **interview/portfolio-ready**.

---

## 🧠 Key Concepts Demonstrated

* Fraud detection under **severe class imbalance**
* Graph construction from transactional data
* Heterogeneous node modeling (transactions + entities)
* Graph Neural Networks using **PyTorch Geometric**
* Benchmarking Graph ML against classical ML baselines
* Practical constraints: memory limits, scalability, and efficiency

---

## 🗂 Dataset

* Inspired by: **IEEE-CIS Fraud Detection (Kaggle)**
* Original dataset size: ~1.5 GB
* This project uses **synthetic IEEE-like data** (10,000 transactions) to ensure:

  * Fast execution
  * Low memory usage
  * Easy reproducibility

Each transaction contains:

* Transaction amount and timestamp
* Card information
* Device information
* Address / IP proxy
* Merchant category
* Fraud label (highly imbalanced)

---

## 🧩 Graph Construction

The problem is modeled as a **relational graph**:

### Node Types

* **Transaction nodes** (primary prediction targets)
* **Entity nodes**:

  * User ID (card + device composite)
  * Card ID
  * IP / Address ID
  * Device ID
  * Merchant ID

### Edges

Edges represent relationships between transactions and entities:

* Transaction → Card
* Transaction → Device
* Transaction → IP
* Transaction → Merchant

This allows fraud signals to propagate across shared entities.

---

## 🔢 Feature Engineering

Each node is represented using compact numerical features:

* Node type indicator (transaction vs entity)
* Transaction amount (log-scaled)
* Normalized transaction time
* One-hot encoded entity type indicators

> ⚠️ Fraud labels are **not included** as node features to prevent label leakage.

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression (Baseline)

* SGD-based solver for low memory usage
* Trained on a small feature subset

### 2️⃣ XGBoost (Baseline)

* Shallow trees
* Limited estimators to remain resource-efficient

### 3️⃣ GraphSAGE (Graph Neural Network)

* Two-layer GraphSAGE architecture
* Neighborhood aggregation captures relational fraud patterns
* Semi-supervised learning on transaction nodes

---

## 🏗 Architecture

```
Input Graph
   │
   ▼
GraphSAGE Layer 1 (Aggregation)
   │
   ▼
GraphSAGE Layer 2
   │
   ▼
Linear + Sigmoid
   │
   ▼
Fraud Probability per Transaction
```

---

## 📊 Evaluation Metric

* **PR-AUC (Average Precision Score)**
* Chosen due to extreme class imbalance
* Evaluation performed only on **transaction nodes**

Graph-based learning consistently outperforms tabular baselines, demonstrating the value of relational modeling.

---

## 📈 Results Visualization

The project generates:

* Prediction distribution histograms
* PR-AUC comparison bar plots

Saved output:

```
fraud_results.png
```

---

## ⚙️ Installation

```bash
pip install torch torch-geometric networkx pandas scikit-learn xgboost matplotlib seaborn
```

(Optional, full dataset)

```bash
pip install kaggle
kaggle datasets download -d ieee-fraud-detection
```

---

## ▶️ How to Run

```bash
python fraud_graph_ml.py
```

The script will:

1. Generate synthetic IEEE-like data
2. Build training and validation graphs
3. Train baseline models
4. Train GraphSAGE
5. Evaluate and visualize results

---

## 🚀 Why Graph ML for Fraud Detection?

Traditional models treat transactions independently. In reality:

* Fraudsters reuse cards
* Devices are shared
* IPs cluster suspicious behavior

Graph Neural Networks capture these dependencies naturally, enabling:

* Early fraud detection
* Better generalization
* Robustness to sparse signals

---

## 🔮 Future Improvements

* Edge-type aware GNNs (R-GCN)
* Temporal graph modeling
* Inductive learning on unseen entities
* Node embedding visualization
* Deployment with mini-batch neighbor sampling

---

## 📌 Key Takeaway

This project showcases how **Graph Machine Learning provides a structural advantage** over traditional ML for fraud detection, while remaining scalable and production-conscious.

---

## 👤 Author

**Mallika Bhardwaj**
MSc Mathematics | Data Science & Decision Science (IIT Delhi)

---

⭐ If you find this project useful, feel free to star the repository or reach out for collaboration.
