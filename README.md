# Real-Time Detection of Slow DDoS Attacks in IoT 

## 1. Project Overview

The purpose of this project is to implement **real-time detection of Slow DDoS attacks in IoT environments using resource-constrained devices**.

Unlike traditional detection systems that rely on high-performance servers, this implementation focuses on deploying and evaluating machine learning–based detection mechanisms on low-power embedded and edge hardware. The goal is to validate feasibility, performance, and detection effectiveness under constrained computational and memory conditions commonly found in real-world IoT scenarios.

---

## 2. Testbed Platform

The experimental testbed consists of three devices with different computational capabilities, enabling cross-platform deployment and performance evaluation.

### 2.1 PC Platform

**Processor:** AMD Ryzen 5 5600X 6-Core Processor @ 4.58 GHz
**RAM:** 32.0 GB
**Operating System:** Windows 11 Professional, Version 24H2

The PC is primarily used for:

* Data preprocessing
* Model training
* Performance evaluation
* Dataset management

---

### 2.2 Raspberry Pi 4

**Model:** Raspberry Pi 4 Model B Rev 1.1
**Operating System:** Debian GNU/Linux 12 (Bookworm)
**Kernel Version:** 6.12.25+rpt-rpi-v8
**CPU:** Quad-core @ 1.500 GHz
**Memory:** 1846 MiB (~2 GB)

The Raspberry Pi 4 serves as an **edge computing node**, responsible for:

* Running trained detection models
* Handling network traffic feature inputs
* Performing real-time inference tests

---

### 2.3 Raspberry Pi Pico 2W

**Processor:** Dual Arm Cortex-M33 / Hazard3 RISC-V cores @ 150 MHz
**Memory:** 520 KB on-chip SRAM
**Wireless:** 2.4 GHz 802.11n WLAN, Bluetooth 5.2

**System Information:**

* sysname: `rp2`
* release: `1.25.0`
* build: v1.25.0 (2025-04-15)

**Python Environment:**
MicroPython v1.25.0 (Python 3.4.0 core)

The Pico 2W represents a **highly resource-constrained IoT device**, used to validate:

* Lightweight model deployment feasibility
* Memory footprint constraints
* Real-time inference latency

---

## 3. Dataset

This project uses the **CIC IoT 2023 Dataset**, developed by the Canadian Institute for Cybersecurity (CIC). The dataset is designed to support intrusion detection research in realistic IoT environments.

* Total files: 169
* Total size: ~12.8 GB

**Full dataset download link:**
https://www.unb.ca/cic/datasets/iotdataset-2023.html

---

## 4. Data Preprocessing

Effective preprocessing is critical to ensure model validity and detection performance. The main objectives are:

* Removing redundant data
* Focusing on Slow DDoS traffic
* Achieving dataset balance

### 4.1 Dataset Merging

Since the original dataset consists of **169 separate files**, all files are first merged into a unified dataset to facilitate large-scale processing and analysis.

---

### 4.2 Irrelevant Traffic Removal

All attack categories other than **Slow DDoS** are removed.

Only the following classes are retained:

* Benign Traffic
* Slow DDoS Traffic

This step ensures the dataset aligns with the project’s binary classification objective.

---

### 4.3 Data Balancing

There is a significant volume imbalance between:

* **BenignTraffic**
* **Slow DDoS**

To ensure model effectiveness and avoid bias toward the majority class, data balancing techniques are applied, including controlled sampling to produce a more evenly distributed training dataset.

---
## 5. Functional Description

This section describes the main functional modules and directory structure of the project repository.

---

### 5.1 `Dataset/CICIoT2023`

This directory contains the **fully processed and balanced dataset** used for training and evaluation.

Key characteristics:

* Merged from the original 169 raw files
* Irrelevant attack categories removed
* Retains only:

  * Benign Traffic
  * Slow DDoS Traffic
* Class imbalance has been addressed through data balancing techniques

This dataset is ready for direct use in model training and testing without additional preprocessing.

---

### 5.2 `Data Preprocessing`

This module includes all scripts related to dataset preparation.

Main functions:

* Raw dataset merging
* Traffic filtering (Slow DDoS extraction)
* Removal of redundant or invalid records
* Feature cleaning and normalization
* Dataset balancing

These scripts reproduce the full preprocessing pipeline from the original CICIoT2023 dataset to the finalized training dataset.

---

### 5.3 `models`

This directory stores the machine learning and deep learning models developed for Slow DDoS detection.

Contents may include:

* Model architecture definitions
* Training scripts
* Evaluation scripts
* Exported model weights
* Lightweight model variants for embedded deployment

These models are optimized for inference on resource-constrained IoT hardware.

---

### 5.4 `data_visualisation_image`

This folder contains data visualization outputs generated during exploratory data analysis and preprocessing validation.

---

### 5.5 `send_csv_to_*`

Due to the **performance and storage limitations of the Raspberry Pi Pico**, the full dataset cannot be stored locally on the device.

Therefore, this module provides scripts that:

* Read CSV samples from the PC
* Transmit data batches to the hardware device
* Enable real-time inference testing on the Pico
* Receive prediction results back to the host system

This mechanism allows large-scale testing while keeping the embedded device lightweight and memory-efficient.

---
### 5.6 `Pi_Pico*`

This directory contains all MicroPython firmware and deployment files designed specifically for the Raspberry Pi Pico 2W.

It enables on-device inference and network communication required for real-time Slow DDoS detection.

Main components include:

* WiFi Configuration Script
Handles wireless network setup and connection, allowing the Pico to communicate with external hosts (e.g., PC or Raspberry Pi).
* Logistic Regression (LR) Model
A lightweight statistical detection model optimized for ultra-low resource consumption and fast inference.
* MLP Model
A compact Multi-Layer Perceptron implementation adapted for MicroPython execution on constrained hardware.
* MLP Weight Files
Pretrained weight parameters exported from the training environment and converted into MicroPython-compatible format for on-device loading and inference.

These components collectively enable the Pico to:

* Receive feature data over WiFi
* Execute real-time attack detection=
* Return prediction results to the host system






