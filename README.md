# Fintech-Support-Intent-Extractor
# 🏦 Fintech Intent & Root Cause Extractor (QLoRA Fine-Tuning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R8s2HxrpepTv_OyPhJ_ox9ZR0Z1pVIpO?usp=sharing)

## 📌 The Business Problem
Product and Analytics teams are overwhelmed by unstructured customer support tickets. This project automates the categorization of raw, emotional customer feedback into 77 strict banking intents using a fine-tuned Small Language Model (SLM).

## 🛠️ Technical Approach
Instead of relying on expensive API calls to massive foundation models, this project utilizes **Parameter-Efficient Fine-Tuning (PEFT)** to train a model locally on consumer hardware.

* **Base Model:** `TinyLlama-1.1B-Chat`
* **Dataset:** `banking77` (Hugging Face)
* **Optimization:** **QLoRA** (4-bit Quantization + Low-Rank Adaptation). This reduced the trainable parameters to less than 1%, allowing the model to be trained entirely on a free Google Colab T4 GPU (16GB VRAM).

## 📊 Batch Inference Results
After fine-tuning, the model is capable of zero-shot categorization of messy support tickets into structured labels suitable for downstream analytics pipelines (e.g., Looker Studio / PowerBI).

![Inference Results](model_inference_results.png)

## 🚀 How to Run
Click the "Open in Colab" badge above. The notebook is fully self-contained. It will:
1. Install necessary dependencies (`bitsandbytes`, `peft`, `trl`).
2. Load the base model in 4-bit precision.
3. Attach the custom LoRA adapters.
4. Run batch inference and generate visualizations.
