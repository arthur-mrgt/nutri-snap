# NutriSnap 🍽️📸

NutriSnap is a multimodal AI project that aims to estimate the nutritional content of meals from images. By combining food recognition, image segmentation, depth prediction, and optional text input, NutriSnap provides fast and accurate assessments of calories, proteins, carbohydrates, and fats — helping users make better dietary choices with less effort.

---

## 🧠 Motivation

Tracking nutrition is essential for health, but most apps require tedious manual input. NutriSnap automates this process using state-of-the-art transformer models, enabling nutrient estimation from just a photo and optional textual metadata. This project was developed as part of the EPFL COM-304 course.

---

## 🎯 Objectives

- Automatically identify ingredients from an image of a meal.
- Estimate each ingredient’s weight using segmentation and depth prediction.
- Retrieve nutritional values from databases like Edamam.
- Explore the impact of additional modalities (e.g., text) on model performance.
- Achieve fast and accurate estimations to encourage healthy eating habits.

---

## 🔍 Research Questions

1. How accurately can a multimodal model estimate a meal’s nutritional content?
2. How do additional modalities (text, depth) improve estimation quality?

---

## 🛠️ Methodology

### 📊 Datasets
- **SEG-FOOD**: For training the semantic segmentation stage (RGB → Segmentation).
- **Nutrition5k**: For depth prediction and per-ingredient mass estimation (RGB/Segmentation → Depth → Mass).

### 🔁 Pipeline Overview
1. **Image Input** (RGB + optional text)
2. **Semantic Segmentation** (Ingredient detection)
3. **Depth Estimation** (3D understanding of food portions)
4. **Mass Estimation** (Per-ingredient quantity)
5. **Nutritional Calculation** (Query API or predict directly)

### 🧩 Model Architecture
- Fine-tuned **nano4M**, an encoder-decoder multimodal transformer.
- Task-specific heads for segmentation, depth, and mass.
- Optional integration of text embeddings.
- Joint fine-tuning strategy across datasets to mitigate domain shift.

---

## 🧪 Evaluation

- **Segmentation**: Mean IoU on SEG-FOOD test set.
- **Depth & Mass Prediction**: MAE and RMSE on Nutrition5k.
- **End-to-End Accuracy**: Performance of the entire pipeline on held-out Nutrition5k data.
- **A/B Testing**: Compare unimodal vs multimodal models.

---

## 🧬 Key Extensions

- ✅ Joint Fine-Tuning with SEG-FOOD and Nutrition5k
- ✅ Architecture Modifications (task-specific decoders)
- ✅ Optional Text Modality
- ⚙️ Span Masking for better training robustness
- ⏳ (Optional) Classifier-Free Guidance & Optimizer Variants
- ⚡ (Optional) Speed Run for real-time use cases

---

## 🗓️ Timeline Overview

| Week | Milestone |
|------|-----------|
| 10   | Dataset setup and nano4M pipeline initialization |
| 11   | Start joint fine-tuning and initial results |
| 12   | Integrate text modality, apply Span Masking |
| 13   | Final model tuning and validation |
| 14   | Evaluation, documentation, and reporting |

---

## ⚠️ Limitations & Risks

- Misclassification in complex/mixed dishes
- Visual-only estimation limitations
- Domain shift between datasets
- Dataset bias and privacy concerns
- Risk of reinforcing disordered eating habits

---

## 📚 References

1. H. Nogay et al., “Image-based food groups and portion prediction,” *Food Science Wiley*, 2025. [DOI](https://ift.onlinelibrary.wiley.com/doi/10.1111/1750-3841.70116)
2. F. Konstantakopoulos et al., “Food weight estimation with boosting algorithms,” *Scientific Reports*, 2023. [DOI](https://doi.org/10.1038/s41598-023-47885-0)

---

## 💬 Contact

For questions or contributions, feel free to open an issue or submit a pull request!

---


This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
