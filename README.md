# NutriSnap: A Unified Generative Approach to Image-Based Nutritional Analysis

**NutriSnap** is a research project that validates a unified, end-to-end approach to nutritional analysis from a single food image. By fine-tuning a single generative model, the project demonstrates a sequential pipeline that predicts depth, performs semantic segmentation to identify ingredients, and generates a detailed caption with estimated mass and nutritional values.

This project was developed for the **EPFL COM-304 Final Project**.

## Table of Contents
- [Abstract](#abstract)
- [Installation & Usage](#installation--usage)
- [Project Architecture](#project-architecture)
- [Methodology](#methodology)
- [Results and Findings](#results-and-findings)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)
- [License](#license)

## Abstract

Manual nutritional tracking is a significant barrier to maintaining a healthy diet. Current automated solutions are often fragmented, relying on multiple disconnected models or lacking ingredient-level granularity. NutriSnap addresses this by demonstrating the viability of a unified architecture. Using a single fine-tuned **4M-B** model, our system processes an image through a sequential pipeline: it first predicts a depth map, uses this to inform a semantic segmentation mask, and finally generates a textual caption detailing the mass and nutritional values for each identified ingredient. Our primary result is the successful validation of this integrated methodology. While the model's accuracy is currently limited by data quality, our work confirms the potential of a unified approach and highlights that a large-scale, high-quality dataset is the critical next step for practical application.

## Installation & Usage

### Prerequisites

* Install the conda environment `fourm` ([see instructions](https://github.com/apple/ml-4m?tab=readme-ov-file#installation)) and its dependencies.
* Install `detectron2`.
* For the `build_4m_dataset.py` script, clone the [FoodSAM repo](https://github.com/jamesjg/FoodSAM) into `external_libs/` and install its conda environment. **Note:** It is necessary to install PyTorch 1.8.1, CUDA 11.1, and gcc. It will not work with later versions of CUDA. Also install `Owlready2`, `inflect`, and `pandas`.

### Download the Dataset

To download the raw Nutrition5k dataset, run the following command in PowerShell (with administrator rights):

```bash
gsutil -m rsync -r -x "imagery/side_angles/.*" gs://nutrition5k_dataset/nutrition5k_dataset/ "./data/raw/"
```

### Run Generation

To launch the generation program with 2 GPUs, run the following command from the root `/nutri-snap` directory:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 external_libs/ml_4m/run_generation.py \
--config external_libs/ml_4m/cfgs/default/generation/models/nutrisnap_generation_model.yaml \
--data_config external_libs/ml_4m/cfgs/default/generation/data/nutrisnap_generation_data.yaml \
--gen_config external_libs/ml_4m/cfgs/default/generation/settings_base/nutrisnap_generation_settings.yaml \
--output_dir ./nutrisnap_output --name nutrisnap_run_1
```

This will process the first 10 images from the input folder and save the output in the `nutri-snap/nutrisnap_output/nutrisnap_run_1` directory.

### Configuration

To configure the input directory and the number of images to process, modify the following file:
`external_libs/ml_4m/cfgs/default/generation/data/nutrisnap_generation_data.yaml`

* `data_path`: Change the input path (defaults to `/work/com-304/nutri-snap/data/processed/test/`).
* `num_samples`: Specify the number of images to process.

## Project Architecture

Here is the project architecture. The files detailed here are those created, modified, or used by our project. The `data` and `external_libs` folders are not complete in the Git repository for space reasons, but the full project is located on the cluster at `/work/com-304/nutri-snap`.

```text
nutri-snap/
│
├── data/
│   ├── category_id_files/
│   │   ├── foodsam_to_n5k_mapping.json
│   │   ├── foodseg103_category_id.txt
│   │   └── nutrition5k_category.txt
│   ├── raw/
│   │   └── nutrition5k_dataset/
│   └── processed/
│       ├── test/
│       └── train/
│
├── external_libs/
│   ├── FoodSAM/
│   └── ml-4m/
│       ├── run_generation.py
│       ├── run_training_4m.py
│       ├── run_training_vqvae.py
│       ├── save_vq_tokens.py
│       ├── cfgs/
│       │   └── default/
│       │       ├── 4m/
│       │       │   ├── alphas_mixture/specialized/nutrisnap_pipeline_alphas_eval.yaml
│       │       │   ├── data/nutrition5k/nutrisnap_data_config.yaml
│       │       │   └── models/4m-b_mod7_500b--nutri_snap.yaml
│       │       ├── generation/
│       │       │   ├── data/nutrisnap_generation_data.yaml
│       │       │   ├── models/nutrisnap_generation_model.yaml
│       │       │   └── settings_base/nutrisnap_generation_settings.yaml
│       │       └── tokenization/vqvae/semseg_n5k/
│       │           ├── ViTB-ViTB_4k_448.yaml
│       │           └── ...
│       └── fourm/
│           ├── data/modality_info.py
│           └── utils/plotting_utils.py
│
├── notebooks/
│   ├── bbox_visualizer.ipynb
│   ├── depth_tokenizer_test.ipynb
│   ├── sam_instance_visualizer.ipynb
│   ├── sam_tokenizer_test.ipynb
│   ├── semseg_find_cat.ipynb
│   └── visualize_rgb_reconstruction.ipynb
│
├── src/
│   ├── analysis/
│   │   ├── analyze_processed_dataset_stats.py
│   │   └── validate_and_correct_mapping.py
│   └── data_processing/
│       ├── build_4m_dataset.py
│       ├── caption_generator.py
│       ├── copy_depth_images_conditional.py
│       └── utils/
│           ├── config.py
│           ├── common_utils.py
│           ├── n5k_utils.py
│           ├── foodsam_handler.py
│           └── output_generator.py
│
├── toks/
│   └── text_tokenizer_4m_wordpiece_30k.json
│
├── utils/
│   ├── generation_abstract_functions.py
│   └── semseg_helper_utils.py
│
└── README.md
```

## Methodology

### Model Architecture

Our approach is centered on fine-tuning the pre-trained **4M-B (198M parameters, 7 modalities)** generative model. This model was chosen as a strategic compromise, offering a powerful multimodal foundation while being computationally efficient and mitigating the risk of overfitting on our relatively small, custom-built dataset.

### The Unified Pipeline

The core contribution of this project is a complete, end-to-end pipeline embedded within a single model. The model executes the following sequence of conditional predictions:

1.  **`RGB -> Depth`**: From a single input image, the model first predicts a depth map to understand the volume and 3D layout of the food on the plate.
2.  **`RGB & Depth -> Semantic Segmentation`**: Conditioned on the original image and the generated depth map, the model produces a semantic mask to identify and delineate each ingredient.
3.  **`All -> Caption`**: Finally, using all available information (RGB, depth, and the segmentation mask), the model generates a structured textual caption detailing the estimated weight and nutritional values for each ingredient identified in the previous step.

This sequential, conditional approach ensures that the final nutritional estimates are grounded in the model's own intermediate outputs, creating a coherent and interpretable result.

### Dataset Creation

A major challenge was the lack of a public dataset containing all required modalities (RGB, depth, semantic segmentation, and nutritional metadata). To overcome this, we created a novel dataset by:

1.  **Merging Datasets**: We combined **Nutrition5K** (providing RGB, depth, and nutritional info) and **Food103** (providing segmentation data).
2.  **Generating Missing Modalities**: We used **FoodSAM** (a segmentation model pre-trained on Food103) to generate initial segmentation masks for the Nutrition5K images.
3.  **Semantic Alignment**: A custom script (`alignment_utils.py`) was developed to align the different category labels from the two datasets. This script uses a semantic mapping table created with BERT to associate FoodSAM categories with Nutrition5K ingredients, prioritizing nutritional accuracy when merges were required.
4.  **Filtering for Quality**: We implemented a confidence score based on Jaccard distance and Mean Squared Relative Error to filter the generated data, resulting in a final, high-quality dataset of **~1,300 samples**.
5.  **Custom Tokenization**: As the default segmentation tokenizer was unsuitable, we trained a custom **VQ-VAE** on our new dataset to handle the specific food ingredient categories.

## Results and Findings

The project successfully demonstrated that the **4M-B model could learn the complex, sequential task chain** of our proposed pipeline. The training and evaluation losses converged, indicating that the model did not overfit and was able to understand the structured generative task.

However, the final performance remains modest.

* The **predicted depth maps** were of relatively good quality.
* The **generated segmentation masks** were weak, suffering from artifacts introduced by the noisy and small training data.
* The **generated captions** adhered to the desired format but failed to correctly align ingredient identifiers with their corresponding masks, and the predicted weights were often inaccurate.

Our primary conclusion is that **the unified, end-to-end approach is viable**, but its practical effectiveness is contingent on data quality.

## Limitations and Future Work

### Limitations

1.  **Dataset Quality**: The primary limitation was our dataset. At only 1,300 samples, it was small, noisy (due to the programmatic generation of masks), and lacked diversity (consisting mostly of simple, Western-style dishes).
2.  **Visual-Only Analysis**: A purely visual model cannot account for hidden, high-impact ingredients like oil, sugar, salt, or spices. It also struggles with complex, mixed dishes like stews or soups.
3.  **Tokenizer Performance**: The noise in the dataset significantly impacted the training of the VQ-VAE tokenizer for segmentation, leading to suboptimal performance.

### Future Work

* **Build a Large-Scale Dataset**: The highest priority is the creation of a large, diverse, and clean dataset with ground-truth annotations for all required modalities.
* **Improve Tokenizers**: Training dedicated tokenizers for RGB and depth specifically on a food dataset could enhance performance.
* **Enforce Generation Consistency**: A mechanism could be introduced to constrain the caption decoder, forcing it to only use ingredient identifiers that were actually detected in the semantic segmentation mask from the previous step.

## References

[1] R. Bachmann, O. F. Kar, D. Mizrahi, A. Garjani, M. Gao, D. Griffiths, J. Hu, A. Dehghan, and A. Zamir, "4m-21: An any-to-any vision model for tens of tasks and modalities," 2024. [Online]. Available: <https://arxiv.org/abs/2406.09406>

[2] Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, "Nutrition5k: Towards automatic nutritional understanding of generic food," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8903-8911. [Online]. Available: <https://github.com/google-research-datasets/Nutrition5k>

[3] X. Wu, X. Fu, Y. Liu, E.-P. Lim, S. C. H. Hoi, and Q. Sun, "A large-scale benchmark for food image segmentation," in *Proceedings of the ACM International Conference on Multimedia*, 2021. [Online]. Available: <https://xiongweiwu.github.io/foodseg103.html>

[4] X. Lan, J. Lyu, H. Jiang, K. Dong, Z. Niu, Y. Zhang, and J. Xue, "Foodsam: Any food segmentation," 2023. [Online]. Available: <https://github.com/jamesjg/FoodSAM>

[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," 2019. [Online]. Available: <https://arxiv.org/abs/1810.04805>

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit <http://creativecommons.org/licenses/by-nc/4.0/>
