

![Eagle](https://github.com/adithya-s-k/eagle/assets/27956426/77351cfa-dfed-41ae-a481-6d761cd53e8d)
<div align="center">
A framework streamlining Training, Finetuning, Evaluation and Deployment of Multi Modal Language models
</div>


### Features

- **Diverse Model Support**: Llama3, Phi, Mistral, Gemma, and more.
- **Versatile Image Encoding**: CLIP, Seglip, RADIO, and others.
- **Customization Made Simple**: YAML files and CLI for adaptability.
- **Efficient Resource Utilization**: Seamless operation on a single GPU.
- **Seamless Deployment**: Docker locally or on cloud with Skypilot.
- **Comprehensive Documentation**: Includes datasets for successful implementation.

### Table of Content

1. [Introduction](#introduction)
2. [Supported_Models](#supported-models)
3. [Changelog](#changelog)
4. [Installation](#installation)
5. [Pretrain](#pretrain)
6. [Finetune](#finetune)
7. [Evaluate](#evaluate)
8. [Inference](#inference-deploy)
9. [Features to be Added](#features-to-be-added)
10. [Citation](#citation)
11. [Acknowledgement](#acknowledgement)
---

### SUPPORTED MODELS

### LLMS
- Llama3
- Phi
- Mistral
- Gemma

### Vision Encoder/Transformer

### Audio Encoder/Transformer

### Video Encode/Transformer

### Multi Model 

### CHANGLE LOGS (What's New)

- Version 1.0.1:
  - Added support for distributed training.
  - Included accelerate library.
- Version 1.0.0:
  - Initial release.

### Installation

1. Clone the repository from [GitHub](https://github.com/adithya-s-k/eagle).
2. Install dependencies using pip: `pip install -r requirements.txt`.
3. Run `setup.sh` to set up the environment.
4. Start using Eagle!

### PRETRAIN

- Utilize supported models for pretraining multimodal models.

### FINETUNE

- Fine-tune pretrained models with custom datasets or tasks.

### EVALUATE

- Evaluate model performance using specified metrics and datasets.

### INFERENCE/DEPLOY

- Deploy models for inference on new data or integrate them into existing systems.

### Features to be Added

- Add support for accelerate.
- Add support for additional Huggingface models such as falcon, mpt.

### CITATION

```
@article{AdithyaSKolavi2024,
  title={Eagle: Unified Platform to train multimodal models},
  author={Adithya S Kolavi},
  year={2024},
  url={https://github.com/adithya-s-k/eagle}
}
```

### ACKNOWLEDGEMENT

We would like to express our gratitude to the creators of LLaVA (Large Language and Vision Assistant) for providing the groundwork for our project. Visit their repository [here](https://github.com/haotian-liu/LLaVA).

