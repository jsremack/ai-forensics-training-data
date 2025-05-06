# AI Forensics Training Data Examples

This repository contains examples and tools for performing forensic analysis on AI training data as described in the book "AI Forensics: Investigation and Analysis of Artificial Intelligence Systems" by Joseph C. Sremack.

## Overview

The examples in this repository demonstrate practical techniques for analyzing AI training data for forensic purposes, including:

- Source verification and provenance tracing
- Synthetic content detection
- Bias analysis
- Quality assessment
- Lineage tracking
- Documentation analysis

## Structure

- `examples/` - Example notebooks and scripts for different data types
  - `financial_data/` - Financial dataset analysis examples
  - `nlp_data/` - NLP dataset investigation examples
  - `image_data/` - Image data forensic examples
  - `multimodal_data/` - Multi-modal analysis examples
- `datasets/` - Sample datasets for demonstration
  - `synthetic/` - Synthetic datasets with known issues
  - `anonymized/` - Anonymized real-world examples
- `common/` - Shared utility functions and metrics
- `docs/` - Additional documentation and guides

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - torch
  - transformers
  - spacy
  - opencv-python
  - textstat
  - imagehash
  - networkx

### Installation

1. Clone this repository:
git clone https://github.com/jsremack/ai-forensics-training-data.git
cd ai-forensics-training-data

2. Install required packages:
pip install -r requirements.txt

3. Download required models:
python -m spacy download en_core_web_md

### Usage

Each example can be run as a standalone script or through the provided Jupyter notebooks:
Financial data analysis
python examples/financial_data/financial_data_analysis.py
NLP data forensics
python examples/nlp_data/nlp_data_forensics.py
Image data forensics
python examples/image_data/image_data_forensics.py
Multi-modal analysis
python examples/multimodal_data/multimodal_data_forensics.py

## Example Scenarios

1. **Financial Dataset Bias Detection**: Analyze a loan approval dataset for potential demographic biases.
2. **Synthetic Text Identification**: Detect AI-generated content within a text corpus.
3. **Image Dataset Quality Assessment**: Evaluate the quality and consistency of an image collection.
4. **Cross-Modal Consistency Verification**: Verify alignment between image-text pairs in a multimodal dataset.


## License

This project is licensed under the MIT License.
