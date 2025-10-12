# Attention Mechanism & Transformer Implementation

A comprehensive implementation and analysis of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017). This project includes a complete PyTorch implementation of the Transformer model with detailed examples, training scripts, and educational demonstrations.

## 🔗 Interactive Notebooks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1shyaOwqaOr6l018CCMnJa4xr7aRJ6tZO?usp=sharing) **Transformer Implementation & Training**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L88g0X-T6-Jc0rJKkm0XoTekfzjrdx0H?usp=sharing) **Attention Mechanism Analysis**

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Training](#training)
- [Documentation](#documentation)
- [Requirements](#requirements)
- [Contributing](#contributing)

## ✨ Features

### Core Components
- **Multi-Head Attention**: Complete implementation with visualization capabilities
- **Positional Encoding**: Sinusoidal position embeddings for sequence modeling
- **Transformer Encoder/Decoder**: Full encoder-decoder architecture
- **Layer Normalization**: Pre-norm architecture for stable training
- **Feed-Forward Networks**: Position-wise fully connected layers

### Training & Evaluation
- **Multiple Datasets**: Support for dummy data and WMT14 translation dataset
- **Training Scripts**: Comprehensive training pipeline with loss tracking
- **Model Evaluation**: BLEU score calculation and performance metrics
- **Visualization**: Attention heatmaps and training curves

### Educational Features
- **Interactive Examples**: Step-by-step demonstrations of attention mechanisms
- **Attention Visualization**: Heatmaps showing what the model attends to
- **Component Testing**: Individual tests for each Transformer component

## 📁 Project Structure

```
Attention-Mechanism/
├── code/
│   ├── transformer.py          # Core Transformer implementation
│   ├── train.py               # Main training script
│   ├── examples.py            # Educational examples and demonstrations
│   ├── wmt14_dataset.py       # WMT14 dataset handling
│   ├── train_wmt14.py         # WMT14-specific training
│   ├── test_transformer.py    # Unit tests for Transformer components
│   ├── test_trained_model.py  # Evaluation scripts for trained models
│   └── test_wmt14.py          # WMT14 dataset tests
├── doc/
│   ├── requirements.txt       # Python dependencies
│   ├── training_curves.png    # Training visualization
│   └── RaghavMishra_TransformerAnalysis.pdf  # Detailed analysis report
└── README.md                  # This file
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Raghav0079/Attention-Mechanism.git
   cd Attention-Mechanism
   ```

2. **Install dependencies:**
   ```bash
   pip install -r doc/requirements.txt
   ```

3. **Verify installation:**
   ```bash
   cd code
   python test_transformer.py
   ```

## 💻 Usage

### Quick Start

```python
from code.transformer import Transformer
import torch

# Initialize Transformer
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=100
)

# Example forward pass
src = torch.randint(1, 10000, (32, 50))  # batch_size=32, seq_len=50
tgt = torch.randint(1, 10000, (32, 40))  # batch_size=32, seq_len=40

output = model(src, tgt)
print(f"Output shape: {output.shape}")  # [32, 40, 10000]
```

### Training a Model

```bash
cd code
python train.py --epochs 100 --batch_size 64 --lr 0.0001
```

### Running Examples

```bash
cd code
python examples.py
```

This will run interactive demonstrations showing:
- How attention mechanisms work
- Positional encoding visualization
- Multi-head attention patterns
- Complete translation examples

## 📊 Examples

The `examples.py` script provides several educational demonstrations:

### 1. Attention Mechanism Visualization
```python
python examples.py
# Shows attention weights for sample sequences
# Demonstrates how the model focuses on different parts of the input
```

### 2. Positional Encoding
```python
# Visualizes sinusoidal positional embeddings
# Shows how position information is encoded
```

### 3. Translation Demo
```python
# Complete translation example from scratch
# Shows encoder-decoder interaction
```

## 🏋️ Training

### Basic Training
```bash
python train.py
```

### WMT14 Dataset Training
```bash
python train_wmt14.py --data_path /path/to/wmt14
```

### Training Parameters
- **Learning Rate**: 0.0001 (with warmup)
- **Batch Size**: 64
- **Model Dimensions**: 512
- **Attention Heads**: 8
- **Layers**: 6
- **Dropout**: 0.1

### Monitoring Training
Training curves and metrics are automatically saved and can be visualized:
- Loss curves
- BLEU scores
- Attention visualizations

## 📚 Documentation

- **`doc/RaghavMishra_TransformerAnalysis.pdf`**: Comprehensive analysis of the Transformer architecture
- **Inline Documentation**: All code is thoroughly documented with docstrings
- **Examples**: Interactive examples with explanations

## 📋 Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **NumPy**: 1.21.0+
- **Matplotlib**: 3.5.0+
- **Additional dependencies**: See `doc/requirements.txt`

### Hardware Requirements
- **GPU**: Recommended for training (CUDA-capable)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for datasets

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
cd code
python test_transformer.py      # Test Transformer components
python test_trained_model.py    # Test trained model functionality
python test_wmt14.py           # Test dataset handling
```

## 🎯 Key Features Demonstrated

1. **Scaled Dot-Product Attention**: Core attention mechanism with mathematical foundation
2. **Multi-Head Attention**: Parallel attention heads for different representation subspaces
3. **Positional Encoding**: Sinusoidal embeddings for sequence position information
4. **Layer Normalization**: Pre-norm architecture for training stability
5. **Residual Connections**: Skip connections for gradient flow
6. **Encoder-Decoder Architecture**: Complete seq2seq framework
7. **Masking**: Proper attention masking for autoregressive generation

## 📈 Results

The implementation achieves:
- Convergent training on dummy datasets
- Proper attention pattern formation
- Reasonable BLEU scores on translation tasks
- Educational clarity with visualization capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes. Please cite the original "Attention Is All You Need" paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## 👨‍💻 Author

**Raghav Mishra**
- GitHub: [@Raghav0079](https://github.com/Raghav0079)

## 🙏 Acknowledgments

- Original Transformer paper authors (Vaswani et al.)
- PyTorch team for the excellent framework
- The open-source community for inspiration and resources

---

**Note**: This implementation is designed for educational purposes to understand the Transformer architecture. For production use, consider using optimized libraries like Hugging Face Transformers.