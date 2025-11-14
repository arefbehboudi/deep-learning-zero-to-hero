# Deep Learning From Scratch — Python 🚀

A practical, code-focused journey to learn **Deep Learning from zero to expert level**.

No long theoretical chapters — every concept is learned by **building models**, writing **real training loops**, and experimenting with **actual datasets**.

This repository covers:
- Core neural network fundamentals  
- Implementing models **from scratch** (NumPy)  
- Building real models with **PyTorch**  
- Computer Vision (CNNs)  
- NLP with RNN/LSTM  
- Transformers (Attention, BERT, GPT)  
- Generative models (GAN, VAE, Diffusion)  
- Model deployment with FastAPI & ONNX  

---

## 📁 Repository Structure
```
deep-learning-zero-to-hero/
│
├── 01-mlp-numpy/
│   ├── 01-perceptron-binary-classification/
│   ├── 02-mlp-from-scratch-xor/
│   └── 03-mlp-from-scratch-toy2d/
│
├── 02-pytorch-basics/
│   ├── 01-linear-regression-pytorch/
│   ├── 02-mlp-mnist/
│   └── 03-custom-training-loop/
│
├── 03-cnn-vision/
│   ├── 01-cnn-mnist/
│   ├── 02-cnn-cifar10/
│   ├── 03-resnet-block-from-scratch/
│   └── 04-transfer-learning-resnet50/
│
├── 04-rnn-lstm-nlp/
│   ├── 01-char-rnn-text-generation/
│   ├── 02-lstm-sentiment-imdb/
│   └── 03-seq2seq-toy-translation/
│
├── 05-transformers/
│   ├── 01-self-attention-from-scratch/
│   ├── 02-mini-transformer-encoder/
│   ├── 03-mini-gpt-from-scratch/
│   ├── 04-bert-finetune-classification/
│   └── 05-gpt2-finetune-custom-text/
│
├── 06-training-tricks/
│   ├── 01-optimizers-comparison/
│   ├── 02-lr-schedulers/
│   ├── 03-mixed-precision-training/
│   └── 04-logging-with-wandb/
│
├── 07-advanced-generative/
│   ├── 01-gan-mnist/
│   ├── 02-dcgan-celeba/
│   ├── 03-vae-from-scratch/
│   └── 04-ddpm-minimal-diffusion/
│
└── 08-deployment/
    ├── 01-fastapi-inference-server/
    ├── 02-dockerize-model-service/
    └── 03-export-onnx-and-benchmark/
```

---

## 🎯 Goal of This Repository

This project is designed for developers, students, and ML enthusiasts who want to:

- Understand how deep learning works **under the hood**
- Build and train actual models step-by-step  
- Explore modern architectures used today in production  
- Learn DL in a clean, practical, and project-driven way  
- Gradually move from beginner → intermediate → advanced  

Every folder contains:
- Clean & well-commented code  
- A clear explanation (`README.md`)  
- Minimal dependencies  
- Reproducible experiments  

---

## 🧠 Requirements

### Python version
Python 3.9+

### Main libraries
numpy
matplotlib
pytorch (torch, torchvision)
fastapi
uvicorn
wandb
onnxruntime


(Each sub-project has its own requirements.txt)

---

## 🚀 How to Use

Clone the repo:

```bash
git clone https://github.com/<your-username>/deep-learning-zero-to-hero
cd deep-learning-zero-to-hero
```
Go into any project folder, install its dependencies, and run the code.

## 🗺 Roadmap (Progress)

     NumPy MLP implementations
     PyTorch basics
     CNNs
     RNN & LSTM
     Transformers
     GAN / VAE / Diffusion
     Deployment
    
The repository is updated continuously.

## 🤝 Contributing

Pull requests, issues, and suggestions are welcome!
Feel free to add new examples, improve code quality, or create documentation.

## 📜 License

MIT License — free to use and modify.

## ⭐ Support

If you find this project useful, consider giving it a star ⭐ on GitHub.

Happy coding and enjoy your deep learning journey! 🚀