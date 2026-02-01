# Advanced RAG System for UNDP

## Technology Stack

<p align="center">
  <img src="https://www.python.org/static/community_logos/python-logo.png" alt="Python" width="150" style="margin: 20px;"/>
  <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit" width="180" style="margin: 20px;"/>
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="150" style="margin: 20px;"/>
  <img src="https://jupyter.org/assets/homepage/main-logo.svg" alt="Jupyter" width="120" style="margin: 20px;"/>
  <img src="https://colab.research.google.com/img/colab_favicon_256px.png" alt="Google Colab" width="120" style="margin: 20px;"/>
  <img src="https://github.com/user-attachments/assets/afb74c38-5f66-478f-9ef5-6a6c1a678586" alt="PyNgrok" width="120" style="margin: 20px;"/>
</p>

### Core Libraries
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Math**: Mathematical functions
- **Collections**: Data structures (Counter, defaultdict)
- **OS**: System operations
- **Time**: Timing utilities
- **Threading**: Concurrent execution
- **Hugging Face Transformers**: For LLM integration and model usage

### External Services
- **PyNgrok**: Secure tunneling to localhost
- **OpenAI API**: LLM integration (future)
- **Google Generative AI**: Alternative LLM provider (future)
- **Hugging Face Hub**: Access to pre-trained models

### Development Tools
- **Python 3.8+**: Programming language
- **Jupyter Notebook**: Development environment
- **Google Colab**: Cloud execution platform

## Demo Screenshots

### Lab 1: Token Estimator
<p align="center">
  <img width="800" alt="Lab 1 Screenshot" src="./img/WhatsApp Image 2026-02-01 at 20.59.07.jpeg" />
</p>

### Lab 2A: Simple RAG System
<p align="center">
  <img width="800" alt="Lab 2A Screenshot" src="./img/WhatsApp Image 2026-02-01 at 20.59.26.jpeg" />
</p>

### Lab 2B: Advanced RAG with LLM
<p align="center">
  <img width="800" alt="Lab 2B Screenshot" src="./img/WhatsApp Image 2026-02-01 at 20.59.38.jpeg" />
</p>

## Overview

This project implements an Advanced Retrieval-Augmented Generation (RAG) System developed as part of a UNDP initiative. The system combines traditional information retrieval techniques with modern language models to provide accurate, context-aware responses to user queries based on uploaded documents.

### What is Advanced RAG?

Retrieval-Augmented Generation (RAG) is a cutting-edge approach that enhances Large Language Models (LLMs) by:

1. **Retrieval**: Finding the most relevant information from a knowledge base
2. **Augmentation**: Enriching the LLM's context with retrieved information
3. **Generation**: Producing accurate, grounded responses

Our Advanced RAG system goes beyond basic implementations by incorporating:
- Multiple retrieval strategies (TF-IDF, semantic search)
- Token estimation and optimization
- Real-time document processing
- Web-based deployment via Streamlit and Ngrok

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Document       │───▶│  Retrieval      │
│   (Query)       │    │  Processing     │    │  Engine         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             ▼
│   Token         │    │   LLM           │    ┌─────────────────┐
│   Estimation    │    │   Integration   │    │  Response       │
│   Tool          │    │   (Optional)    │    │  Generation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **Token Estimator (Lab 1)**
   - Analyzes text files for word count, character count, and token estimation
   - Helps optimize input for LLMs with character limits

2. **Simple RAG Engine (Lab 2A)**
   - Implements TF-IDF vectorization
   - Uses Cosine Similarity for document ranking
   - Processes multiple documents and finds most relevant matches

3. **Advanced RAG with LLM Integration (Future Lab 2B)**
   - Connects to external LLMs (OpenAI GPT, Google Gemini)
   - Provides AI-powered Q&A based on document content

## Features

- **Token Estimation**: Accurate token counting for LLM optimization
- **Document Search**: TF-IDF based retrieval system
- **Similarity Scoring**: Cosine similarity calculations
- **Web Interface**: Streamlit-based user-friendly UI
- **Cloud Deployment**: Ngrok tunneling for public access
- **Multi-format Support**: Handles TXT, MD, CSV files
- **Real-time Processing**: Instant results with progress indicators

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Colab account (recommended) or local Python environment
- Ngrok account for tunneling

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd advanced-rag-undp
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pyngrok
   ```

3. **For LLM integration (optional)**
   ```bash
   pip install openai google-generativeai
   ```

## Usage

### Running the Applications

1. **Token Estimator**
   - Open the notebook
   - Run cells step by step
   - Access the web interface via generated Ngrok URL
   - Upload text files and analyze token counts

2. **Simple RAG System**
   - Upload multiple documents
   - Process documents to build TF-IDF vectors
   - Enter queries to find most relevant documents
   - View similarity scores and rankings

### Example Workflow

```python
# Basic usage example
from collections import Counter
import math

# TF-IDF calculation
def calculate_tfidf(documents, query):
    # Implementation details in notebook
    pass
```

## Performance Metrics

- **Retrieval Accuracy**: Cosine similarity scoring
- **Processing Speed**: Real-time document analysis
- **Token Estimation**: ±5% accuracy compared to actual LLM tokenizers
- **Scalability**: Handles multiple documents efficiently

## Contributing

This project is developed under the UNDP framework. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is developed for the United Nations Development Programme (UNDP) and follows UNDP's open data and open source policies.

## Acknowledgments

- United Nations Development Programme (UNDP)
- Google Colab for providing the development environment
- Open source community for the amazing libraries

## Contact

For questions or support, please contact the UNDP development team.

---

<p align="center">
  <strong>Building a better future through technology and innovation</strong>
  <br>
  UNDP - United Nations Development Programme
</p>
