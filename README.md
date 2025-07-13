# RAG-memory-ChatBot

# 🤖 RAG Memory ChatBot 📚

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-orange.svg)](https://pinecone.io)
[![Google AI](https://img.shields.io/badge/Google%20AI-Gemini-red.svg)](https://ai.google.dev)

> 🚀 An intelligent document chatbot with memory capabilities using RAG (Retrieval Augmented Generation) architecture powered by Google's Gemini LLM and Pinecone vector database.

## ✨ Features

- 📖 **PDF Document Processing**: Load and process multiple PDF research papers
- 🧠 **Memory Integration**: Maintains conversation context across interactions
- 🔍 **Semantic Search**: Advanced document retrieval using HuggingFace embeddings
- ⚡ **Real-time Chat**: Interactive conversation interface
- 📊 **Source Citations**: Automatic source referencing with page numbers
- 🎯 **Literature Review Generation**: Automated academic writing assistance
- 🔧 **Modular Architecture**: Clean, maintainable code structure

## 🎯 RAG vs RAG with Memory: Key Differences

| Feature | Traditional RAG 🔄 | RAG with Memory 🧠 |
|---------|---------------------|-------------------|
| **Context Awareness** | Each query is independent | Remembers previous conversations and maintains context |
| **Conversation Flow** | No conversation history | Seamless multi-turn conversations with reference to past exchanges |
| **Response Quality** | Basic retrieval-based answers | Enhanced responses using conversation history and document context |

## 📦 Required Libraries & Installation

### 🛠️ Core Dependencies

The project uses the following libraries:

```python
# LangChain ecosystem
langchain>=0.3.26
langchain-community>=0.3.27
langchain_google_genai>=2.1.7
langchain_pinecone>=0.2.8
langchain_huggingface>=0.3.0

# Vector Database
pinecone>=7.3.0

# PDF Processing
pypdf>=5.7.0

# Additional utilities
python-dotenv>=1.1.1
getpass
```

### 📋 Complete Installation Commands

Run these commands in your terminal:

```bash
# Install all required packages
pip install langchain langchain_google_genai
pip install pinecone
pip install -U langchain-community
pip install langchain_pinecone
pip install langchain_huggingface
pip install pypdf
```

Or install everything at once:

```bash
pip install langchain langchain-community langchain_google_genai langchain_pinecone langchain_huggingface pinecone pypdf python-dotenv
```

## 🔑 API Configuration

### 🌟 Google AI (Gemini) API Setup

1. **Visit Google AI Studio**: Go to [Google AI Studio](https://ai.google.dev)
2. **Create Project**: Create a new project or select existing one
3. **Generate API Key**: Navigate to API keys section and create new key
4. **Set Environment Variable**:

```python
import os
import getpass

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
```

### 🌲 Pinecone Vector Database Setup

1. **Sign Up**: Create account at [Pinecone](https://pinecone.io)
2. **Create Project**: Set up a new project in dashboard
3. **Get API Key**: Copy your API key from project settings
4. **Configure Environment**:

```python
import os
import getpass

if "PINECONE_API_KEY" not in os.environ:
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
```

## 🚀 Quick Start Guide

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/abuzar01440/RAG-memory-ChatBot.git
cd RAG-memory-ChatBot
```

### 2️⃣ Install Dependencies

```bash
# Install all required libraries
pip install langchain langchain_google_genai pinecone langchain-community langchain_pinecone langchain_huggingface pypdf
```

### 3️⃣ Prepare Your Documents

```python
# Create papers directory
import os
if not os.path.exists("papers"):
    os.makedirs("papers")
    print("Created 'papers' directory")
    print("Please upload your PDF files to this directory before continuing.")

# Upload your PDF research papers to the /papers/ directory
```

### 4️⃣ Configure and Run

```python
# Import required libraries
import os
import re
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Run the main function
if __name__ == "__main__":
    main()
```

## 💻 Usage Examples

### 🗣️ Basic Conversation

```
You: What are the main contributions of the papers?
Assistant: Based on the research papers, the main contributions include:
1. Novel deep learning architectures for domain generation detection...
2. Weakly supervised learning approaches for limited labeled data...
3. Efficient memory usage techniques for resource-constrained environments...

You: Can you compare the methodologies used?
Assistant: Comparing the methodologies from our previous discussion about the main contributions, 
I can see several distinct approaches...
```

### 📚 Literature Review Generation

```python
# Automatic literature review generation
review_prompt = """
Based on all the research papers you have access to, write a comprehensive literature review paragraph 
in IEEE citation style. The paragraph should:

1. Summarize the main contributions of each paper
2. Highlight the pros and cons of different approaches
3. Identify research gaps and limitations
4. Show the progression of ideas in the field
5. Be written in academic style suitable for a research paper

Please provide a well-structured literature review paragraph that synthesizes 
the findings from all available papers.
"""

result = qa_chain({"question": review_prompt})
print(result['answer'])
```

### 🔍 Source Citation Requests

```
You: What sources support the claims about weakly supervised learning?
Assistant: Here are the relevant sources:

Sources:
1. Weakly_Supervised_Deep_Learning_for_the_Detection_of_Domain_Generation_Algorithms.pdf (Page 5)
   Excerpt: The methodology demonstrates improved performance with limited labeled data...

2. deep fake weak supervise learning - Consensus.pdf (Page 12)
   Excerpt: Results indicate that weakly supervised approaches can achieve...

3. journal.pone.0326565.pdf (Page 8)
   Excerpt: Our single-layer KAN model effectively balances efficiency and performance...
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   📄 PDF Docs   │ => │  🔄 Doc Loader   │ => │ ✂️ Text Splitter│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        ⬇️
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 💬 Chat Interface│ <= │ 🧠 Memory Buffer │ <= │🤖 HF Embeddings │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ⬆️                       ⬆️                       ⬇️
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│🎯 Conversational│ <= │  🌟 Gemini LLM   │ <= │🌲 Pinecone VectorDB│
│     Chain       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ⬆️                       ⬆️
                       ┌──────────────────┐    ┌─────────────────┐
                       │   🔍 Retriever   │ => │  📊 Vector Store │
                       └──────────────────┘    └─────────────────┘
```

## ⚙️ Configuration Options

### 🎛️ Model Parameters

```python
# Gemini LLM Configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",      # Model version
    temperature=0.6,               # Creativity level (0.0-1.0)
    max_output_tokens=1500,        # Maximum response length
    top_k=40,                      # Token selection diversity
    top_p=0.95                     # Nucleus sampling threshold
)

# HuggingFace Embedding Configuration
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Embedding model
    model_kwargs={'device': 'cpu'},                        # Device configuration
    encode_kwargs={'normalize_embeddings': True}           # Normalization
)
```

### 🔧 Vector Store Settings

```python
# Document Retrieval Configuration
retriever = vector_store.as_retriever(
    search_type="mmr",                           # Maximal Marginal Relevance
    search_kwargs={
        "k": 5,                                  # Number of documents to retrieve
        "score_threshold": 0.7,                  # Minimum relevance score
        "filter": {"source": {"$exists": True}} # Source filtering
    }
)

# Memory Configuration
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",     # Key for storing chat history
    k=3,                          # Number of previous exchanges to remember
    return_messages=True,         # Return full message objects
    output_key="answer"          # Key for response output
)
```

### 📄 Document Processing Settings

```python
# Text Splitting Configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,              # Maximum chunk size in characters
    chunk_overlap=150,            # Overlap between chunks
    length_function=len           # Function to measure text length
)

# Pinecone Index Configuration
INDEX_NAME = "rag-pdf-index"      # Index name in Pinecone
EMBEDDING_DIMENSION = 768         # Dimension for all-mpnet-base-v2
METRIC = "cosine"                 # Similarity metric
```

## 📊 Performance Metrics & Benchmarks

### 🎯 System Specifications

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- **Vector Database**: Pinecone Serverless (AWS us-east-1)
- **LLM**: Google Gemini 1.5 Flash
- **Retrieval Strategy**: Maximal Marginal Relevance (MMR)
- **Memory Window**: 3 previous conversation turns
- **Document Chunking**: 1200 characters with 150 character overlap

### 📈 Performance Benchmarks

Based on the test documents:

```
Document Processing:
├── 📄 Total PDFs: 3 research papers
├── 📑 Total Pages: 58 pages processed
├── 🔤 Text Chunks: 195 chunks generated
├── ⏱️ Processing Time: ~20-30 seconds
└── 💾 Vector Storage: Pinecone serverless

Retrieval Performance:
├── 🎯 Retrieval Accuracy: High relevance scoring
├── ⚡ Response Time: 2-5 seconds per query
├── 🔍 Search Type: MMR for diversity
└── 📊 Top-K Results: 5 most relevant chunks
```

## 🛠️ Troubleshooting

### ❌ Common Issues & Solutions

#### 1. **Pinecone Connection Error**
```python
# Error: "No active indexes found in your Pinecone project"
# Solution: Check API key and ensure index is created
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
existing_indexes = pc.list_indexes().names()
print(f"Available indexes: {existing_indexes}")
```

#### 2. **PDF Loading Issues**
```bash
# Error: PDF files not loading properly
# Solution: Ensure PDFs are not password-protected and install latest pypdf
pip install --upgrade pypdf

# Check if PDF files exist
import os
papers = os.listdir("/content/papers")
pdf_files = [f for f in papers if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
```

#### 3. **Memory Deprecation Warnings**
```python
# Warning: LangChainDeprecationWarning about memory migration
# This is expected due to LangChain version updates
# Functionality remains unaffected, warning can be ignored
```

#### 4. **API Key Issues**
```python
# Ensure API keys are properly set
import os
print("Google API Key set:", "GOOGLE_API_KEY" in os.environ)
print("Pinecone API Key set:", "PINECONE_API_KEY" in os.environ)
```

### 🔧 Debug Mode

```python
# Enable verbose output for debugging
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    get_chat_history=lambda h: h,
    verbose=True  # Enable debug output
)
```

## 🎓 Advanced Features

### 📚 Literature Review Generator

```python
def generate_literature_review(qa_chain):
    """Generate a literature review paragraph from loaded papers"""
    
    review_prompt = """
    Based on all the research papers you have access to, write a comprehensive literature review paragraph 
    in IEEE citation style. The paragraph should:
    
    1. Summarize the main contributions of each paper
    2. Highlight the pros and cons of different approaches
    3. Identify research gaps and limitations
    4. Show the progression of ideas in the field
    5. Be written in academic style suitable for a research paper
    
    Please provide a well-structured literature review paragraph that synthesizes 
    the findings from all available papers.
    """
    
    try:
        result = qa_chain({"question": review_prompt})
        return result['answer']
    except Exception as e:
        return f"Error generating literature review: {e}"

# Usage example
literature_review = generate_literature_review(qa_chain)
print(literature_review)
```

### 🔍 Source Verification

```python
# Automatic source citation when keywords are detected
def should_show_sources(query):
    keywords = ["source", "reference", "cite", "paper", "study", "research"]
    return any(word in query.lower() for word in keywords)

# Enhanced query processing with automatic citations
if should_show_sources(user_query):
    # Display sources automatically
    print("\nSources:")
    for i, doc in enumerate(result["source_documents"][:3]):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"{i+1}. {os.path.basename(source)} (Page {page})")
        print(f"   Excerpt: {doc.page_content[:150]}...\n")
```

### 🛣️ Development Roadmap

1. **🍴 Fork the Repository**
   ```bash
   git clone https://github.com/abuzar01440/RAG-memory-ChatBot.git
   ```

2. **🌿 Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **✏️ Make Changes**
   - Follow Python PEP 8 style guidelines
   - Add comprehensive docstrings
   - Include error handling

4. **🧪 Test Your Changes**
   ```bash
   python test_chatbot.py
   ```

5. **📤 Submit Pull Request**
   ```bash
   git commit -m 'Add amazing feature'
   git push origin feature/amazing-feature
   ```

### 🎯 Contribution Guidelines

- 📝 **Code Style**: Follow PEP 8
- 📚 **Documentation**: Update README for new features
- 🐛 **Bug Reports**: Use issue templates
- 💡 **Feature Requests**: Provide detailed specifications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 abuzar01440

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- 🤗 **HuggingFace**: For providing excellent embedding models
- 🌲 **Pinecone**: For scalable vector database services
- 🌟 **Google AI**: For powerful Gemini LLM capabilities
- 🦜 **LangChain**: For the comprehensive RAG framework
- 📚 **Open Source Community**: For continuous inspiration and support

## 📞 Support & Contact

- 📧 **Email**: abuzar01440@example.com

### 📊 Project Stats

```
📈 Project Metrics:
├── 📅 Created: July 13, 2025
├── 👤 Maintainer: abuzar01440
├── 🔧 Language: Python 3.11+
├── 📦 Dependencies: 8 core libraries
├── 🧪 Test Coverage: Expanding
└── 📝 Documentation: Comprehensive
```

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

**🚀 Happy **Chatting** with RAG Memory ChatBot! 🚀**

Made with ❤️ by [abuzar01440](https://github.com/abuzar01440)

*Last Updated: July 13, 2025*

</div>
```
