# 📚 Document Explainer - Advanced AI-Po## 🚀 Quick Start

### Prerequisites

- **Python 3.8+---

## 🔧 How the System Works

### Document Processing Pipeline

1. **Document Scanning**: Recursively scans the `docs/` folder for PDF and HTML files
2. **Text Extraction**:
   - **PDFs**: Uses PyMuPDF (fitz) to extract clean text from all pages
   - **HTML**: Uses BeautifulSoup to parse and extract readable content
3. **Intelligent Chunking**: Splits text into 2000-token chunks with 200-token overlap using OpenAI's tiktoken
4. **Embedding Generation**: Creates vector embeddings using `text-embedding-3-small`
5. **Persistence**: Saves chunks (`parsed_chunks.json`) and embeddings (`embeddings.npy`) for future use

### Query Processing & AI Interaction

1. **Initial Semantic Search**: User query is embedded and compared against document embeddings using cosine similarity
2. **Context Assembly**: Top 10 most relevant chunks are selected and provided to the AI
3. **AI Analysis**: GPT-4o-mini analyzes the context and generates responses
4. **Tool-Enhanced Iteration**: AI can autonomously use tools to:
   - Perform additional semantic searches with different keywords
   - Request more document chunks (up to 50)
   - Record unanswered questions or suggestions
5. **Response Generation**: Final answer is provided with clear source attribution

### Memory & Session Management

- **Conversation Persistence**: All interactions saved to `history.json`
- **State Management**: Chunk limits reset after each query to optimize performance
- **Caching**: Document embeddings persist across sessions to reduce API costsn 3.12.8 recommended)
- **Conda** (recommended for environment management)
- **OpenAI API key** with access to GPT modelsDocument Q&A System

A sophisticated document analysis system powered by OpenAI's latest GPT models with advanced agentic AI capabilities. Features intelligent semantic search, iterative query refinement, conversation memory, and a comprehensive suite of AI tools for enhanced document exploration.

---

## ✨ Key Features

### 🔍 **Advanced Document Processing**

- **Multi-format Support**: Automatically processes PDF and HTML documents from the `docs/` folder
- **Intelligent Chunking**: Splits documents into optimal 2000-token chunks with 200-token overlap for better context preservation
- **Semantic Search**: Uses OpenAI's `text-embedding-3-small` model for high-quality vector embeddings
- **Persistent Caching**: Saves document chunks and embeddings locally (`parsed_chunks.json` and `embeddings.npy`) to minimize API costs

### 🤖 **Agentic AI Integration**

- **Chain Semantic Search**: AI can perform iterative searches on documents with custom queries for deeper exploration
- **Dynamic Tool Usage**: Four specialized tools that the AI can use autonomously during conversations
- **Conversation Memory**: Complete chat history persistence across sessions
- **Context-Aware Responses**: Transparent about whether answers come from documents or general knowledge

### 🛠️ **Intelligent Tool Suite**

The AI assistant has access to four powerful tools:

1. **`request_semantic_search`**: Performs additional semantic searches with custom keywords for deeper document exploration
2. **`request_more_info`**: Dynamically increases the number of relevant chunks (from 10 to max 50) when more context is needed
3. **`record_unknown_question`**: Logs questions that couldn't be answered for system improvement
4. **`record_suggestion`**: Captures user feedback and improvement suggestions

### 🧠 **Smart Search Strategy**

- **Initial Search**: Every user query triggers semantic search for the top 10 most relevant chunks
- **Iterative Refinement**: AI can perform additional searches with different keywords for comprehensive coverage
- **Context Expansion**: Automatically requests more chunks when initial results are insufficient
- **Reset Mechanism**: Chunk limit resets to default (10) after each conversation round for optimal performance

---

## � Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### 1. Environment Setup (Recommended: Conda)

```bash
# Clone or download the project
cd document-explainer

# Create and activate conda environment
conda env create -f environment.yml
conda activate document-explainer

# Alternative: Manual conda setup
conda create -n document-explainer python=3.12
conda activate document-explainer
pip install openai python-dotenv PyMuPDF beautifulsoup4 numpy tiktoken
```

### 2. Configure OpenAI API

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Add Your Documents

Place your documents in the `docs/` folder:

```
docs/
├── research-paper.pdf
├── technical-manual.pdf
├── documentation.html
└── ... (any PDF or HTML files)
```

### 4. Run the System

```bash
python document_explainer.py
```

**Note**: On first run, the system will process all documents and generate embeddings, which may take a few minutes depending on document size and quantity.

---

## � How It Works

1. **Document Processing**: On first run, the system scans `./docs`, chunks documents, and generates embeddings
2. **Semantic Search**: For each question, finds the most relevant document chunks
3. **AI Analysis**: GPT analyzes the chunks and provides contextual answers
4. **Tool Integration**: AI can use built-in tools to enhance responses or gather feedback
5. **Memory**: Conversations are saved and restored between sessions

---

## ⚙️ Configuration

The system uses these default settings (customizable in `document_explainer.py`):

- **AI Model**: `gpt-4o-mini` (latest OpenAI model for optimal performance)
- **Embedding Model**: `text-embedding-3-small` (high-quality, cost-effective embeddings)
- **Chunk Size**: 2000 tokens with 200-token overlap for context preservation
- **Initial Search**: Top 10 chunks (dynamically expandable to 50 via AI tools)
- **Document Folder**: `./docs` (scans recursively)
- **Supported Formats**: PDF (.pdf) and HTML (.html, .htm)
- **Token Counting**: Uses OpenAI's tiktoken for accurate tokenization

### Advanced Settings

```python
# In document_explainer.py, modify these variables:
self.LLM_MODEL = "gpt-4o-mini"          # AI model
self.LLM_EMBEDDING_MODEL = "text-embedding-3-small"  # Embedding model
self.DOC_DIR_PATH = "./docs"            # Document directory
self.TOP_N_DEFAULT = 10                 # Initial chunk count
self.TOP_N_MAX = 50                     # Maximum chunk count
self.MAX_TOKENS = 2000                  # Chunk size
self.OVERLAP = 200                      # Chunk overlap
```

---

## 📝 Usage Examples & AI Capabilities

### Basic Document Q&A

```
You: What are the main methodologies discussed in the power systems papers?
Bot: Based on my analysis of the documents, the main methodologies focus on robust state estimation techniques including maximum likelihood estimation, t-distribution noise models, and LAV (Least Absolute Value) approaches for power system state estimation...
```

### Chain Semantic Search in Action

```
You: Tell me about PMU placement optimization
Bot: [Uses request_semantic_search tool with "PMU placement optimization"]
     Based on the search results, I found several approaches to PMU placement optimization...
     [May use request_semantic_search again with "optimal PMU placement algorithms"]
     Let me search for more specific algorithmic details...
```

### Dynamic Context Expansion

```
You: I need comprehensive details about the mathematical formulations
Bot: [Uses request_more_info tool to increase chunks from 10 to 15, then 20...]
     With additional context, I can provide more detailed mathematical formulations...
```

### Intelligent Question Handling

```
You: What about quantum computing applications?
Bot: I couldn't find information about quantum computing applications in the provided documents.
     [Uses record_unknown_question tool]
     This question has been recorded for potential system enhancement.
```

### Multi-Tool Orchestration

The AI can seamlessly combine multiple tools in a single response:

1. Perform initial semantic search
2. Request additional chunks for more context
3. Perform secondary semantic search with refined keywords
4. Record suggestions for system improvement

---

## �️ File Structure

After running the system, your project will look like this:

```
document-explainer/
├── .env                      # OpenAI API key configuration (create this)
├── .gitignore               # Git exclusions
├── environment.yml          # Conda environment specification
├── document_explainer.py    # Main application with agentic AI
├── README.md               # Documentation (this file)
├── LICENSE                 # MIT License
├── docs/                   # Document repository
│   ├── research-paper1.pdf
│   ├── technical-doc.html
│   └── ... (your documents)
└── Generated Files (auto-created):
    ├── history.json            # Persistent conversation history
    ├── parsed_chunks.json      # Processed document chunks
    ├── embeddings.npy         # Vector embeddings cache
    ├── suggestions.json       # User improvement suggestions  
    └── unknown_questions.json # Questions requiring attention
```

### Key Files Explained

- **`environment.yml`**: Conda environment with all dependencies
- **`parsed_chunks.json`**: Text chunks from all processed documents
- **`embeddings.npy`**: Precomputed vector embeddings (saves API costs)
- **`history.json`**: Complete conversation history across sessions
- **`suggestions.json`**: AI-collected user feedback for system improvements
- **`unknown_questions.json`**: Questions the system couldn't answer

---

## 🔧 Troubleshooting

**"No OpenAI API key found"**

- Ensure your `.env` file exists in the project root
- Verify it contains `OPENAI_API_KEY=your-actual-api-key`
- Check that your API key has access to GPT-4o models

**"No documents found"**

- Place PDF or HTML files in the `docs/` folder
- Ensure the folder exists and contains supported file types (.pdf, .html, .htm)
- Check file permissions and ensure files aren't corrupted

**"Model not found" or API errors**

- The system uses `gpt-4o-mini` by default - ensure your API key has access
- If needed, change `self.LLM_MODEL` to `gpt-4o-mini` or `gpt-4-turbo`
- Verify your OpenAI account has sufficient credits

**High API costs**

- Document embeddings are cached locally after first processing
- Only new documents trigger embedding generation
- Consider reducing `TOP_N_MAX` if using too many chunks

**Conda environment issues**

- Use `conda env create -f environment.yml` to create the environment
- Activate with `conda activate document-explainer`
- If using different Python versions, update `environment.yml` accordingly

**Tool calling errors**

- Ensure you're using a compatible OpenAI model that supports function calling
- GPT-4o-mini and GPT-4o are recommended for best tool usage

---

## 🤝 Contributing

This project demonstrates advanced concepts in RAG (Retrieval-Augmented Generation) and agentic AI systems. Feel free to contribute:

- **Document Format Support**: Add support for DOCX, TXT, Markdown, or other formats
- **Advanced Chunking**: Implement semantic-aware chunking strategies
- **New AI Tools**: Create additional tools for document analysis and interaction
- **Performance Optimization**: Improve embedding storage, search algorithms, or memory usage
- **UI Development**: Build web or desktop interfaces for the system
- **Multi-language Support**: Add support for non-English documents

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

### Core Technologies

- **[OpenAI API](https://openai.com/api/)** - GPT-4o models and text embeddings
- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)** - High-performance PDF text extraction
- **[BeautifulSoup](https://beautiful-soup-4.readthedocs.io/)** - HTML parsing and content extraction
- **[tiktoken](https://github.com/openai/tiktoken)** - Accurate tokenization for OpenAI models
- **[NumPy](https://numpy.org/)** - Vector operations and embedding storage

### Inspiration

This project is inspired by Ed Donner's excellent work on agentic AI systems. Check out his [Complete Agentic AI Engineering Course](https://github.com/ed-donner/agents) for more advanced AI agent patterns and implementations.

### Documentation

This README document was written with the assistance of AI to ensure comprehensive coverage of the system's features and capabilities.
