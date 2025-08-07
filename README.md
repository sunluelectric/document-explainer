# 📚 Document Explainer - AI-Powered Document Q&A System

A sophisticated document analysis system that uses OpenAI's GPT models to answer questions about your documents. Features semantic search, intelligent chunking, conversation history, and smart tool integration for enhanced user experience.

---

## ✨ Key Features

### 🔍 **Smart Document Processing**

- **Multi-format Support**: Automatically processes PDF and HTML documents
- **Intelligent Chunking**: Splits documents into optimal chunks with configurable overlap
- **Semantic Search**: Uses OpenAI embeddings to find the most relevant content
- **Caching System**: Saves embeddings locally to reduce API costs and improve speed

### 🤖 **Advanced AI Integration**

- **Tool-Enhanced Responses**: AI can request more context, record unknown questions, and capture user suggestions
- **Conversation Memory**: Maintains chat history across sessions
- **Context-Aware**: Provides answers based on document content with transparency about knowledge sources

### 🛠️ **Built-in Tools**

- **Unknown Question Tracking**: Records questions that couldn't be answered
- **Suggestion Collection**: Captures user feedback for system improvements
- **Dynamic Context Expansion**: Can request more document chunks when needed
- **Privacy Protection**: Keeps sensitive documents local and secure

---

## � Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### 1. Setup Environment

```bash
# Clone or download the project
cd document-explainer

# Install dependencies
pip install openai python-dotenv PyMuPDF beautifulsoup4 numpy tiktoken
```

### 2. Configure API Access

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Add Your Documents

Place your documents in the `./docs` folder:

```
docs/
├── your-document.pdf
├── research-paper.html
├── research-paper.pdf
└── ... (any PDF or HTML files)
```

### 4. Run the System

```bash
python document_explainer.py
```

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

- **Model**: `gpt-4o-mini` (configurable)
- **Embedding Model**: `text-embedding-3-small`
- **Chunk Size**: 2000 tokens with 200 token overlap
- **Search Results**: Top 10 chunks (expandable to 50)
- **Document Folder**: `./docs`

---

## 📝 Usage Examples

**Basic Question:**

```
You: What is the main topic of the research papers?
Bot: Based on the documents, the main focus appears to be power system state estimation...
```

**Request More Context:**

```
You: Can you provide more details about the methodology?
Bot: [AI automatically requests more document chunks if needed]
```

**System Feedback:**
The AI automatically records unknown questions and suggestions to help improve the system.

---

## �️ File Structure

After running the system, your project will look like this:

```
document-explainer/
├── .env                    # Your OpenAI API key (create this)
├── .gitignore             # Excludes sensitive files from git
├── document_explainer.py  # Main CLI application
├── app.py                 # Web interface (Gradio)
├── chatbot.py            # Legacy chatbot (if present)
├── README.md             # This file
├── docs/                 # Place your documents here
│   ├── document1.pdf
│   ├── research.html
│   └── research2.pdf
└── Generated Files (auto-created):
    ├── history.json          # Conversation history
    ├── parsed_chunks.json    # Document chunks
    ├── embeddings.npy       # Vector embeddings
    ├── suggestions.json     # User suggestions
    └── unknown_questions.json # Unanswered questions
```

---

## 🔧 Troubleshooting

**"No OpenAI API key found"**

- Ensure your `.env` file contains `OPENAI_API_KEY=your-key-here`

**"No documents found"**

- Place PDF or HTML files in the `./docs` folder
- Check that the folder exists and contains supported file types

**"Out of tokens" or high API costs**

- Embeddings are cached locally after first run
- Consider reducing `MAX_TOKENS` or `TOP_N` in the code

**Tool calling errors**

- Ensure you're using a compatible OpenAI model (gpt-4o-mini or gpt-4o recommended)

---

## 🤝 Contributing

This project demonstrates key concepts in RAG (Retrieval-Augmented Generation) and agentic AI. Feel free to:

- Add support for more document formats
- Improve the chunking strategy
- Add new AI tools and capabilities
- Optimize embedding and search performance

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

Built with:

- [OpenAI API](https://openai.com/api/) for language models and embeddings
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/) for HTML parsing
- [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/) for HTML parsing
- Ed Donner. This project is inspired by his Udemy courses and Github project [GitHub - ed-donner/agents: Repo for the Complete Agentic AI Engineering Course](https://github.com/ed-donner/agents). Why does he look so young on the profile picture.
