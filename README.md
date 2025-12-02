# 📰 News Research Tool

> **An intelligent news research assistant powered by Generative AI**

A user-friendly news research tool that leverages LangChain and OpenAI to transform how you interact with news articles. Simply input article URLs, ask questions, and get instant, relevant insights from your sources.

---

## ✨ Features

- 🔗 **Multi-URL Processing** - Process up to 3 news article URLs simultaneously
- 🤖 **AI-Powered Q&A** - Ask questions and get intelligent answers based on article content
- 📊 **Vector Search** - Advanced FAISS-based semantic search for accurate retrieval
- 📝 **Source Citation** - Every answer includes proper source attribution
- 🎨 **Streamlit UI** - Clean, intuitive web interface

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/widushan/News-Research-Tool.git
   cd News-Research-Tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501`

   <img width="1903" height="1036" alt="Image" src="https://github.com/user-attachments/assets/26177802-d4ca-4349-bd84-b75b8c90cf73" />
   
---

## 🎯 How It Works

1. **Input URLs** - Enter up to 3 news article URLs in the sidebar
2. **Process** - Click "Process URLs" to extract and index the content
3. **Ask Questions** - Type your question in the main interface
4. **Get Answers** - Receive AI-generated answers with source citations

---

## 🛠️ Tech Stack

- **LangChain** - Framework for building LLM applications
- **OpenAI** - GPT models for embeddings and text generation
- **FAISS** - Vector similarity search
- **Streamlit** - Web application framework
- **Python** - Core programming language

---

## 📦 Project Structure

```
News-Research-Tool/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── faiss_store_openai/     # Vector store directory (auto-generated)
└── notebooks/              # Jupyter notebooks for experimentation
```

---

## 🔧 Configuration

The application uses the following default settings:

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Retrieval Count**: Top 4 most relevant chunks
- **Temperature**: 0.9 (for creative responses)
- **Max Tokens**: 500

---

## 📝 Usage Example

1. Enter article URLs:
   ```
    https://srilankacricket.lk/
    https://en.wikipedia.org/wiki/Sri_Lanka_national_cricket_team
    https://www.espncricinfo.com/team/sri-lanka-8

   ```

2. Click "Process URLs"

3. Ask a question:
   ```
   Give me details of cricketing world after winning the 1996 Cricket World Cup under the captaincy of Arjuna Ranatunga.
   ```

4. Get your answer with sources!

    <img width="1905" height="1023" alt="Image" src="https://github.com/user-attachments/assets/bb20ffe4-6099-4a9d-981c-bb42ec217429" />

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📖 Improve documentation

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🔗 Links

- **GitHub Repository**: [https://github.com/widushan/News-Research-Tool](https://github.com/widushan/News-Research-Tool)
- **Issues**: [Report a bug or request a feature](https://github.com/widushan/News-Research-Tool/issues)

---

## ⚠️ Notes

- Make sure you have a valid OpenAI API key
- The first run will take longer as it builds the vector index
- Processed articles are stored locally in the `faiss_store_openai/` directory

---

<div align="center">

**Made with ❤️ using LangChain & OpenAI**

⭐ Star this repo if you find it helpful!

</div>
