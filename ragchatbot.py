"""
RAG Chatbot - Working Version
Simple and Reliable Document Q&A System
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5

# Initialize FastAPI
app = FastAPI(title="RAG Chatbot", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vectorstore = None
qa_chain = None
current_document = None
embeddings = None

# Create upload directory
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)


# Models
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None


# Initialize embeddings once
def initialize_embeddings():
    """Load embeddings model once at startup"""
    global embeddings
    if embeddings is None:
        print("🔄 Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings loaded!")


def load_document(file_path: str):
    """Load document based on file type"""
    extension = Path(file_path).suffix.lower()

    if extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif extension in ['.docx', '.doc']:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return loader.load()


def process_document(file_path: str):
    """Process document and create QA chain"""
    global vectorstore, qa_chain, embeddings

    print(f"\n{'=' * 60}")
    print(f"🔄 Processing document...")
    print(f"{'=' * 60}")

    # Load document
    print("📄 Loading document...")
    documents = load_document(file_path)
    print(f"✅ Loaded {len(documents)} pages")

    # Split into chunks
    print("🔪 Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ Created {len(splits)} chunks")

    # Create vector store
    print("🧠 Creating vector store...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("✅ Vector store created!")

    # Create LLM
    print("🤖 Setting up AI model...")
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Create QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    print("✅ AI ready for questions!")
    print(f"{'=' * 60}\n")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Main interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Parser</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        #uploadSection, #chatSection {
            padding: 40px;
        }

        #chatSection {
            display: none;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
        }

        .upload-area:hover {
            background: #eef1ff;
            border-color: #764ba2;
        }

        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }

        .file-input {
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 20px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            display: none;
            text-align: center;
        }

        .status.show {
            display: block;
        }

        .status.info {
            background: #d1ecf1;
            color: #0c5460;
        }

        .status.success {
            background: #d4edda;
            color: #155724;
        }

        .status.error {
            background: #f8d7da;
            color: #721c24;
        }

        .chat-box {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 15px;
            margin-bottom: 20px;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 20px;
            line-height: 1.8;
            white-space: pre-wrap;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .message.bot .message-content {
            background: white;
            border: 1px solid #e0e0e0;
            color: #333;
        }

        .input-area {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #667eea;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
        }

        .send-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .doc-info {
            background: #f8f9ff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 600;
            color: #667eea;
        }

        .loader {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Document Parser</h1>
            <p>Upload a document and chat with AI</p>
        </div>

        <!-- Upload Section -->
        <div id="uploadSection">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📄</div>
                <h2>Click to upload your document</h2>
                <input type="file" id="fileInput" class="file-input" 
                       accept=".pdf,.txt,.doc,.docx">
                <p style="margin-top: 20px; color: #666;">
                    Supported: PDF, TXT, DOC, DOCX (Max 10MB)
                </p>
            </div>
            <div class="status" id="status"></div>
        </div>

        <!-- Chat Section -->
        <div id="chatSection">
            <div class="doc-info" id="docInfo">📄 Document: <span id="docName">-</span></div>

            <div class="chat-box" id="chatBox">
                <div class="message bot">
                    <div class="message-content">👋 Hi! Ask me anything about your document!</div>
                </div>
            </div>

            <div class="input-area">
                <input type="text" id="chatInput" class="chat-input" 
                       placeholder="Type your question..."
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send 🚀</button>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadSection = document.getElementById('uploadSection');
        const chatSection = document.getElementById('chatSection');
        const status = document.getElementById('status');
        const chatBox = document.getElementById('chatBox');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const docName = document.getElementById('docName');

        // Handle file selection
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // Validate file
            const validTypes = [
                'application/pdf',
                'text/plain',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ];

            if (!validTypes.includes(file.type)) {
                showStatus('❌ Invalid file type! Use PDF, TXT, DOC, or DOCX', 'error');
                return;
            }

            if (file.size > 10 * 1024 * 1024) {
                showStatus('❌ File too large! Maximum 10MB', 'error');
                return;
            }

            // Upload file
            await uploadFile(file);
        });

        async function uploadFile(file) {
            showStatus('⏳ Uploading and processing... Please wait', 'info');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus('✅ ' + result.message, 'success');

                    // Switch to chat
                    setTimeout(() => {
                        docName.textContent = result.filename;
                        uploadSection.style.display = 'none';
                        chatSection.style.display = 'block';
                        chatInput.focus();
                    }, 2000);
                } else {
                    showStatus('❌ ' + result.detail, 'error');
                }
            } catch (error) {
                showStatus('❌ Error: ' + error.message, 'error');
            }
        }

        function showStatus(message, type) {
            status.textContent = message;
            status.className = `status ${type} show`;
        }

        async function sendMessage() {
            const question = chatInput.value.trim();
            if (!question) return;

            addMessage(question, 'user');
            chatInput.value = '';
            sendBtn.disabled = true;

            const loadingMsg = addMessage('Thinking... <span class="loader"></span>', 'bot');

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                const result = await response.json();
                loadingMsg.remove();

                if (response.ok) {
                    addMessage(result.answer, 'bot');
                } else {
                    addMessage('❌ Error: ' + result.detail, 'bot');
                }
            } catch (error) {
                loadingMsg.remove();
                addMessage('❌ Error: ' + error.message, 'bot');
            } finally {
                sendBtn.disabled = false;
                chatInput.focus();
            }
        }

        function addMessage(text, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = text;

            messageDiv.appendChild(contentDiv);
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            return messageDiv;
        }
    </script>
</body>
</html>
    """


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    global current_document

    print(f"\n{'=' * 60}")
    print(f"📥 File received: {file.filename}")

    file_path = None
    try:
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"💾 File saved: {file_path}")

        # Process document
        current_document = file.filename
        process_document(str(file_path))

        print(f"✅ SUCCESS: Document ready!")
        print(f"{'=' * 60}\n")

        return JSONResponse({
            "message": f"Document processed successfully!",
            "filename": file.filename,
            "status": "success"
        })

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        if file_path and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with document"""
    global qa_chain

    if qa_chain is None:
        raise HTTPException(status_code=400, detail="Please upload a document first!")

    try:
        print(f"\n💬 Question: {request.question}")
        result = qa_chain({"question": request.question})
        print(f"✅ Answer generated\n")

        return ChatResponse(
            answer=result["answer"],
            confidence=0.85
        )
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "document_loaded": current_document is not None,
        "current_document": current_document
    }


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    print("\n" + "=" * 70)
    print("🚀 RAG CHATBOT STARTING")
    print("=" * 70)
    initialize_embeddings()
    print("✅ Ready to accept documents!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")