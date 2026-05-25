# Parth.ai — Premium Vector Database & RAG Engine

A high-fidelity, interactive **Vector Database Engine** and **Retrieval-Augmented Generation (RAG) pipeline** built entirely from scratch in Python with a premium, glassmorphic dark-theme web UI.

Parth.ai implements **HNSW (Hierarchical Navigable Small World)**, **KD-Tree (K-Dimensional Tree)**, and **Brute Force** search algorithms side-by-side, allowing you to test, visualize, and benchmark them in real-time.

---

## ✨ Features & Capabilities

* **🧠 Three Search Algorithms Side-by-Side:**
  * **HNSW (Approximate):** Production-grade multi-layer proximity graph with $O(\log N)$ search complexity.
  * **KD-Tree (Exact):** Axis-aligned binary partitioning tree with backtracking.
  * **Brute Force (Exact):** Linear scan baseline ($O(N \cdot d)$) guaranteed to find absolute nearest neighbors.
* **📏 Distance Metrics:** Cosine Distance ($1 - \text{sim}$), Euclidean Distance ($L_2$), and Manhattan Distance ($L_1$).
* **🗺️ PCA Dimensionality Projection:** Custom Principal Component Analysis engine in JavaScript, projecting 16D and 768D embeddings down to a stunning 2D interactive canvas.
* **📄 Dense Document Embedding:** Chunking text and generating real **768-D** embeddings locally via Ollama's `nomic-embed-text` model.
* **🤖 Fully Local RAG Pipeline:** Ask questions about your documents, retrieve semantic contexts via HNSW, and generate grounded answers using the local `llama3.2:1b` model.
* **📱 Premium Cyber-Glassmorphic UI:** Flawless responsive design that adapts dynamically to laptops (3-column), tablets (slide-out drawer), and mobile devices (sticky bottom navigation).

---

## 🛠️ Prerequisites

You need **two things** installed on your system:

1. **Python 3.8+** (to run the Flask server)
2. **[Ollama](https://ollama.com)** (to run local embeddings and generation LLMs)

---

## 🚀 Quick Start Guide

### Step 1 — Setup the AI Models (Ollama)
Install Ollama, ensure it is running in your system tray, and run the following commands in your terminal:

```powershell
# 1. Pull the 768-D Embedding Model (~274 MB)
ollama pull nomic-embed-text

# 2. Pull the 1.3 GB Generation LLM (~1.3 GB)
ollama pull llama3.2:1b
```

---

### Step 2 — Install Project Dependencies
Open your terminal in the project directory (`C:\Users\nerka\.gemini\antigravity-ide\scratch\parth-ai`) and install the requirements:

```powershell
pip install -r requirements.txt
```

---

### Step 3 — Start the Server
Start the Python backend server:

```powershell
python app.py
```

You should see:
```text
=== Parth.ai Vector Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2:1b
 * Running on http://127.0.0.1:8080
```

---

### Step 4 — Open the UI
Open your browser and navigate to:
👉 **[http://localhost:8080](http://localhost:8080)**

---

## 🗺️ Interactive UI Guide

### 1. Vector Search & PCA Projection
* **Query Space:** Enter descriptions (e.g. `sushi`, `basketball`, `binary search tree`) in the search box.
* **Algorithm & Metrics:** Switch between **HNSW**, **KD-Tree**, and **Brute Force**, and select your distance metric.
* **Canvas:** Watch query points and nearest neighbor matches glow, pulse, and connect via dashed semantic links on the custom PCA canvas.
* **All-Algo Benchmark:** Click **Compare All Algos** to run all three search engines at the same time and compare their latencies in microseconds.

### 2. Document Indexing (RAG Context)
* Paste lecture notes, documentation, articles, or books into the **Documents** tab.
* Click **Embed & Insert** to automatically split the document into 250-word chunks (with 30-word overlaps), generate 768D embeddings, and insert them into the Document HNSW graph.

### 3. Ask AI (Retrieval-Augmented Generation)
* Type a natural question about the documents you inserted.
* The system will perform an HNSW nearest neighbor lookup on your documents, inject the most relevant segments into a custom system prompt, and stream a typed response from `llama3.2:1b`.
* Click any **retrieved context chips** underneath the answer to view exactly which chunks of your files were read by the AI!

---

## 📡 REST API Reference

The server exposes a full REST API for programmatic interaction:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive user interface |
| `GET` | `/search?v=...&k=5&metric=cosine&algo=hnsw` | Performs a $k$-nearest neighbors query |
| `POST` | `/insert` | Inserts a custom demo vector (JSON body) |
| `DELETE` | `/delete/<id>` | Deletes a demo vector by ID |
| `GET` | `/items` | Lists all active demo vectors |
| `GET` | `/benchmark?v=...` | Compares execution times of all 3 algorithms |
| `GET` | `/hnsw-info` | Returns the multi-layer HNSW graph edge structure |
| `GET` | `/stats` | Returns database statistics (vector count, dims, metrics) |
| `POST` | `/doc/insert` | Embeds and stores a new document (JSON body) |
| `GET` | `/doc/list` | Lists all stored documents |
| `DELETE` | `/doc/delete/<id>` | Deletes a document chunk by ID |
| `POST` | `/doc/search` | Performs semantic search on documents only |
| `POST` | `/doc/ask` | Full RAG pipeline (retrieval + LLM generation) |
| `GET` | `/status` | Returns local Ollama model readiness and counts |

---

## 📁 Directory Structure

```text
parth-ai/
├── app.py                     # Python server (HNSW, KD-Tree, Brute Force, RAG, REST API)
├── index.html                 # Cyber-glassmorphic frontend & interactive PCA canvas
├── requirements.txt           # Python package dependencies
├── concept_documentation.md   # Mathematical & algorithmic deep dive
└── README.md                  # This setup and quickstart guide
```

---

## 📄 License
This project is licensed under the MIT License — feel free to modify, extend, and use it however you see fit!
