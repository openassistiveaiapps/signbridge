# signbridge
AI-powered open-source communication app that helps non-verbal autistic children express themselves through gesture recognition and personalized suggestions.

SignBridge is an open-source assistive communication app designed for children on the autism spectrum, especially those who are non-verbal. Using AI-powered gesture recognition, personalization, and context-aware suggestions, SignBridge bridges the gap between sign-based communication and real-world needs.

When a child performs a gesture (e.g., “More”), the system interprets it, retrieves relevant options (e.g., more water, more juice, more playtime), and presents them visually or via text-to-speech. Over time, the app learns the child’s preferences (favorite foods, activities, routines) and tailors responses accordingly.

Built with open-source AI frameworks, SignBridge is:

🎥 Gesture-aware (MediaPipe/YOLO/OpenPose)

🧠 Contextual & Adaptive (RAG with FAISS/Qdrant + Embeddings)

🗣️ Accessible (React Native mobile app + TTS output)

🔓 Free & Open-source (runs fully offline for privacy)

Our mission is to empower children with autism and their families by providing a cost-free, intelligent, and customizable communication tool.

Flow Summary (RAG-enabled)

Gesture Capture: Camera → landmarks → Gesture Agent → embedding.

Intent Mapping: Embedding + context → Intent Agent → intent vector.

RAG Retrieval: Query vector → Retriever → VectorStore → top-K relevant favorites/notes.

Generation: LLM Generator produces natural-language suggestion.

Output: Suggestion returned to UI + TTS.

Feedback Loop: Child/caregiver choice → DB & VectorStore update for personalization.

| Layer / Function      | Component              | Technology / Library                              | Role / Purpose                                                                   |
| --------------------- | ---------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------- |
| Frontend / Device     | Child UI               | React Native + Expo                               | Displays suggestions, camera preview, collects input from child                  |
| Gesture Recognition   | Gesture Agent          | MediaPipe Hands/Pose, PyTorch / TFLite            | Detects gestures, extracts landmarks, outputs embeddings; AI-powered             |
| Backend Orchestration | Inference Service      | FastAPI (Python), Uvicorn                         | Receives gesture events, orchestrates agents, RAG pipeline, TTS                  |
| Intent Understanding  | Intent Agent           | Custom rules + embeddings (Sentence-Transformers) | Maps gestures → semantic intents; context-aware; part of AI/ML logic             |
| RAG Layer             | Embeddings             | Sentence-Transformers                             | Converts intents + context + caregiver notes → vector representation             |
| RAG Layer             | Vector Retriever       | FAISS (local) / Qdrant (Docker optional)          | Retrieves top-K relevant items from vector store (favorites, notes, history)     |
| RAG Layer             | Vector Store           | FAISS / Qdrant                                    | Stores retrievable knowledge as vectors; supports personalization & context      |
| Generator (GenAI)     | LLM Generator          | Llama 3 / Mistral 7B (quantized, local)           | Generates natural-language suggestions dynamically; integrates RAG results       |
| Database              | DB                     | SQLite (local) / Postgres                         | Stores users, gestures, preferences, history, feedback; supports personalization |
| Storage               | Clip / Example Storage | Local filesystem / optional S3                    | Stores landmark sequences, videos, labeled examples for few-shot learning        |
| Output / UX           | TTS                    | Coqui TTS / System TTS                            | Converts suggestions → audio feedback for the child                              |
| Caregiver / Admin     | Teach UI               | React Native / Web                                | Allows caregivers to add new gestures, update preferences, edit vector store     |

✅ Highlights:

AI/ML:

Gesture Agent: Computer Vision (MediaPipe + PyTorch/TFLite)

Intent Agent: Embeddings + context mapping

Embeddings Layer: Sentence-Transformers (vector representations)

GenAI:

Generator: Llama 3 / Mistral 7B generates natural-language suggestions dynamically

RAG:

Vector Store + Retriever: FAISS / Qdrant

DB / Storage:

SQLite/Postgres for history/preferences

Clip storage for few-shot personalization

Agents:

Gesture Agent, Intent Agent, Generator, UI/TTS orchestrated via FastAPI

# frontend

1. Install dependencies

   ```bash
   npm install
   ```

2. Start the app

   ```bash
   npx expo start
   ```

# backend

1. In your backend project folder - This will create a folder named venv containing an isolated Python environment.

   ```bash
  python -m venv venv
   ```

2. Activate the virtual environment

```Windows (PowerShell):
.\venv\Scripts\Activate
```

```
Windows (cmd):
venv\Scripts\activate
```

```
macOS / Linux:
source venv/bin/activate
```
✅ When activated, you should see (venv) at the beginning of your terminal prompt

3. Install dependencies - Make sure you’re inside the virtual environment (you’ll see (venv)), then:

```
pip install fastapi uvicorn pydantic
```
(You can also install any other packages like torch, scikit-learn, etc., if needed.)

4. Run the backend app

```
uvicorn main:app --reload --port 8000
```

main → name of your Python file (without .py)
app → FastAPI instance variable inside that file
--reload → automatically restarts the server on code changes
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
✅ Now your backend is live and your frontend can call http://localhost:8000/api/gesture.

5. To deactivate the venv

```
deactivate
```