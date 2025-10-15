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