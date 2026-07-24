# Requirements Document

## Introduction

This document specifies the requirements for `app.py`, the Phase 2 web-based Chat GUI for the local RAG pipeline. The application connects to the existing persistent ChromaDB vector store at `./chroma_db`, retrieves semantically relevant document chunks for each user query, and generates answers using Google's Gemini 1.5 Flash model via LangChain. The interface is built with Streamlit and enforces strict source-only answering with inline citations. The API key is loaded at runtime from the `.env` file — no environment setup, test files, or package installation is in scope for this document.

## Glossary

- **Chat_App**: The `app.py` Streamlit application responsible for the full query-to-answer pipeline.
- **Session_State**: Streamlit's `st.session_state` dictionary used to persist chat history across reruns.
- **Chat_History**: The list of `{"role": "user"|"assistant", "content": str}` dictionaries stored in Session_State.
- **Vector_Store**: The persistent Chroma instance loaded from `./chroma_db` using `HuggingFaceEmbeddings`.
- **Retriever**: A LangChain retriever wrapping the Vector_Store, configured to return the top 3 most relevant chunks (`k=3`).
- **LLM**: The `ChatGoogleGenerativeAI` instance using model `gemini-1.5-flash`, authenticated via the `GOOGLE_API_KEY` environment variable.
- **System_Prompt**: A fixed instruction template injected into every LLM call instructing it to answer only from retrieved context and to append source filenames.
- **Context_Block**: The concatenated text of the retrieved chunks, passed to the LLM as part of the System_Prompt.
- **Citation**: The source filename extracted from each retrieved chunk's `metadata["source"]` field, appended to the LLM answer.
- **Sidebar**: The Streamlit sidebar panel containing the app title and the Clear Chat button.

---

## Requirements

### Requirement 1: Session State Initialisation

**User Story:** As a user, I want my conversation history to persist across Streamlit reruns, so that the chat screen does not reset every time I submit a message.

#### Acceptance Criteria

1. WHEN the Chat_App starts or reruns, THE Chat_App SHALL check whether the key `"messages"` exists in Session_State.
2. IF the key `"messages"` does not exist in Session_State, THEN THE Chat_App SHALL initialise it as an empty list before any other UI component renders.
3. WHILE a session is active, THE Chat_App SHALL append every user message and every assistant response to Session_State `"messages"` as a dictionary with keys `"role"` and `"content"`, where `"role"` is exactly `"user"` for user messages and `"assistant"` for assistant responses, and `"content"` is a non-empty string of at most 32,000 characters.
4. WHEN the Chat_App reruns, THE Chat_App SHALL iterate over Session_State `"messages"` and render each entry that contains both a valid `"role"` value (`"user"` or `"assistant"`) and a non-empty `"content"` string using `st.chat_message`, skipping any malformed entries, so that the full valid conversation history is visible to the user.

---

### Requirement 2: Vector Store and Retriever Initialisation

**User Story:** As a developer, I want the app to load the existing ChromaDB vector store once at startup, so that each query uses the same pre-built index without re-loading on every rerun.

#### Acceptance Criteria

1. WHEN the Chat_App initialises, THE Vector_Store SHALL be loaded from `./chroma_db` using `Chroma` with the `HuggingFaceEmbeddings` model `all-MiniLM-L6-v2` as the embedding function.
2. WHEN the Vector_Store is successfully loaded, THE Chat_App SHALL wrap it as a Retriever using `as_retriever` with `search_kwargs={"k": 3}` to return the top 3 most relevant chunks per query; both the Vector_Store and the Retriever SHALL be initialised within a single `@st.cache_resource`-decorated function.
3. THE Chat_App SHALL use Streamlit's `@st.cache_resource` decorator so that the Vector_Store and Retriever are created only once per server process and reused on every rerun.
4. IF the `./chroma_db` directory is missing or the Chroma collection cannot be loaded, THEN THE Chat_App SHALL display a `st.error` message describing the failure and call `st.stop()` to halt further rendering.

---

### Requirement 3: LLM Initialisation

**User Story:** As a developer, I want the app to load the Gemini API key from the `.env` file and initialise the LLM once, so that secrets are not hard-coded and the model is ready before the first query.

#### Acceptance Criteria

1. WHEN the Chat_App starts, THE Chat_App SHALL call `load_dotenv()` to load the `GOOGLE_API_KEY` from the `.env` file into the environment before initialising the LLM.
2. WHEN `load_dotenv()` has been called and `GOOGLE_API_KEY` is present in the environment, THE LLM SHALL be initialised as a `ChatGoogleGenerativeAI` instance with `model="gemini-1.5-flash"` inside a `@st.cache_resource`-decorated function, ensuring it is created only once per server process.
3. IF the `GOOGLE_API_KEY` environment variable is not set after calling `load_dotenv()`, THEN THE Chat_App SHALL display a `st.error` message stating the key is missing and call `st.stop()` to halt further rendering.

---

### Requirement 4: System Prompt and Strict Context-Only Answering

**User Story:** As a product owner, I want the LLM to answer only from retrieved document chunks, so that the app does not hallucinate or produce answers beyond the ingested knowledge base.

#### Acceptance Criteria

1. THE System_Prompt SHALL instruct the LLM to answer the user's question using only the information present in the provided Context_Block and to refuse any inference beyond that content.
2. WHEN the Context_Block consists entirely of chunks whose combined text does not contain any terms or concepts related to the user's question, THE LLM SHALL respond with exactly: `"I do not have enough information to answer that based on the provided documents."` and SHALL NOT append a `Sources:` section to that response.
3. THE System_Prompt SHALL be a fixed string template with exactly two placeholders: `{context}` for the Context_Block and `{question}` for the user's query, with no other variable interpolation.
4. WHEN the LLM generates a substantive answer (i.e., not the fallback phrase), THE System_Prompt SHALL instruct the LLM to append a `Sources:` section listing each unique source filename present in the retrieved chunks' metadata; IF a retrieved chunk has no `metadata["source"]` field, THE System_Prompt SHALL instruct the LLM to label that source as `"Unknown"`.

---

### Requirement 5: Citation of Source Documents

**User Story:** As a user, I want each answer to include the filenames of the source documents it drew from, so that I can verify which document the information came from.

#### Acceptance Criteria

1. WHEN the LLM generates a response, THE Chat_App SHALL extract the `metadata["source"]` field from each of the retrieved chunks; IF a chunk is missing the `metadata["source"]` field, THE Chat_App SHALL substitute the label `"Unknown"` for that chunk.
2. WHEN source filenames have been extracted from all retrieved chunks, THE Chat_App SHALL deduplicate them and pass the resulting unique list into the formatted System_Prompt so the LLM can reference them in its answer.
3. WHEN the LLM produces a substantive answer, the response text displayed in the chat window SHALL end with a `Sources:` section formatted as a labelled list of unique source filenames, one per line, appearing after the answer body.

---

### Requirement 6: Sidebar

**User Story:** As a user, I want a clean sidebar with an app title and a button to clear the conversation, so that I can start a fresh session without refreshing the browser.

#### Acceptance Criteria

1. THE Sidebar SHALL display the title `"📚 KTlite — RAG Chat"` as its header.
2. THE Sidebar SHALL contain a button labelled `"Clear Chat"`.
3. WHEN the `"Clear Chat"` button is clicked, THE Chat_App SHALL set Session_State `"messages"` to an empty list, so that the chat window displays no messages on the next rerun.

---

### Requirement 7: Chat Input and Message Rendering

**User Story:** As a user, I want a chat input box at the bottom of the page and clearly styled message bubbles, so that the interface feels like a familiar chat application.

#### Acceptance Criteria

1. THE Chat_App SHALL render a chat input field using `st.chat_input` with the placeholder text `"Ask a question about your documents..."`.
2. WHEN the user submits a query via `st.chat_input`, THE Chat_App SHALL immediately render the user's message in the chat window using `st.chat_message("user")` and append a `{"role": "user", "content": query}` entry to Session_State `"messages"` before invoking the Retriever.
3. WHEN the user submits a query, THE Chat_App SHALL invoke the Retriever with the query string to obtain up to 3 relevant chunks; IF the Retriever returns zero chunks, THE Context_Block SHALL be set to an empty string and the source list SHALL be set to `["Unknown"]`.
4. WHEN chunks are retrieved, THE Chat_App SHALL build the Context_Block by joining the `page_content` of all retrieved chunks with `"\n\n"` as separator, deduplicate the source filenames, and pass both into the formatted System_Prompt along with the user's question.
5. WHEN the formatted prompt is ready, THE Chat_App SHALL invoke the LLM using `.stream()` and render the streaming response tokens inside an `st.chat_message("assistant")` block using `st.write_stream`.
6. WHEN the LLM response stream is complete, THE Chat_App SHALL append a `{"role": "assistant", "content": full_response}` entry to Session_State `"messages"`.
7. IF the LLM invocation raises an exception, THE Chat_App SHALL display a `st.error` message inside the assistant chat block describing the failure, and SHALL NOT append a malformed entry to Session_State `"messages"`.
