from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import tempfile
from database import db
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import traceback
import chromadb
from .web_helpers import run_web_search, route_query

router = APIRouter(prefix="/chat", tags=["chat"])

# --- Models ---
class CreateChatResponse(BaseModel):
    uuid: str

class Message(BaseModel):
    text: str
    sender: str
    timestamp: str

class ChatHistoryResponse(BaseModel):
    messages: List[Message]
    summary: Optional[str] = None

class UserMessage(BaseModel):
    message: str
    mode: str = "general" # 'general' or 'medgamma'

from .services import llm, embeddings
# Constants
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT") 
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# Initialize Chroma Client with Auth
# Note: For hosted Chroma, we use HttpClient. If local, standard Client.
# Based on env vars presence, we assume hosted if keys are present.
try:
    if CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE:
        print("🔌 Connecting to Hosted ChromaDB...")
        chroma_client = chromadb.HttpClient(
            ssl=True,
            host='api.trychroma.com',
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            headers={
                'x-chroma-token': CHROMA_API_KEY
            }
        )
    else:
        print("📂 Using Local ChromaDB...")
        # Fallback to local persistent storage usually
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Wrapper for Langchain
    vector_store = Chroma(
        client=chroma_client,
        collection_name="chatbot_docs",
        embedding_function=embeddings,
    )
except Exception as e:
    print(f"🔥 Error connecting to ChromaDB: {e}")
    vector_store = None


# --- Helper Functions ---
async def update_summary(chat_id: str):
    """
    Background task to update the summary of the conversation.
    Summarizes all messages except the last 5.
    """
    try:
        session = await db.chatsession.find_unique(
            where={"id": chat_id},
            include={"messages": True}
        )
        if not session or not session.messages:
            return

        sorted_messages = sorted(session.messages, key=lambda m: m.timestamp)
        
        if len(sorted_messages) <= 5:
            return

        messages_to_summarize = sorted_messages[:-5]
        
        conversation_text = ""
        for msg in messages_to_summarize:
            conversation_text += f"{msg.sender}: {msg.text}\n"

        summary_prompt = f"""
        Summarize the following conversation concisely, retaining key facts and context.
        
        Conversation:
        {conversation_text}
        
        Summary:
        """
        
        response = llm.invoke(summary_prompt)
        new_summary = response.content

        await db.chatsession.update(
            where={"id": chat_id},
            data={"summary": new_summary}
        )
        print(f"✅ Summary updated for chat {chat_id}")

    except Exception as e:
        print(f"🔥 Error updating summary: {e}")
        traceback.print_exc()

# --- Routes ---

@router.post("/new", response_model=CreateChatResponse)
async def create_new_chat():
    try:
        session = await db.chatsession.create(data={})
        return {"uuid": session.id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat_history(chat_id: str):
    try:
        session = await db.chatsession.find_unique(
            where={"id": chat_id},
            include={"messages": True},
        )

        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        sorted_messages = sorted(session.messages, key=lambda m: m.timestamp)
        msg_list = [
            Message(text=m.text, sender=m.sender, timestamp=m.timestamp.isoformat())
            for m in sorted_messages
        ]
        
        return {"messages": msg_list, "summary": session.summary}

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 ERROR /chat/{chat_id}:", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{chat_id}/message")
async def send_message(chat_id: str, body: UserMessage, background_tasks: BackgroundTasks):
    try:
        # 1. Save User Message
        await db.message.create(
            data={
                "text": body.message,
                "sender": "user",
                "chatSessionId": chat_id
            }
        )

        # 2. Retrieve Context (History + Summary + RAG)
        session = await db.chatsession.find_unique(
            where={"id": chat_id},
            include={"messages": True}
        )
        if not session:
             raise HTTPException(status_code=404, detail="Chat session not found")

        # RAG Retrieval
        context_docs = []
        if vector_store:
            try:
                results = vector_store.similarity_search(body.message, k=3, filter={"chat_id": chat_id})
                context_docs = results
                print(f"📚 Retrieved {len(results)} chunks from Chroma")
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")

        rag_context = ""
        if context_docs:
            rag_context = "\n\nRelevant Document Excerpts:\n" + "\n---\n".join([doc.page_content for doc in context_docs])

        sorted_messages = sorted(session.messages, key=lambda m: m.timestamp)
        recent_messages = sorted_messages[-5:] 
        
        # Build history string for router
        history_str = "\n".join([f"{m.sender}: {m.text}" for m in recent_messages])
        
        # Decide: Web Search?
        routing_decision = await route_query(body.message, history_str)
        print(f"🧭 Routing Decision: {routing_decision}")
        
        web_context = ""
        if routing_decision == "WEB":
            print("🌐 Performing Web Search...")
            search_results = run_web_search(body.message)
            print(f"🕸️ Search Results Length: {len(search_results)}")
            if search_results:
                web_context = f"\n\nWeb Search Results:\n{search_results}\n"
        print("web_context", web_context)
        lc_messages = []
        
        # System Message
        system_text = "You are a helpful AI assistant."
        
        if body.mode == "medgamma":
            system_text = """You are MedGamma, an advanced AI health assistant. 
            Your goal is to provide helpful, accurate, and empathetic health information.
            ALWAYS include a disclaimer: "I am an AI, not a doctor. Please consult a professional for medical advice."
            
            CRITICAL INSTRUCTION:
            If the user expresses clear intent of SUICIDE, SELF-HARM, or is in an IMMEDIATE LIFE-THREATENING EMERGENCY, you MUST:
            1. Start your response with the exact token: "[SOS]" (with brackets).
            2. Then, calmly urge them to call emergency services or a suicide hotline.
            
            IMPORTANT DISTINCTION:
            - "I want to hurt myself" (Self-Harm) -> Use [SOS_SMS]. (Medium Severity)
            - "I want to kill myself" (Suicide) -> Use [SOS_CALL]. (High Severity/Critical)
            - "I am stressed/anxious" -> NO TOKEN. Supportive response only.

            EXAMPLES:
            User: "I am so stressed."
            AI: "I hear you. Have you tried deep breathing?" (NO SOS)
            
            User: "I want to hurt myself. I might cut my arm."
            AI: "[SOS_SMS] Please don't. You are valuable. Reach out to a friend." (SMS Only)

            User: "I am going to kill myself now. Goodbye."
            AI: "[SOS_CALL] PLEASE STOP. Call 911 immediately. We are here for you." (Call + SMS)

            Keep your answers concise, professional, and supportive.
            """
            
        if session.summary:
            system_text += f"\n\nContext Summary of previous conversation:\n{session.summary}"
        
        if web_context:
            system_text += f"{web_context}\n\nUse the Web Search Results to answer the user's question if it requires up-to-date information."

        if rag_context:
            system_text += f"{rag_context}\n\nAnswer using the provided document excerpts if relevant."
        
        lc_messages.append(SystemMessage(content=system_text))
        
        for msg in recent_messages:
            if msg.sender == "user":
                lc_messages.append(HumanMessage(content=msg.text))
            else:
                lc_messages.append(AIMessage(content=msg.text))
        
        async def response_generator():
            full_response_raw = ""
            full_response_clean = ""
            buffer = ""
            checked_for_token = False
            
            try:
                # LLM Stream
                for chunk in llm.stream(lc_messages):
                    content = chunk.content
                    if not content: continue
                    
                    full_response_raw += content
                    
                    if not checked_for_token:
                        buffer += content
                        # Check if buffer has enough chars (longest token is [SOS_CALL] = 10 chars)
                        if len(buffer) > 15 or "]" in buffer:
                            # Strip known tokens
                            clean_buffer = buffer.replace("[SOS_CALL]", "").replace("[SOS_SMS]", "").replace("[SOS]", "").lstrip()
                            
                            full_response_clean += clean_buffer
                            yield clean_buffer
                            
                            buffer = ""
                            checked_for_token = True
                    else:
                        full_response_clean += content
                        yield content
                
                # Yield any remaining buffer if stream ended early
                if buffer:
                   clean_buffer = buffer.replace("[SOS_CALL]", "").replace("[SOS_SMS]", "").replace("[SOS]", "").lstrip()
                   full_response_clean += clean_buffer
                   yield clean_buffer

                # After stream finishes, save CLEANED message to DB
                print("✅ Stream finished. Saving to DB...")
                await db.message.create(
                    data={
                        "text": full_response_clean,
                        "sender": "bot",
                        "chatSessionId": chat_id
                    }
                )

                # Use RAW response for trigger logic
                full_response = full_response_raw

                # --- AI DISTRESS DETECTION ---
                from .emergency import trigger_emergency, EmergencyRequest
                
                if "[SOS_CALL]" in full_response:
                    print("🚨 CRITICAL DISTRESS (CALL+SMS): Triggering Emergency...")
                    request = EmergencyRequest(type="sos", severity="critical", location="High Risk Detected via Chat")
                    background_tasks.add_task(trigger_emergency, request)
                    
                elif "[SOS_SMS]" in full_response:
                    print("⚠️ MEDIUM DISTRESS (SMS ONLY): Triggering Alert...")
                    request = EmergencyRequest(type="sos", severity="medium", location="Medium Risk Detected via Chat")
                    background_tasks.add_task(trigger_emergency, request)
                
                # Handling legacy cases just in case
                elif "[SOS]" in full_response:
                    print("🚨 LEGACY DISTRESS: Triggering Critical...")
                    request = EmergencyRequest(type="sos", severity="critical", location="Crisis Detected")
                    background_tasks.add_task(trigger_emergency, request)
                
                # Trigger summary update (manually calling logic or using a separate non-background task approach since background_tasks might run early)
                # Background tasks in FastAPI run after response is sent.
                # Since we are streaming, we can't easily attach BackgroundTask to StreamingResponse in a standard way that guarantees execution order relative to stream end inside the generator.
                # So we just run it here (awaiting might delay logic, but it's background for user perspective since stream is done).
                # Actually, better to fire-and-forget or await quick update.
                await update_summary(chat_id)
                
            except Exception as e:
                print(f"🔥 Error during stream: {e}")
                traceback.print_exc()

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        print(f"🔥 Error in send_message: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{chat_id}/upload")
async def upload_pdf(chat_id: str, file: UploadFile = File(...)):
    try:
        session = await db.chatsession.find_unique(where={"id": chat_id})
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
        finally:
            file.file.close()

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata (chat_id) to link docs to this session
            for chunk in chunks:
                chunk.metadata["chat_id"] = chat_id
                
            # Store in Chroma
            if vector_store:
                # Add unique IDs for chunks to avoid duplicates if re-uploaded (optional logic)
                vector_store.add_documents(chunks)
                print(f"✅ Indexed {len(chunks)} chunks for chat {chat_id}")
            else:
                print("⚠️ Vector store not initialized, skipping indexing")

        except Exception as e:
             print(f"🔥 Error processing PDF: {e}")
             raise HTTPException(status_code=500, detail=f"Error parsing PDF: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        await db.message.create(
            data={
                "text": f"PDF '{file.filename}' uploaded and analyzed. I am ready to answer questions about it.",
                "sender": "bot",
                "chatSessionId": chat_id,
            }
        )

        return {"success": True, "message": "File uploaded and indexed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 ERROR /chat/{chat_id}/upload:", e)
        raise HTTPException(status_code=500, detail=str(e))
