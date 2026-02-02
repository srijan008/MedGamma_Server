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
import chromadb
from .tools import WebSearchTool, EmergencyCallTool, EmergencySmsTool
from langchain_core.messages import ToolMessage
import json

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
        print("Context retrieved", recent_messages)
        # 3. Available Tools
        tools = [WebSearchTool, EmergencyCallTool, EmergencySmsTool]
        llm_with_tools = llm.bind_tools(tools)
        
        # Initialize context vars
        web_context = "" # Kept if we want to manually inject, but mostly handled by tool now
        print(f"Context Docs: {len(context_docs)}")
        lc_messages = []
        
        # System Message
        system_text = "You are a helpful AI assistant."
        
        if body.mode == "medgamma":
            system_text = """You are MedGamma, an advanced AI health assistant. 
            Your goal is to provide helpful, accurate, and empathetic health information.
            ALWAYS include a disclaimer: "I am an AI, not a doctor. Please consult a professional for medical advice."
            
            CRITICAL INSTRUCTION:
            You have access to tools for Emergency situations and Web Search.
            
            1. **Emergency**: If the user expresses CLEAR INTENT of SUICIDE ("I want to kill myself") or IMMEDIATE LIFE-THREAT ("I am bleeding out"), 
               you MUST call the `EmergencyCallTool`.
               If they express self-harm ("I might cut myself") but not immediate death, call `EmergencySmsTool`.
               If they are just stressed, anxious, or down, DO NOT call any tool. Provide support.
            
            2. **Information**: If the user asks about current events, news, or facts you don't know, call `WebSearchTool`.
            
            Keep your answers concise, professional, and supportive.
            """
            
        if session.summary:
            system_text += f"\n\nContext Summary of previous conversation:\n{session.summary}"
        


        if rag_context:
            system_text += f"{rag_context}\n\nAnswer using the provided document excerpts if relevant."
        
        lc_messages.append(SystemMessage(content=system_text))
        
        for msg in recent_messages:
            if msg.sender == "user":
                lc_messages.append(HumanMessage(content=msg.text))
            else:
                lc_messages.append(AIMessage(content=msg.text))
        
        async def response_generator():
            try:
                # Initial Invocation
                # Note: Streaming with tools in LangChain can be tricky.
                # We will loop: stream -> if tool_calls -> execute -> stream again
                
                final_answer_accumulated = ""
                
                # We need to manually manage the loop because `stream` essentially gives us chunks
                # and if a chunk indicates a tool call, we need to gather the full call, execute, and recurse.
                # Simplified approach: Use `invoke` for tool logic steps, and `stream` ONLY for final response?
                # OR: Standard ReAct loop.
                
                # Let's try a simpler approach compatible with streaming:
                # 1. Invoke with tools. 2. If valid response, yield. 3. If tool call, execute and re-invoke.
                
                current_messages = lc_messages.copy()
                
                # Step 1: Get Initial Response (possibly tool call)
                response = llm_with_tools.invoke(current_messages)
                
                # Loop to resolve all tool calls first
                # We use .invoke() here to effectively "think" and decide on tools.
                # If a tool is called, we execute it, append the result, and loop again.
                # We discard the text content from these intermediate steps because we want
                # to stream the FINAL answer freshly after all context is gathered.
                for _ in range(3):
                    if response.tool_calls:
                        # Append the AIMessage *once* with all tool calls
                        current_messages.append(response)
                        
                        for tool_call in response.tool_calls:
                            print(f"🔧 Tool Call: {tool_call['name']}: {tool_call['args']}")
                            
                            tool_result_content = ""
                            try:
                                if tool_call["name"] == "WebSearchTool":
                                    tool_result_content = WebSearchTool.invoke(tool_call["args"])
                                elif tool_call["name"] == "EmergencyCallTool":
                                    tool_result_content = EmergencyCallTool.invoke(tool_call["args"])
                                elif tool_call["name"] == "EmergencySmsTool":
                                    tool_result_content = EmergencySmsTool.invoke(tool_call["args"])
                            except Exception as e:
                                tool_result_content = f"Error executing tool: {str(e)}"
                                
                            print(f"✅ Tool Result: {tool_result_content[:50]}...")
                            
                            # Append Tool Result
                            current_messages.append(ToolMessage(content=tool_result_content, tool_call_id=tool_call["id"]))
                        
                        # Get next response (continue thinking with new context)
                        response = llm_with_tools.invoke(current_messages)
                        
                    else:
                        # No more tools, we have the final logic state.
                        # We break here and proceed to stream the final answer below.
                        break
                
                # Now that all tools are resolved and context is full, 
                # we stream the FINAL answer to the user.
                async for chunk in llm_with_tools.astream(current_messages):
                    content = chunk.content
                    if content:
                        final_answer_accumulated += content
                        yield content
                        
                # Save to DB
                print("✅ Conversation turn finished. Saving.")
                await db.message.create(
                    data={
                        "text": final_answer_accumulated,
                        "sender": "bot",
                        "chatSessionId": chat_id
                    }
                )
                
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
