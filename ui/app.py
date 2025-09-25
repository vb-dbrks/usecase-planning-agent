from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import json
import os
from typing import List, Dict, Any

# Create FastAPI app
app = FastAPI(title="Usecase Delivery Planning Agent", version="1.0.0")

# Mount static files (React build)
app.mount("/static", StaticFiles(directory="build/static"), name="static")

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# Dependency to get WorkspaceClient
def get_workspace_client() -> WorkspaceClient:
    return WorkspaceClient()

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    client: WorkspaceClient = Depends(get_workspace_client)
):
    """
    Chat with the Usecase Delivery Planning Agent
    """
    try:
        # Prepare the conversation history
        messages = []
        
        # Add conversation history
        for msg in request.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Prepare the request payload for the serving endpoint
        payload = {
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1
        }
        
        # Get the serving endpoint name from environment variable
        serving_endpoint_name = os.getenv("SERVING_ENDPOINT_NAME", "usecase-planning-agent")
        
        # Query the serving endpoint
        response = client.serving_endpoints.query(
            name=serving_endpoint_name,
            dataframe_records=[payload]
        )
        
        # Extract the response
        if response and len(response.predictions) > 0:
            agent_response = response.predictions[0].get("choices", [{}])[0].get("message", {}).get("content", "No response generated")
            
            return ChatResponse(
                response=agent_response,
                status="success"
            )
        else:
            return ChatResponse(
                response="I apologize, but I couldn't generate a response. Please try again.",
                status="error"
            )
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with the agent: {str(e)}"
        )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "usecase-delivery-planning-agent"}

# Serve React app for all other routes
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # Serve index.html for all non-API routes
    return FileResponse("build/index.html")
