"""
Standalone Databricks Migration Planning Agent for MLflow deployment.
This file contains a simplified agent that can be deployed without serialization issues.
"""

import mlflow.pyfunc
import dspy
from databricks.vector_search import VectorSearchClient
import os


class DatabricksMigrationAgent(mlflow.pyfunc.PythonModel):
    """
    Simplified Databricks Migration Planning Agent for MLflow deployment.
    """
    
    def load_context(self, context):
        """Initialize the agent when the model is loaded."""
        # Configure DSPy
        token = os.environ.get("DATABRICKS_TOKEN")
        workspace_url = os.environ.get("DATABRICKS_HOST", "https://adb-984752964297111.11.azuredatabricks.net")
        
        lm = dspy.LM(
            model="databricks/databricks-claude-sonnet-4",
            api_key=token,
            api_base=f"{workspace_url}/serving-endpoints",
        )
        dspy.configure(lm=lm)
        
        # Initialize vector search
        self.vsc = VectorSearchClient(disable_notice=True)
        
        # Configuration from environment or defaults
        self.vector_search_endpoint = os.environ.get("VECTOR_SEARCH_ENDPOINT", "use-case-planning-agent")
        self.vector_search_index = os.environ.get("VECTOR_SEARCH_INDEX", "vbdemos.usecase_agent.migration_plan_pdfs")
        
        # Simple DSPy modules
        self.qa_module = dspy.ChainOfThought("question, context -> answer")
        
    def predict(self, context, model_input, params=None):
        """Predict method for MLflow serving."""
        try:
            # Handle messages format
            if isinstance(model_input, dict) and "messages" in model_input:
                user_message = ""
                for message in model_input["messages"]:
                    if message.get("role") == "user":
                        user_message = message.get("content", "")
                        break
                
                # Extract question and context from message
                question = user_message
                context_info = ""
                
                if "Context:" in user_message:
                    parts = user_message.split("Context:")
                    question = parts[0].strip()
                    context_info = parts[1].strip() if len(parts) > 1 else ""
                
                # Retrieve relevant documents
                try:
                    search_results = self.vsc.get_index(
                        endpoint_name=self.vector_search_endpoint,
                        index_name=self.vector_search_index
                    ).similarity_search(
                        query_text=question,
                        columns=["path", "text"],
                        num_results=3
                    )
                    
                    # Format documents
                    docs_context = ""
                    if search_results:
                        if isinstance(search_results, dict) and 'result' in search_results:
                            data_array = search_results['result'].get('data_array', [])
                        elif isinstance(search_results, list):
                            data_array = search_results
                        else:
                            data_array = []
                        
                        docs_context = "\n".join([
                            f"Source: {doc.get('path', 'Unknown')}\nContent: {doc.get('text', '')[:500]}..."
                            for doc in data_array if isinstance(doc, dict)
                        ])
                
                except Exception as e:
                    docs_context = f"Error retrieving documents: {str(e)}"
                
                # Generate answer with migration focus
                full_context = f"""
Question: {question}
User Context: {context_info}
Retrieved Documents: {docs_context}

Please provide a comprehensive answer for this Databricks migration planning question. 
Focus on migrating TO Databricks from existing data/analytics platforms.
If this is about planning, provide structured information including:
- Timeline considerations
- Resource requirements  
- Migration phases
- Risk assessment
- Best practices

Format the response as structured tables when appropriate using markdown table format.
"""
                
                result = self.qa_module(question=question, context=full_context)
                
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": result.answer
                            }
                        }
                    ]
                }
            
            # Handle simple string input
            elif isinstance(model_input, str):
                result = self.qa_module(question=model_input, context="Databricks migration planning")
                return {"answer": result.answer}
            
            else:
                return {"error": "Invalid input format"}
                
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}


# Set the model for MLflow
mlflow.models.set_model(DatabricksMigrationAgent())

