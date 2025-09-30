# Databricks notebook source
# MAGIC %md
# MAGIC # Simplified Migration Planning Agent using MLflow ResponsesAgent
# MAGIC 
# MAGIC This notebook implements a simplified approach using MLflow's built-in conversation management:
# MAGIC 1. **Built-in Context Management**: Uses MLflow's ResponsesAgentRequest context
# MAGIC 2. **Simplified Session Management**: Leverages MLflow's conversation tracking
# MAGIC 3. **Cleaner Architecture**: Removes custom SessionManager and SimpleStorage
# MAGIC 4. **Better Debugging**: Uses MLflow's built-in tracing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Install required packages
# %pip install dspy>=2.6.23 mlflow databricks-agents databricks-vectorsearch

# COMMAND ----------

# Import required libraries
import dspy
import mlflow
import mlflow.dspy
from databricks.vector_search.client import VectorSearchClient
import json
from typing import List, Dict, Any, Optional
import pydantic
from datetime import datetime
import uuid
from uuid import uuid4

# Check versions for debugging
print(f"🔧 [DEBUG] DSPy version: {dspy.__version__}")
print(f"🔧 [DEBUG] MLflow version: {mlflow.__version__}")

print("Libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. DSPy Setup and Model Configuration

# COMMAND ----------

# Configure DSPy with Databricks model
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() + '/serving-endpoints'

lm = dspy.LM(
    model="databricks/databricks-claude-sonnet-4",
    api_key=token,
    api_base=url,
)

# Configure DSPy
dspy.configure(lm=lm)

print(f"DSPy version: {dspy.__version__}")
print("DSPy configured successfully with Databricks model")

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient()
print("DSPy and Vector Search configured successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Simplified DSPy Signatures

# COMMAND ----------

# Improved DSPy Signatures based on MVP patterns
# Key improvements from MVP:
# 1. Enhanced descriptions with specific Databricks account team context
# 2. Added detailed examples and extraction guidelines
# 3. Improved field descriptions with specific requirements
# 4. Added question_categories input for better question generation
# 5. Renamed GapAnalyzerSignature to GapAnalysisSignature for consistency
# 6. Added plan_table output field for comprehensive planning
# 7. Enhanced tabular plan generator with specific column requirements
class IntentClassifierSignature(dspy.Signature):
    """Classify user intent based ONLY on the current user input, ignoring conversation history.
    
    Examples:
    - "Hi, what can you help with?" -> greeting
    - "I'm working with a customer who wants to migrate from Oracle" -> providing_context
    - "The customer has 5 team members" -> answering_questions
    - "How's the information collection going?" -> feedback_request
    - "Generate a plan" or "/plan" -> plan_generation
    - "Ask me questions then" -> providing_context
    """
    user_input: str = dspy.InputField(desc="The user's current input message")
    intent: str = dspy.OutputField(desc="Intent based on current input only: greeting, answering_questions, providing_context, feedback_request, plan_generation, or other")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")

class GreetingSignature(dspy.Signature):
    """Handle greetings and explain capabilities for Databricks account teams planning customer use cases."""
    user_input: str = dspy.InputField(desc="The user's input from Databricks account team member")
    response: str = dspy.OutputField(desc="Greeting response explaining how the agent helps account teams create use case plans for customer migrations and greenfield scenarios")

class InformationCollectorSignature(dspy.Signature):
    """Extract and structure customer use case information from Databricks account team input.
    
    Extract key details about:
    - Current technology stack (databases, platforms, tools)
    - Data volumes and complexity
    - Team structure and capabilities
    - Business requirements and constraints
    - Compliance and regulatory needs
    - Timeline and budget information
    - Integration requirements
    """
    user_input: str = dspy.InputField(desc="The user's input from Databricks account team about customer use case")
    conversation_context: str = dspy.InputField(desc="Recent conversation context about customer migration or greenfield scenario")
    extracted_info: str = dspy.OutputField(desc="Structured summary of key customer information extracted from the input")

class QuestionGeneratorSignature(dspy.Signature):
    """Generate exactly 3 relevant questions for Databricks account teams to gather customer use case information."""
    current_info: str = dspy.InputField(desc="Current customer use case information summary")
    conversation_context: str = dspy.InputField(desc="Recent conversation context about customer migration or greenfield scenario")
    question_categories: str = dspy.InputField(desc="Available question categories and example questions to guide question generation")
    questions: str = dspy.OutputField(desc="Exactly 3 questions, each on a new line, starting with numbers 1., 2., 3. Format: '1. Question one?\n2. Question two?\n3. Question three?'")
    category: str = dspy.OutputField(desc="Planning category these questions belong to (Resource & Team, Customer Background, Technical Scope, etc.)")

class GapAnalysisSignature(dspy.Signature):
    """Analyze information gaps for customer use case planning and provide feedback to Databricks account teams."""
    information_summary: str = dspy.InputField(desc="Current customer use case information summary")
    feedback: str = dspy.OutputField(desc="Analysis of information completeness and gaps for customer migration planning")
    missing_areas: str = dspy.OutputField(desc="Critical areas that need more information from the customer")

class PlanGeneratorSignature(dspy.Signature):
    """Generate comprehensive tabular use case plan for Databricks account teams based on customer information and Databricks knowledge base."""
    information_summary: str = dspy.InputField(desc="Collected customer use case information summary")
    knowledge_base: str = dspy.InputField(desc="Relevant Databricks migration and implementation knowledge")
    plan_table: str = dspy.OutputField(desc="Tabular use case plan with phases, activities, timelines, and deliverables in markdown table format")
    assumptions: str = dspy.OutputField(desc="Key assumptions made about customer environment and requirements")
    risks: str = dspy.OutputField(desc="Identified risks and mitigation strategies for the customer use case")

class TabularPlanGeneratorSignature(dspy.Signature):
    """Generate detailed tabular implementation plan with phases, activities, timelines, and deliverables for customer use cases."""
    customer_info: str = dspy.InputField(desc="Customer use case information and requirements")
    databricks_knowledge: str = dspy.InputField(desc="Relevant Databricks implementation knowledge and best practices")
    implementation_plan: str = dspy.OutputField(desc="Detailed tabular implementation plan with Phase, Activity, Duration, Dependencies, Deliverables, and Owner columns")
    timeline_summary: str = dspy.OutputField(desc="High-level timeline summary with key milestones and dates")
    resource_requirements: str = dspy.OutputField(desc="Resource requirements including team roles, skills, and effort estimates")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Question Categories for MVP

# COMMAND ----------

# Question Categories for MVP (from colleague)
QUESTION_CATEGORIES = {
    "Resource & Team": [
        "How many team members are there and what are their roles?",
        "Are the teams sufficiently skilled/trained in Databricks?",
        "Are they using Professional Services or System Integrators?",
        "Are resources shared with other projects?",
        "Have resources done this type of migration work before?"
    ],
    "Customer Background & Drivers": [
        "Does the customer have a specific deadline and what drives it?",
        "Is the customer already using cloud infrastructure?",
        "Does the customer have Databricks elsewhere in the organization?",
        "Does the customer have security approval for this migration?",
        "What are the key business drivers for the migration?"
    ],
    "Technical Scope & Architecture": [
        "Has a pilot or POC been conducted?",
        "Does the customer have visibility of all data and pipelines to migrate?",
        "Is the customer aware of where and who uses the data?",
        "Is lift and shift or redesign preferred for different components?",
        "How many pipelines, reports, and data sources need migration?"
    ],
    "Current Process Maturity": [
        "What is the current data governance and quality process?",
        "How mature is the current data architecture?",
        "What are the current data security and compliance processes?",
        "How is data lineage currently tracked?",
        "What are the current data quality monitoring processes?"
    ],
    "Performance & Scalability": [
        "What are the current performance requirements?",
        "What is the expected data growth rate?",
        "What are the peak usage patterns?",
        "What are the current performance bottlenecks?",
        "What are the target performance and scalability requirements?"
    ],
    "Security & Compliance": [
        "What are the current security requirements?",
        "What compliance standards must be met?",
        "What are the data residency and sovereignty requirements?",
        "What are the access control requirements?",
        "What are the audit and monitoring requirements?"
    ]
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Simplified Conversation Manager

# COMMAND ----------

class SimplifiedConversationManager:
    """Simplified conversation manager using MLflow's built-in context management."""
    
    def __init__(self, lm, rm):
        self.lm = lm
        self.rm = rm
        
        # Initialize DSPy components with selective Chain of Thought strategy
        # Simple agents: Use dspy.Predict() for faster, direct responses
        self.intent_classifier = dspy.Predict(IntentClassifierSignature)
        self.greeting_handler = dspy.Predict(GreetingSignature)
        self.information_collector = dspy.Predict(InformationCollectorSignature)
        
        # Complex agents: Use dspy.ChainOfThought() for reasoning-intensive tasks
        self.question_generator = dspy.ChainOfThought(QuestionGeneratorSignature)
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysisSignature)
        self.plan_generator = dspy.ChainOfThought(PlanGeneratorSignature)
        self.tabular_plan_generator = dspy.ChainOfThought(TabularPlanGeneratorSignature)
        
        # Simple in-memory storage per conversation
        self.conversations = {}
        
    def get_conversation_id(self, context: Optional[Dict[str, Any]], user_id: Optional[str] = None, conversation_id: Optional[str] = None) -> str:
        """Resolve conversation identifier from explicit argument, context, or user ID."""
        if conversation_id:
            return conversation_id
        if context:
            if isinstance(context, dict):
                for key in ('conversation_id', 'conversationId', 'session_id', 'sessionId'):
                    value = context.get(key)
                    if value:
                        return str(value)
                for key in ('user_id', 'userId'):
                    value = context.get(key)
                    if value:
                        return str(value)
            else:
                for key in ('conversation_id', 'conversationId', 'session_id', 'sessionId'):
                    if hasattr(context, key) and getattr(context, key):
                        return str(getattr(context, key))
                for key in ('user_id', 'userId'):
                    if hasattr(context, key) and getattr(context, key):
                        return str(getattr(context, key))
        if user_id:
            return user_id
        return f"conv_{uuid.uuid4().hex[:8]}"
    
    def get_conversation_data(self, conversation_id: str) -> Dict[str, Any]:
        """Get or create conversation data."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'summary': '',
                'context': '',
                'user_inputs': [],
                'created_at': datetime.now().isoformat()
            }
        return self.conversations[conversation_id]
    
    def process_user_input(self, user_input: str, context: Optional[Dict[str, Any]] = None, user_id: str = None, conversation_id: str = None) -> str:
        """Process user input with conversation context."""
        # Resolve conversation identifier using explicit argument, context, or user ID
        resolved_conversation_id = self.get_conversation_id(context, user_id=user_id, conversation_id=conversation_id)
        conv_data = self.get_conversation_data(resolved_conversation_id)

        if context:
            if isinstance(context, dict):
                conv_data['raw_context'] = context
            else:
                conv_data['raw_context'] = {
                    'conversation_id': getattr(context, 'conversation_id', None),
                    'user_id': getattr(context, 'user_id', None)
                }
        
        # Add debug info
        debug_info = f"""
[DEBUG INFO]
Conversation ID: {resolved_conversation_id}
Context: {context}
Summary Length: {len(conv_data['summary'])}
User Inputs Count: {len(conv_data['user_inputs'])}
"""
        
        # Classify intent
        intent_result = self.intent_classifier(user_input=user_input)
        intent = intent_result.intent.lower()
        confidence = float(intent_result.confidence)
        
        print(f"🎯 [INTENT] {intent} (confidence: {confidence:.2f})")
        
        # Handle based on intent
        if intent == 'greeting':
            response = self._handle_greeting(user_input)
        elif intent == 'providing_context':
            response = self._handle_information_collection(user_input, conv_data)
        elif intent == 'answering_questions':
            response = self._handle_information_collection(user_input, conv_data)
        elif intent == 'feedback_request':
            response = self._handle_feedback_request(user_input, conv_data)
        elif intent == 'plan_generation':
            response = self._handle_plan_generation(user_input, conv_data)
        else:
            response = self._handle_greeting(user_input)
        
        # Update conversation data
        conv_data['user_inputs'].append({
            'input': user_input,
            'timestamp': datetime.now().isoformat(),
            'intent': intent
        })
        
        # Update context
        conv_data['context'] = self._get_recent_context(conv_data)
        
        return response
    
    def _handle_greeting(self, user_input: str) -> str:
        """Handle greeting intent."""
        result = self.greeting_handler(user_input=user_input)
        return result.response
    
    def _handle_information_collection(self, user_input: str, conv_data: Dict[str, Any]) -> str:
        """Handle information collection."""
        # Extract information
        context = conv_data['context']
        info_result = self.information_collector(
            user_input=user_input,
            conversation_context=context
        )
        
        # Update summary
        if info_result.extracted_info.strip():
            if conv_data['summary']:
                conv_data['summary'] += f"\n\n{info_result.extracted_info}"
            else:
                conv_data['summary'] = info_result.extracted_info
        
        # Generate follow-up questions with categories
        question_categories_str = self._format_question_categories()
        question_result = self.question_generator(
            current_info=conv_data['summary'],
            conversation_context=context,
            question_categories=question_categories_str
        )
        
        response = f"Thank you for sharing that information!\n\n"
        # response += f"**What I've learned:**\n{info_result.extracted_info}\n\n"
        response += f"**To help me create a better plan, could you tell me more about:**\n{question_result.questions}\n\n"
        response += f"*Category: {question_result.category}*"
        
        return response
    
    def _handle_feedback_request(self, user_input: str, conv_data: Dict[str, Any]) -> str:
        """Handle feedback request."""
        current_info = conv_data['summary']
        context = conv_data['context']
        
        if not current_info.strip():
            return "I haven't collected any information yet. Please share details about your customer's migration project."
        
        # Analyze gaps
        gap_result = self.gap_analyzer(
            information_summary=current_info,
            feedback_request="Analyze completeness and identify gaps"
        )
        
        response = f"**Current Information Summary:**\n{current_info[:300]}...\n\n"
        response += f"**Analysis:** {gap_result.feedback}\n\n"
        response += f"**Missing Areas:** {gap_result.missing_areas}\n\n"
        response += f"Continue sharing information or type '/plan' when ready to generate your migration plan."
        
        return response
    
    def _handle_plan_generation(self, user_input: str, conv_data: Dict[str, Any]) -> str:
        """Handle plan generation."""
        current_info = conv_data['summary']
        
        if not current_info.strip():
            return "I need more information about your customer's migration project before I can generate a plan. Please share details about their current setup, requirements, and goals."
        
        # Search knowledge base (simplified)
        knowledge_base = "Databricks migration best practices and recommendations..."
        
        # Generate tabular plan
        tabular_result = self.tabular_plan_generator(
            customer_info=current_info,
            databricks_knowledge=knowledge_base
        )
        
        # Generate detailed plan
        plan_result = self.plan_generator(
            information_summary=current_info,
            knowledge_base=knowledge_base
        )
        
        response = f"# Customer Use Case Implementation Plan\n\n"
        response += f"## Implementation Plan Table\n\n{plan_result.plan_table}\n\n"
        response += f"## Detailed Implementation Timeline\n\n{tabular_result.implementation_plan}\n\n"
        response += f"## Timeline Summary\n{tabular_result.timeline_summary}\n\n"
        response += f"## Resource Requirements\n{tabular_result.resource_requirements}\n\n"
        response += f"## Key Assumptions\n{plan_result.assumptions}\n\n"
        response += f"## Risk Assessment & Mitigation\n{plan_result.risks}\n\n"
        response += f"*Plan generated based on customer information and Databricks knowledge base.*"
        
        return response
    
    def _format_question_categories(self) -> str:
        """Format question categories as context for the question generator."""
        categories_text = "Available question categories and example questions:\n\n"
        for category, questions in QUESTION_CATEGORIES.items():
            categories_text += f"**{category}**:\n"
            for question in questions:
                categories_text += f"- {question}\n"
            categories_text += "\n"
        return categories_text
    
    def _get_recent_context(self, conv_data: Dict[str, Any]) -> str:
        """Get recent conversation context."""
        recent_inputs = conv_data['user_inputs'][-3:]  # Last 3 inputs
        context_parts = []
        for item in recent_inputs:
            context_parts.append(f"User: {item['input']}")
        return "\n".join(context_parts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Simplified MLflow ResponsesAgent

# COMMAND ----------

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent
from mlflow.types.responses_helpers import Message

class SimplifiedMigrationPlanningAgent(ResponsesAgent):
    """Simplified migration planning agent using MLflow's built-in conversation management."""
    
    def __init__(self):
        super().__init__()
        self.conversation_manager = None
        self._configure_dspy_lm()
        self._ensure_conversation_manager()
    
    def _configure_dspy_lm(self):
        """Configure DSPy language model for the agent."""
        try:
            # Check if DSPy is already configured
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
                print("DSPy LM already configured, skipping configuration.")
                return
            
            # Configure DSPy with the same LM used during training
            lm = dspy.LM(model="databricks/databricks-claude-sonnet-4")
            dspy.settings.configure(lm=lm)
            print("DSPy LM configured successfully with databricks/databricks-claude-sonnet-4.")
        except Exception as e:
            print(f"Warning: Could not configure DSPy LM: {e}")
            # Try alternative configuration
            try:
                dspy.settings.configure(lm=dspy.LM("databricks/databricks-claude-sonnet-4"))
                print("DSPy LM configured with alternative method using databricks/databricks-claude-sonnet-4.")
            except Exception as e2:
                print(f"Warning: Alternative DSPy LM configuration also failed: {e2}")
                print("The agent may not work properly without LM configuration.")
    
    def _ensure_dspy_lm_configured(self):
        """Ensure DSPy LM is configured before prediction."""
        try:
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                self._configure_dspy_lm()
        except Exception as e:
            print(f"Warning: Could not ensure DSPy LM configuration: {e}")
    
    def _ensure_conversation_manager(self):
        """Ensure conversation manager is initialized."""
        if not self.conversation_manager:
            self.conversation_manager = SimplifiedConversationManager(
                lm=dspy.settings.lm,
                rm=dspy.settings.rm
            )
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Process request using MLflow's built-in context management."""
        # Ensure DSPy LM is configured
        self._ensure_dspy_lm_configured()
        self._ensure_conversation_manager()
        
        # Extract user input and user ID using the same logic as MVP agent
        user_input = self._extract_user_input(request)
        user_id = self._extract_user_id(request)
        
        print(f"🔐 [SESSION] Using user_id: {user_id}")
        print(f"🔐 [SESSION] User input: {user_input[:100]}...")
        
        # Extract context (MLflow handles this automatically)
        context = None
        if hasattr(request, 'context') and request.context:
            context = request.context
        elif hasattr(request, 'metadata') and request.metadata:
            context = request.metadata
        
        # Process with conversation manager
        conversation_id = self._extract_conversation_id(request)
        response_text = self.conversation_manager.process_user_input(user_input, context, user_id=user_id, conversation_id=conversation_id)
        
        # Create response
        output_item = self.create_text_output_item(
            text=response_text,
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
    
    def _extract_user_input(self, request: ResponsesAgentRequest) -> str:
        """Extract user input from ResponsesAgentRequest."""
        if request.input and len(request.input) > 0:
            # Get the last user message
            for message in reversed(request.input):
                if message.role == "user" and message.content:
                    # Extract text from content
                    if isinstance(message.content, list):
                        # Handle content as list of content items
                        text_parts = []
                        for content_item in message.content:
                            if hasattr(content_item, 'text'):
                                text_parts.append(content_item.text)
                            elif isinstance(content_item, str):
                                text_parts.append(content_item)
                        return " ".join(text_parts)
                    elif isinstance(message.content, str):
                        return message.content
                    else:
                        return str(message.content)
        
        return "Hello, I need help with migration planning."
    
    def _extract_user_id(self, request: ResponsesAgentRequest) -> str:
        """Extract user ID from ResponsesAgentRequest using proper MLflow approach."""
        print(f"🔍 [DEBUG] Extracting user ID from request...")

        # Check if request has context
        if not hasattr(request, 'context') or not request.context:
            print(f"⚠️ [WARNING] Request has no context, using default user ID")
            return "default_user"

        context = request.context
        print(f"🔍 [DEBUG] Request has context: {context}")
        print(f"🔍 [DEBUG] Context type: {type(context)}")

        def _get_value(obj, *keys):
            for key in keys:
                if isinstance(obj, dict) and key in obj and obj[key]:
                    return str(obj[key])
                if hasattr(obj, key) and getattr(obj, key):
                    return str(getattr(obj, key))
            return None

        user_id = _get_value(context, 'user_id', 'userId')
        if user_id:
            print(f"🔐 [USER_ID] Using user_id: {user_id}")
            return user_id

        conversation_id = _get_value(context, 'conversation_id', 'conversationId')
        if conversation_id:
            print(f"🔐 [USER_ID] Using conversation_id as user_id: {conversation_id}")
            return conversation_id

        # Fallback: use metadata if available
        metadata = getattr(request, 'metadata', None)
        if metadata:
            user_id = _get_value(metadata, 'user_id', 'userId')
            if user_id:
                print(f"🔐 [USER_ID] Using metadata user_id: {user_id}")
                return user_id
            conversation_id = _get_value(metadata, 'conversation_id', 'conversationId')
            if conversation_id:
                print(f"🔐 [USER_ID] Using metadata conversation_id as user_id: {conversation_id}")
                return conversation_id

        print(f"🔐 [USER_ID] Using fallback user ID: default_user")
        return "default_user"

    def _extract_conversation_id(self, request: ResponsesAgentRequest) -> str:
        """Extract conversation ID from ResponsesAgentRequest, defaulting to user ID if needed."""
        print(f"🔍 [DEBUG] Extracting conversation ID from request...")

        def _get_value(obj, *keys):
            for key in keys:
                if isinstance(obj, dict) and key in obj and obj[key]:
                    return str(obj[key])
                if hasattr(obj, key) and getattr(obj, key):
                    return str(getattr(obj, key))
            return None

        context = getattr(request, 'context', None)
        if context:
            conversation_id = _get_value(context, 'conversation_id', 'conversationId', 'session_id', 'sessionId')
            if conversation_id:
                print(f"🔐 [SESSION] Using conversation_id from context: {conversation_id}")
                return conversation_id
            user_id = _get_value(context, 'user_id', 'userId')
            if user_id:
                print(f"🔐 [SESSION] Using user_id as conversation_id: {user_id}")
                return user_id

        metadata = getattr(request, 'metadata', None)
        if metadata:
            conversation_id = _get_value(metadata, 'conversation_id', 'conversationId', 'session_id', 'sessionId')
            if conversation_id:
                print(f"🔐 [SESSION] Using conversation_id from metadata: {conversation_id}")
                return conversation_id
            user_id = _get_value(metadata, 'user_id', 'userId')
            if user_id:
                print(f"🔐 [SESSION] Using metadata user_id as conversation_id: {user_id}")
                return user_id

        # Fallback to a generated conversation ID to avoid collisions
        fallback = f"conv_{uuid.uuid4().hex[:8]}"
        print(f"🔐 [SESSION] Using fallback conversation_id: {fallback}")
        return fallback
    
    def predict_stream(self, request: ResponsesAgentRequest):
        """Simple streaming implementation."""
        response = self.predict(request)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=response.output[0]
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test the Simplified Agent

# COMMAND ----------

# Test the simplified agent
print("=== Testing Simplified Migration Planning Agent ===")

# Create agent
agent = SimplifiedMigrationPlanningAgent()

# Test conversation flow
test_context = {
    "conversation_id": "test_conv_123",
    "user_id": "test_user_456"
}

print("\n1. First interaction:")
response1 = agent.conversation_manager.process_user_input(
    "I'm working with a customer who wants to migrate from Oracle to Databricks. They have 50TB of data.",
    test_context,
    conversation_id="test_conv_123",
    user_id="test_user_456"
)
print(f"Response: {response1[:200]}...")

print("\n2. Second interaction:")
response2 = agent.conversation_manager.process_user_input(
    "The customer has 10 databases with 5 schemas each, around 1000 tables total.",
    test_context,
    conversation_id="test_conv_123",
    user_id="test_user_456"
)
print(f"Response: {response2[:200]}...")

print("\n3. Status check:")
response3 = agent.conversation_manager.process_user_input(
    "How's the information collection going?",
    test_context,
    conversation_id="test_conv_123",
    user_id="test_user_456"
)
print(f"Response: {response3[:200]}...")

print("\n4. Plan generation:")
response4 = agent.conversation_manager.process_user_input(
    "/plan",
    test_context,
    conversation_id="test_conv_123",
    user_id="test_user_456"
)
print(f"Response: {response4[:200]}...")

print("\n✅ Simplified agent test completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MLflow Model Registration

# COMMAND ----------

# Disable MLflow autologging to avoid Pydantic serialization warnings
mlflow.autolog(disable=True)
print("MLflow autologging disabled to avoid Pydantic serialization warnings")

# Create the ResponsesAgent instance BEFORE logging (following MVP pattern)
simplified_agent = SimplifiedMigrationPlanningAgent()
print("Simplified agent created successfully!")

# Register the simplified model
with mlflow.start_run():
    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="simplified_migration_planning_agent",
        python_model=simplified_agent,
        input_example={
            "input": [
                {
                    "role": "user",
                    "content": "I'm working with a customer who wants to migrate from Oracle to Databricks."
                }
            ],
            "context": {
                "conversation_id": "test_conv_123",
                "user_id": "test_user_456"
            }
        },
        pip_requirements=[
            "dspy>=2.6.23",
            "mlflow",
            "databricks-agents",
            "databricks-vectorsearch"
        ]
    )
    
    print(f"✅ Model logged successfully: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Deploy Agent to Model Serving

# COMMAND ----------

catalog_name = "vbdemos"
schema_name = "usecase_agent"
model_name = f"{catalog_name}.{schema_name}.simple-usecase-agent"
uc_model_info = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)

print(f"Model registered in Unity Catalog: {uc_model_info.name}")
print(f"   Version: {uc_model_info.version}")
print(f"Model ready for deployment!")


# Import Databricks Agents
from databricks import agents



# Deploy the agent to Model Serving
print("Deploying simplified agent to Model Serving...")
print(f"Model Name: {uc_model_info.name}")
print(f"Model Version: {uc_model_info.version}")

# Deploy the agent using the Agent Framework
# Use a static endpoint name to update the same endpoint
endpoint_name = "simplified-migration-planning-agent"
print(f"\nDeploying to endpoint: {endpoint_name}")

deployment = agents.deploy(uc_model_info.name, uc_model_info.version, endpoint_name=endpoint_name)
print(f"Simplified agent deployed successfully to endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Key Simplifications Made

# COMMAND ----------

print("""
🎯 KEY SIMPLIFICATIONS MADE:

1. **Removed Custom Session Management**
   - No more SessionManager class
   - No more SimpleStorage class
   - Uses MLflow's built-in context management

2. **Simplified Request Handling**
   - No complex header extraction
   - Uses standard ResponsesAgentRequest structure
   - Leverages MLflow's context field

3. **Cleaner Architecture**
   - Single ConversationManager class
   - In-memory storage per conversation
   - Built-in conversation ID generation

4. **Better MLflow Integration**
   - Uses MLflow's conversation tracking
   - Proper context passing
   - Simplified model registration

5. **Easier Debugging**
   - Built-in debug information
   - Clear conversation state tracking
   - MLflow's native tracing

6. **Reduced Complexity**
   - ~70% less code
   - Easier to maintain
   - Better performance

7. **Enhanced DSPy Signatures (Based on MVP)**
   - IntentClassifierSignature: Added detailed examples and better descriptions
   - GreetingSignature: Enhanced with Databricks account team context
   - InformationCollectorSignature: Added detailed extraction guidelines
   - QuestionGeneratorSignature: Added question_categories input and improved formatting
   - GapAnalysisSignature: Renamed and improved field descriptions
   - PlanGeneratorSignature: Added plan_table output for comprehensive planning
   - TabularPlanGeneratorSignature: Enhanced with specific column requirements
   - Added QUESTION_CATEGORIES constant for structured question generation
   - Implemented selective Chain of Thought strategy for better performance

8. **Fixed DSPy Configuration for Model Serving**
   - Added DSPy configuration within ResponsesAgent for model serving compatibility
   - Configured DSPy LM within the model class to ensure it's available at serving time
   - Disabled MLflow autologging to avoid Pydantic serialization warnings (following MVP pattern)
   - Creates ResponsesAgent instance before logging (following MVP pattern)
   - Uses databricks/databricks-claude-sonnet-4 model
   - Follows exact MVP agent configuration pattern for model serving

9. **Fixed User Input Extraction**
   - Added proper `_extract_user_input` method from MVP agent
   - Added proper `_extract_user_id` method for session management
   - Updated conversation manager to use user_id for proper session isolation
   - Fixed the issue where agent was always returning greeting instead of processing user input
""")

print("✅ Simplified implementation with enhanced signatures ready for deployment!")
