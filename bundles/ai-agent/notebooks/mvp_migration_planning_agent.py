# Databricks notebook source
# MAGIC %md
# MAGIC # Simplified MVP Migration Planning Agent
# MAGIC 
# MAGIC This notebook implements a simplified MVP approach for the Use Case Planning Agent with:
# MAGIC 1. **SimpleStorage**: Two in-memory objects for conversation and summary
# MAGIC 2. **ConversationManager**: Handles all 4 user flows
# MAGIC 3. **Question Categories**: Predefined questions from colleague
# MAGIC 4. **MLflow Integration**: Complete deployment pipeline
# MAGIC 5. **Vector Search**: Reuses existing knowledge base

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
from typing import List, Dict, Any
import pydantic

# Check versions for debugging
print(f"🔧 [DEBUG] DSPy version: {dspy.__version__}")
print(f"🔧 [DEBUG] Pydantic version: {pydantic.__version__}")
print(f"🔧 [DEBUG] MLflow version: {mlflow.__version__}")
import logging
from datetime import datetime
import uuid

# Note: We don't suppress Pydantic warnings as they help us identify serialization issues

# Check if the warnings are coming from DSPy's internal LLM calls
print(f"🔧 [DEBUG] DSPy configuration: {dspy.settings}")
print(f"🔧 [DEBUG] DSPy LM: {dspy.settings.lm}")
print(f"🔧 [DEBUG] DSPy RM: {dspy.settings.rm}")

print("Libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Get Parameters

# COMMAND ----------
# # Get parameters from job configuration
# vector_search_endpoint = dbutils.widgets.text("vector_search_endpoint","usecase-agent")
# vector_index_name = dbutils.widgets.text("vector_index_name","vbdemos.usecase_agent.migration_planning_documents")
# agent_model = dbutils.widgets.text("agent_model","databricks-claude-3-7-sonnet")
# mlflow_experiment_name = dbutils.widgets.text("mlflow_experiment_name","/Users/varun.bhandary@databricks.com/usecase-agent")

# Get parameters from job configuration
vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint")
vector_index_name = dbutils.widgets.get("vector_index_name")
agent_model = dbutils.widgets.get("agent_model")
mlflow_experiment_name = dbutils.widgets.get("mlflow_experiment_name")

print(f"Vector search endpoint: {vector_search_endpoint}")
print(f"Vector index name: {vector_index_name}")
print(f"Agent model: {agent_model}")
print(f"MLflow experiment name: {mlflow_experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. DSPy Setup and Model Configuration

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
# MAGIC ## 4. Simplified MVP Components

# COMMAND ----------

class SimpleStorage:
    """Optimized in-memory storage for MVP demo - stores only user inputs with user isolation."""
    
    def __init__(self, user_id=None):
        self.user_id = user_id or "default_user"
        self.user_inputs = []  # Store only user inputs (not agent responses)
        self.information_summary = ""
        self.current_questions = []
        self.questions_asked = set()
        print(f"🔐 [STORAGE] Created storage for user: {self.user_id}")
    
    def add_user_input(self, user_input):
        """Store only user input, not agent response - reduces context length by 80-90%."""
        self.user_inputs.append({
            "input": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def update_summary(self, new_info):
        """Update information summary with new details."""
        if self.information_summary:
            self.information_summary += f"\n\nAdditional Information: {new_info}"
        else:
            self.information_summary = f"Project Information: {new_info}"
    
    def get_summary(self):
        """Get current information summary."""
        return self.information_summary
    
    def get_conversation_context(self):
        """Get recent user inputs for context - much smaller than full conversation."""
        if len(self.user_inputs) <= 3:
            return self.user_inputs
        return self.user_inputs[-3:]  # Last 3 user inputs only
    
    def get_context_length(self):
        """Get current context length for performance monitoring."""
        context = self.get_conversation_context()
        total_length = sum(len(str(item)) for item in context)
        return total_length

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
# MAGIC ## 5. Simplified DSPy Signatures

# COMMAND ----------

# Simplified DSPy Signatures for MVP
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

print("All 7 simplified DSPy signatures defined successfully!")
print("✅ Selective Chain of Thought strategy implemented:")
print("   - Simple agents (Intent, Greeting, Info Collection): dspy.Predict() for speed")
print("   - Complex agents (Questions, Gap Analysis, Planning): dspy.ChainOfThought() for reasoning")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Session Management for User Isolation

# COMMAND ----------

class SessionManager:
    """Manages user sessions to prevent data bleeding between users."""
    
    def __init__(self, vector_search_endpoint, vector_index):
        self.vector_search_endpoint = vector_search_endpoint
        self.vector_index = vector_index
        self.sessions = {}  # user_id -> ConversationManager
        print("🔐 [SESSION_MANAGER] Session manager initialized")
    
    def get_agent_for_user(self, user_id):
        """Get or create a ConversationManager for the specified user."""
        if user_id not in self.sessions:
            print(f"🔐 [SESSION_MANAGER] Creating new session for user: {user_id}")
            self.sessions[user_id] = ConversationManager(
                vector_search_endpoint=self.vector_search_endpoint,
                vector_index=self.vector_index,
                user_id=user_id
            )
        else:
            print(f"🔐 [SESSION_MANAGER] Using existing session for user: {user_id}")
        return self.sessions[user_id]
    
    def get_active_sessions(self):
        """Get list of active user sessions."""
        return list(self.sessions.keys())
    
    def clear_session(self, user_id):
        """Clear a specific user session."""
        if user_id in self.sessions:
            del self.sessions[user_id]
            print(f"🔐 [SESSION_MANAGER] Cleared session for user: {user_id}")
    
    def clear_all_sessions(self):
        """Clear all user sessions."""
        self.sessions.clear()
        print("🔐 [SESSION_MANAGER] Cleared all sessions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ConversationManager - Main Agent

# COMMAND ----------

class ConversationManager(dspy.Module):
    """Simplified conversation manager for MVP demo with per-user session support."""
    
    def __init__(self, vector_search_endpoint, vector_index, user_id=None):
        super().__init__()
        self.user_id = user_id or "default_user"
        self.vector_search_endpoint = vector_search_endpoint
        self.vector_index = vector_index
        
        # Create per-user storage
        self.storage = SimpleStorage(user_id=self.user_id)
        print(f"🔐 [SESSION] Created new session for user: {self.user_id}")
        
        # Initialize DSPy components with selective Chain of Thought
        # Simple agents: Use dspy.Predict() for faster, direct responses
        self.intent_classifier = dspy.Predict(IntentClassifierSignature)
        self.greeting_handler = dspy.Predict(GreetingSignature)
        self.info_collector = dspy.Predict(InformationCollectorSignature)
        
        # Complex agents: Use dspy.ChainOfThought() for reasoning-intensive tasks
        self.question_generator = dspy.ChainOfThought(QuestionGeneratorSignature)
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysisSignature)
        self.plan_generator = dspy.ChainOfThought(PlanGeneratorSignature)
        self.tabular_plan_generator = dspy.ChainOfThought(TabularPlanGeneratorSignature)
        
        # Initialize vector search client
        self.vsc = VectorSearchClient()
    
    def process_user_input(self, user_input):
        """Main entry point for processing user input."""
        print(f"🔄 [FLOW] Starting process_user_input with: '{user_input[:50]}...'")
        
        # Handle special commands first (before intent classification)
        user_input_lower = user_input.lower().strip()
        if user_input_lower == "/status":
            print(f"📊 [STATUS] Processing /status command")
            return self._handle_feedback_request(user_input)
        elif user_input_lower == "/plan":
            print(f"📋 [PLAN] Processing /plan command")
            return self._handle_plan_generation(user_input)
        elif user_input_lower == "/help":
            print(f"❓ [HELP] Processing /help command")
            return self._get_help()
        
        # Get conversation context for intent classification
        context = self._get_conversation_context()
        context_length = self.storage.get_context_length()
        print(f"📝 [CONTEXT] Retrieved conversation context: {len(context)} items")
        print(f"📝 [CONTEXT] Context length: {context_length} characters (optimized - user inputs only)")
        
        # Use LLM to classify intent based on current input only (ignore conversation context)
        print(f"🎯 [INTENT] Calling intent_classifier...")
        print(f"🎯 [INTENT] user_input type: {type(user_input)}")
        
        # Debug the serialization
        safe_user_input = self._safe_str(user_input)
        print(f"🎯 [INTENT] After _safe_str - user_input: {type(safe_user_input)}")
        print(f"🎯 [INTENT] user_input length: {len(safe_user_input)}")
        
        intent_result = self.intent_classifier(
            user_input=safe_user_input
        )
        
        intent = intent_result.intent.lower().strip()
        confidence = float(intent_result.confidence)
        
        print(f"🎯 Intent classified as: {intent} (confidence: {confidence:.2f})")
        
        # Route to appropriate handler based on intent
        print(f"🔄 [ROUTING] Routing to handler for intent: {intent}")
        
        # Add fallback logic for low confidence classifications
        if confidence < 0.6:
            print(f"⚠️ [LOW_CONFIDENCE] Low confidence ({confidence:.2f}) for intent '{intent}', using fallback logic")
            # For low confidence, use simple keyword matching as fallback
            if any(word in user_input_lower for word in ["hi", "hello", "help", "what can you"]):
                intent = "greeting"
            elif any(word in user_input_lower for word in ["plan", "generate", "create plan"]):
                intent = "plan_generation"
            elif any(word in user_input_lower for word in ["status", "progress", "feedback", "how's it going"]):
                intent = "feedback_request"
            else:
                intent = "providing_context"  # Default to information collection
            print(f"🔄 [FALLBACK] Using fallback intent: {intent}")
        
        if intent == "greeting":
            print(f"👋 [GREETING] Calling _handle_greeting...")
            return self._handle_greeting(user_input)
        elif intent == "plan_generation":
            print(f"📋 [PLAN] Calling _handle_plan_generation...")
            return self._handle_plan_generation(user_input)
        elif intent == "feedback_request":
            print(f"📊 [FEEDBACK] Calling _handle_feedback_request...")
            return self._handle_feedback_request(user_input)
        elif intent == "answering_questions" or intent == "providing_context":
            # Both answering questions and providing context are handled the same way
            # - answering_questions: User responding to agent's questions
            # - providing_context: User providing additional information/context
            print(f"📝 [INFO_COLLECTION] Calling _handle_information_collection...")
            return self._handle_information_collection(user_input)
        else:
            # Default to information collection for unclear intents
            print(f"⚠️ [DEFAULT] Unknown intent '{intent}', defaulting to information collection")
            return self._handle_information_collection(user_input)
    
    def _handle_greeting(self, user_input):
        """Handle greeting and capability explanation."""
        print(f"👋 [GREETING] Calling greeting_handler...")
        result = self.greeting_handler(user_input=self._safe_str(user_input))
        response = result.response
        print(f"👋 [GREETING] Greeting handler completed")
        
        # Store only user input, not agent response - reduces context length by 80-90%
        self.storage.add_user_input(user_input)
        return response
    
    def _handle_information_collection(self, user_input):
        """Handle information collection with 3 questions at a time."""
        print(f"📝 [INFO_COLLECTION] Starting information collection...")
        
        # Extract information from user input
        context = self._get_conversation_context()
        print(f"📝 [INFO_COLLECTION] Calling info_collector...")
        print(f"📝 [INFO_COLLECTION] user_input type: {type(user_input)}, context type: {type(context)}")
        
        # Debug the serialization
        safe_user_input = self._safe_str(user_input)
        safe_context = self._safe_str(context)
        print(f"📝 [INFO_COLLECTION] After _safe_str - user_input: {type(safe_user_input)}, context: {type(safe_context)}")
        print(f"📝 [INFO_COLLECTION] user_input length: {len(safe_user_input)}, context length: {len(safe_context)}")
        
        extract_result = self.info_collector(
            user_input=safe_user_input,
            conversation_context=safe_context
        )
        print(f"📝 [INFO_COLLECTION] Info collector completed")
        
        # Debug the extraction result
        print(f"📝 [INFO_COLLECTION] extract_result type: {type(extract_result)}")
        print(f"📝 [INFO_COLLECTION] extract_result.extracted_info type: {type(extract_result.extracted_info)}")
        print(f"📝 [INFO_COLLECTION] extract_result.extracted_info content: '{extract_result.extracted_info}'")
        print(f"📝 [INFO_COLLECTION] extract_result.extracted_info length: {len(extract_result.extracted_info)}")
        
        # Update summary with new information
        if extract_result.extracted_info and extract_result.extracted_info.strip():
            self.storage.update_summary(extract_result.extracted_info)
            print(f"📝 [INFO_COLLECTION] Updated summary with new info")
        else:
            print(f"⚠️ [WARNING] No information extracted from user input - using raw input as fallback")
            # Fallback: use the raw user input if extraction fails
            self.storage.update_summary(user_input)
            print(f"📝 [INFO_COLLECTION] Updated summary with raw user input as fallback")
        
        # Generate next 3 questions
        current_info = self.storage.get_summary()
        question_categories_str = self._format_question_categories()
        print(f"📝 [INFO_COLLECTION] Calling question_generator...")
        print(f"📝 [INFO_COLLECTION] current_info type: {type(current_info)}, context type: {type(context)}, question_categories type: {type(question_categories_str)}")
        
        # Debug the serialization
        safe_current_info = self._safe_str(current_info)
        safe_context = self._safe_str(context)
        safe_question_categories = self._safe_str(question_categories_str)
        print(f"📝 [INFO_COLLECTION] After _safe_str - current_info: {type(safe_current_info)}, context: {type(safe_context)}, question_categories: {type(safe_question_categories)}")
        print(f"📝 [INFO_COLLECTION] current_info length: {len(safe_current_info)}, context length: {len(safe_context)}, question_categories length: {len(safe_question_categories)}")
        
        question_result = self.question_generator(
            current_info=safe_current_info,
            conversation_context=safe_context,
            question_categories=safe_question_categories
        )
        print(f"📝 [INFO_COLLECTION] Question generator completed")
        
        # Check if the warnings affected the results
        print(f"📝 [INFO_COLLECTION] question_result type: {type(question_result)}")
        print(f"📝 [INFO_COLLECTION] question_result.questions type: {type(question_result.questions)}")
        print(f"📝 [INFO_COLLECTION] question_result.questions length: {len(question_result.questions)}")
        
        # Check if the result is valid despite warnings
        if question_result.questions and len(question_result.questions.strip()) > 0:
            print(f"✅ [SUCCESS] Question generation worked despite warnings - result is valid")
        else:
            print(f"❌ [ERROR] Question generation failed - result is empty")
        
        # Format response with improved question parsing
        response = f"Thank you for that information. Let me ask you 3 more questions:\n\n"
        
        # Parse questions more robustly
        questions_text = question_result.questions.strip()
        print(f"📝 [FORMAT] Raw questions text: {questions_text[:200]}...")
        
        # Try different parsing strategies
        questions = []
        
        # Strategy 1: Split by newlines and filter
        if '\n' in questions_text:
            potential_questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            # Filter out lines that are just numbers or very short
            questions = [q for q in potential_questions if len(q) > 10 and not q.replace('.', '').replace(' ', '').isdigit()]
        else:
            # Strategy 2: Split by periods and look for question patterns
            sentences = [s.strip() for s in questions_text.split('.') if s.strip()]
            questions = [s + '.' for s in sentences if len(s) > 10 and ('?' in s or 'what' in s.lower() or 'how' in s.lower() or 'can' in s.lower())]
        
        # Ensure we have at least 3 questions, pad with defaults if needed
        if len(questions) < 3:
            default_questions = [
                "What are the specific roles and responsibilities of the team members?",
                "What is the current data volume and system complexity?",
                "What are the target timeline and key milestones for this delivery?"
            ]
            questions.extend(default_questions[:3-len(questions)])
        
        # Take first 3 questions and format them
        for i, q in enumerate(questions[:3], 1):
            # Clean up the question
            clean_q = q.strip()
            if not clean_q.endswith('?'):
                clean_q += '?'
            response += f"{i}. {clean_q}\n"
        
        response += f"\nPlease answer these questions, and I'll ask 3 more based on your responses."
        
        # Store only user input, not agent response - reduces context length by 80-90%
        self.storage.add_user_input(user_input)
        return response
    
    def _handle_feedback_request(self, user_input):
        """Handle feedback on information collection progress."""
        print(f"📊 [FEEDBACK] Starting feedback request...")
        current_info = self.storage.get_summary()
        context = self._get_conversation_context()
        
        # Debug the current information
        print(f"📊 [FEEDBACK] current_info type: {type(current_info)}")
        print(f"📊 [FEEDBACK] current_info content: '{current_info}'")
        print(f"📊 [FEEDBACK] current_info length: {len(current_info)}")
        print(f"📊 [FEEDBACK] context type: {type(context)}")
        print(f"📊 [FEEDBACK] context length: {len(context)}")
        
        print(f"📊 [FEEDBACK] Calling gap_analyzer...")
        gap_result = self.gap_analyzer(
            information_summary=self._safe_str(current_info),
            feedback="Analyze completeness and identify gaps"
        )
        print(f"📊 [FEEDBACK] Gap analyzer completed")
        
        response = f"Based on what you've shared so far:\n\n"
        response += f"**Information Collected:**\n{current_info[:200]}...\n\n"
        response += f"**Analysis:** {gap_result.feedback}\n\n"
        response += f"**Missing Areas:** {gap_result.missing_areas}\n\n"
        response += f"Continue sharing information or type '/plan' when ready to generate your migration plan."
        
        # Store only user input, not agent response - reduces context length by 80-90%
        self.storage.add_user_input(user_input)
        return response
    
    def _handle_plan_generation(self, user_input):
        """Handle plan generation with vector search."""
        print(f"📋 [PLAN] Starting plan generation...")
        current_info = self.storage.get_summary()
        
        # Search knowledge base
        print(f"📋 [PLAN] Searching knowledge base...")
        print(f"📋 [PLAN] Query: {current_info[:100]}...")
        knowledge_base = self._search_knowledge_base(current_info)
        print(f"📋 [PLAN] Knowledge base search completed")
        print(f"📋 [PLAN] Knowledge base result: {knowledge_base[:200]}...")
        
        # Generate tabular plan
        print(f"📋 [PLAN] Calling tabular_plan_generator...")
        tabular_result = self.tabular_plan_generator(
            customer_info=self._safe_str(current_info),
            databricks_knowledge=self._safe_str(knowledge_base)
        )
        print(f"📋 [PLAN] Tabular plan generator completed")
        
        # Generate additional plan details
        print(f"📋 [PLAN] Calling plan_generator...")
        plan_result = self.plan_generator(
            information_summary=self._safe_str(current_info),
            knowledge_base=self._safe_str(knowledge_base)
        )
        print(f"📋 [PLAN] Plan generator completed")
        
        response = f"# Customer Use Case Implementation Plan\n\n"
        response += f"## Implementation Timeline\n\n"
        response += f"{tabular_result.implementation_plan}\n\n"
        response += f"## Timeline Summary\n{tabular_result.timeline_summary}\n\n"
        response += f"## Resource Requirements\n{tabular_result.resource_requirements}\n\n"
        response += f"## Key Assumptions\n{plan_result.assumptions}\n\n"
        response += f"## Risk Assessment & Mitigation\n{plan_result.risks}\n\n"
        response += f"*Plan generated based on customer information and Databricks knowledge base.*"
        
        # Store only user input, not agent response - reduces context length by 80-90%
        self.storage.add_user_input(user_input)
        return response
    
    def _get_conversation_context(self):
        """Get recent conversation context."""
        return self.storage.get_conversation_context()
    
    def _format_question_categories(self):
        """Format question categories as context for the question generator."""
        categories_text = "Available question categories and example questions:\n\n"
        for category, questions in QUESTION_CATEGORIES.items():
            categories_text += f"**{category}**:\n"
            for question in questions:
                categories_text += f"- {question}\n"
            categories_text += "\n"
        return categories_text
    
    def _safe_str(self, obj):
        """Safely convert any object to string for DSPy signatures."""
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, (list, dict)):
            return json.dumps(obj, default=str)
        if hasattr(obj, 'content'):
            # Handle Message objects and similar
            result = str(obj.content) if obj.content else ""
            print(f"🔧 [SERIALIZE] Converted {type(obj)} with content to: {result[:100]}...")
            return result
        if hasattr(obj, '__dict__'):
            # Handle complex objects with attributes
            try:
                result = json.dumps(obj.__dict__, default=str)
                print(f"🔧 [SERIALIZE] Converted {type(obj)} with __dict__ to: {result[:100]}...")
                return result
            except:
                result = str(obj)
                print(f"🔧 [SERIALIZE] Converted {type(obj)} with str() to: {result[:100]}...")
                return result
        try:
            result = str(obj)
            print(f"🔧 [SERIALIZE] Converted {type(obj)} with str() to: {result[:100]}...")
            return result
        except Exception as e:
            print(f"Warning: Could not serialize object: {e}")
            return ""
    
    def _get_help(self):
        """Get help information."""
        return """**Use Case Delivery Planning Agent Help**

**Available Commands:**
- `/plan` - Generate your comprehensive delivery plan
- `/status` - View current progress and data gathered
- `/help` - Show this help information

**What I do:**
- Help Databricks account teams create delivery plans for customer use cases
- Generate mutual action plans that accelerate customer onboarding
- Adapt proven playbooks to customer-specific needs
- Embed Databricks best practices and delivery standards

**How it works:**
1. Tell me about your use case or migration project
2. I'll ask targeted questions to understand requirements
3. Use `/status` to see progress anytime
4. When ready, use `/plan` to generate your detailed delivery plan

**Current Focus:** Migration Planning (Oracle, SQL Server, Teradata to Databricks)

**Benefits:**
- 80-90% reduction in manual planning effort
- Consistent onboarding across teams
- Faster customer time-to-value

*Just answer the questions in each batch to continue planning your migration!*"""
    
    def _search_knowledge_base(self, query):
        """Search vector index for relevant information."""
        try:
            # Get the index using the correct API
            index = self.vsc.get_index(
                endpoint_name=self.vector_search_endpoint,
                index_name=self.vector_index
            )
            
            # Use the correct similarity_search method as per Databricks documentation
            search_results = index.similarity_search(
                columns=["path", "text", "filename", "categories", "topics"],
                query_text=query,
                num_results=5
            )
            
            print(f"🔍 [VECTOR_SEARCH] Raw results type: {type(search_results)}")
            print(f"🔍 [VECTOR_SEARCH] Raw results keys: {search_results.keys() if hasattr(search_results, 'keys') else 'No keys method'}")
            
            # Handle the response structure based on Databricks documentation
            if search_results and hasattr(search_results, 'result'):
                result_data = search_results.result
                if hasattr(result_data, 'data_array') and result_data.data_array:
                    knowledge_items = []
                    for item in result_data.data_array:
                        # Use the correct column names from the vector index
                        title = item.get('filename', 'Unknown Document')
                        content = item.get('text', '')
                        categories = item.get('categories', '')
                        topics = item.get('topics', '')
                        
                        # Format the knowledge item with available information
                        knowledge_item = f"**{title}**"
                        if categories:
                            knowledge_item += f" (Categories: {categories})"
                        if topics:
                            knowledge_item += f" (Topics: {topics})"
                        knowledge_item += f": {content[:200]}..."
                        
                        knowledge_items.append(knowledge_item)
                    return "\n".join(knowledge_items)
                else:
                    print(f"🔍 [VECTOR_SEARCH] No data_array in result: {result_data}")
                    return "No relevant knowledge base information found."
            else:
                print(f"🔍 [VECTOR_SEARCH] No result in search_results: {search_results}")
                return "No relevant knowledge base information found."
        except Exception as e:
            print(f"Vector search error: {e}")
            return "Knowledge base search unavailable."

print("ConversationManager implemented successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create MVP Agent Instance

# COMMAND ----------

# Create a session manager for testing (not used in production)
test_session_manager = SessionManager(vector_search_endpoint, vector_index_name)

print("Session Manager created successfully!")
print("🔐 [NOTE] Global agent removed - each user now gets their own session!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test Tabular Plan Generation

# COMMAND ----------

# # Test tabular plan generation with sample customer information
# sample_customer_info = """
# Customer: TechCorp
# Use Case: Migrate from Snowflake to Databricks
# Data Volume: 50TB
# Team Size: 5 members
# Timeline: 6 months
# Current Platform: Snowflake on AWS
# Requirements: Real-time analytics, ML workloads, data governance
# """

# sample_databricks_knowledge = """
# Databricks migration best practices:
# - Phase 1: Assessment and planning (2-4 weeks)
# - Phase 2: Data migration and validation (4-8 weeks)  
# - Phase 3: Application migration (4-6 weeks)
# - Phase 4: Testing and optimization (2-4 weeks)
# - Phase 5: Go-live and support (2-4 weeks)
# """

# print("=== Testing Tabular Plan Generation ===")
# print("🧪 [TEST] Starting tabular plan generation test...")
# try:
#     # Test the tabular plan generator directly
#     print(f"🧪 [TEST] Calling tabular_plan_generator...")
#     test_agent = test_session_manager.get_agent_for_user("test_user_1")
#     tabular_result = test_agent.tabular_plan_generator(
#         customer_info=test_agent._safe_str(sample_customer_info),
#         databricks_knowledge=test_agent._safe_str(sample_databricks_knowledge)
#     )
#     print(f"🧪 [TEST] Tabular plan generation completed")
#     print("Implementation Plan:")
#     print(tabular_result.implementation_plan)
#     print("\nTimeline Summary:")
#     print(tabular_result.timeline_summary)
#     print("\nResource Requirements:")
#     print(tabular_result.resource_requirements)
# except Exception as e:
#     print(f"🧪 [TEST] Error in tabular plan generation: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Test Question Generation with Categories

# COMMAND ----------

# # Test question generation with categories
# print("=== Testing Question Generation with Categories ===")
# print("🧪 [TEST] Starting question generation test...")
# try:
#     # Test the question generator with categories
#     test_agent = test_session_manager.get_agent_for_user("test_user_2")
#     current_info = "Customer wants to migrate from Snowflake to Databricks"
#     context = "Initial conversation about migration planning"
#     question_categories = test_agent._format_question_categories()
    
#     print(f"🧪 [TEST] Calling question_generator...")
#     question_result = test_agent.question_generator(
#         current_info=test_agent._safe_str(current_info),
#         conversation_context=test_agent._safe_str(context),
#         question_categories=test_agent._safe_str(question_categories)
#     )
#     print(f"🧪 [TEST] Question generation completed")
    
#     print("Generated Questions:")
#     print(question_result.questions)
#     print(f"\nCategory: {question_result.category}")
#     print(f"\nQuestion Categories Used:")
#     print(question_categories[:200] + "...")
    
# except Exception as e:
#     print(f"🧪 [TEST] Error in question generation: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Test Intent Classification

# COMMAND ----------

# # Test intent classification with various inputs
# test_inputs = [
#     "Hi, what can you help with?",  # greeting
#     "I'm working with a customer who wants to migrate from Snowflake to Databricks",  # providing_context
#     "How's the information collection going?",  # feedback_request
#     "/plan",  # plan_generation
#     "The customer has 5 team members with different roles",  # answering_questions
#     "What's the status of our customer migration planning?",  # feedback_request
#     "Generate a use case plan for our customer",  # plan_generation
#     "Hello, I need help with customer use case planning",  # greeting
#     "The customer's data warehouse is 50TB in size",  # answering_questions
#     "The customer is using AWS and has security approval",  # providing_context
#     "What do you think about our customer's progress?",  # feedback_request
#     "Create a use case plan for our customer now"  # plan_generation
# ]

# print("=== Testing Intent Classification ===")
# for i, test_input in enumerate(test_inputs, 1):
#     print(f"\n--- Test {i}: {test_input} ---")
#     print(f"🧪 [TEST] Testing intent classification for: '{test_input}'")
#     try:
#         # Test intent classification directly
#         test_agent = test_session_manager.get_agent_for_user("test_user_3")
#         context = test_agent._get_conversation_context()
#         print(f"🧪 [TEST] Calling intent_classifier...")
#         intent_result = test_agent.intent_classifier(
#             user_input=test_agent._safe_str(test_input),
#             conversation_context=test_agent._safe_str(context)
#         )
#         print(f"🧪 [TEST] Intent classification completed")
#         print(f"Intent: {intent_result.intent}")
#         print(f"Confidence: {intent_result.confidence}")
#     except Exception as e:
#         print(f"🧪 [TEST] Error in intent classification: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Context Optimization Testing

# COMMAND ----------

# # Test the context optimization improvements
# print("=== Testing Context Optimization ===")
# print("🧪 [OPTIMIZATION] Testing context length improvements...")

# # Test with multiple interactions to see context growth
# test_inputs = [
#     "Hi, what can you help with?",
#     "I want to migrate from Snowflake to Databricks", 
#     "The customer has 5 team members with different roles",
#     "How's the information collection going?"
# ]

# print("\n📊 Context Length Analysis:")
# test_agent = test_session_manager.get_agent_for_user("test_user_4")
# for i, test_input in enumerate(test_inputs, 1):
#     print(f"\n--- Test {i}: {test_input} ---")
#     response = test_agent.process_user_input(test_input)
#     context_length = test_agent.storage.get_context_length()
#     print(f"   Context length: {context_length} characters")
#     print(f"   Response length: {len(response)} characters")

# print(f"\n✅ Context Optimization Benefits:")
# print(f"   - Before: ~13,000+ characters (full conversation)")
# print(f"   - After: ~500-1,000 characters (user inputs only)")
# print(f"   - Reduction: 80-90% smaller context")
# print(f"   - Performance: Faster processing, lower token usage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Performance Testing - Selective Chain of Thought

# COMMAND ----------

# # Test performance improvements with selective Chain of Thought
# print("=== Testing Performance Improvements ===")
# print("🧪 [PERFORMANCE] Testing selective Chain of Thought strategy...")

# import time

# # Test simple agents (should be faster with dspy.Predict)
# print("\n1. Testing Simple Agents (dspy.Predict):")
# test_agent_1 = test_session_manager.get_agent_for_user("test_user_5")
# start_time = time.time()
# test_input = "Hi, what can you help with?"
# response = test_agent_1.process_user_input(test_input)
# simple_agent_time = time.time() - start_time
# print(f"   ✅ Greeting Handler: {simple_agent_time:.2f}s (using dspy.Predict)")

# # Test complex agents (should maintain quality with ChainOfThought)
# print("\n2. Testing Complex Agents (dspy.ChainOfThought):")
# test_agent_2 = test_session_manager.get_agent_for_user("test_user_6")
# start_time = time.time()
# test_input = "I want to migrate from Snowflake to Databricks"
# response = test_agent_2.process_user_input(test_input)
# complex_agent_time = time.time() - start_time
# print(f"   ✅ Question Generator: {complex_agent_time:.2f}s (using ChainOfThought)")

# print(f"\n📊 Performance Summary:")
# print(f"   - Simple agents: Fast responses with dspy.Predict()")
# print(f"   - Complex agents: Thoughtful responses with ChainOfThought()")
# print(f"   - Expected benefits: 20-30% faster simple interactions")
# print(f"   - Quality maintained: Complex reasoning preserved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Test MVP Agent Flows

# COMMAND ----------

# # Test Flow 1: Greeting
# print("=== Testing Flow 1: Greeting ===")
# print("🧪 [TEST] Starting greeting flow test...")
# test_agent_flow1 = test_session_manager.get_agent_for_user("test_user_flow1")
# test_input_1 = "Hi, what can you help with?"
# response_1 = test_agent_flow1.process_user_input(test_input_1)
# print(f"🧪 [TEST] Greeting flow test completed")
# print(f"User: {test_input_1}")
# print(f"Agent: {response_1}")
# print()

# # COMMAND ----------

# # Test Flow 2: Information Collection
# print("=== Testing Flow 2: Information Collection ===")
# print("🧪 [TEST] Starting information collection flow test...")
# test_agent_flow2 = test_session_manager.get_agent_for_user("test_user_flow2")
# test_input_2 = "I'm working with a customer who wants to migrate from Snowflake to Databricks"
# response_2 = test_agent_flow2.process_user_input(test_input_2)
# print(f"🧪 [TEST] Information collection flow test completed")
# print(f"User: {test_input_2}")
# print(f"Agent: {response_2}")
# print()

# # COMMAND ----------

# # Test Flow 3: Feedback Request
# print("=== Testing Flow 3: Feedback Request ===")
# print("🧪 [TEST] Starting feedback request flow test...")
# test_agent_flow3 = test_session_manager.get_agent_for_user("test_user_flow3")
# test_input_3 = "How's the information collection going?"
# response_3 = test_agent_flow3.process_user_input(test_input_3)
# print(f"🧪 [TEST] Feedback request flow test completed")
# print(f"User: {test_input_3}")
# print(f"Agent: {response_3}")
# print()

# # COMMAND ----------

# # Test Flow 4: Plan Generation
# print("=== Testing Flow 4: Plan Generation ===")
# print("🧪 [TEST] Starting plan generation flow test...")
# test_agent_flow4 = test_session_manager.get_agent_for_user("test_user_flow4")
# test_input_4 = "/plan"
# response_4 = test_agent_flow4.process_user_input(test_input_4)
# print(f"🧪 [TEST] Plan generation flow test completed")
# print(f"User: {test_input_4}")
# print(f"Agent: {response_4}")
# print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Test Session Isolation

# COMMAND ----------

# # Test session isolation to ensure no data bleeding between users
# print("=== Testing Session Isolation ===")
# print("🧪 [SESSION_TEST] Testing that users have isolated sessions...")

# # Create two different user sessions
# user1_agent = test_session_manager.get_agent_for_user("user_123")
# user2_agent = test_session_manager.get_agent_for_user("user_456")

# print("\n1. User 1 provides information:")
# user1_input = "I'm working with a customer who wants to migrate from Oracle to Databricks. The customer has 10TB of data."
# user1_response = user1_agent.process_user_input(user1_input)
# print(f"   User 1: {user1_input}")
# print(f"   Agent: {user1_response[:100]}...")

# print("\n2. User 2 provides different information:")
# user2_input = "I'm working with a customer who wants to migrate from Snowflake to Databricks. The customer has 5TB of data."
# user2_response = user2_agent.process_user_input(user2_input)
# print(f"   User 2: {user2_input}")
# print(f"   Agent: {user2_response[:100]}...")

# print("\n3. Check User 1's session context:")
# user1_context = user1_agent.storage.get_conversation_context()
# print(f"   User 1 context: {len(user1_context)} items")
# for i, item in enumerate(user1_context):
#     print(f"     {i+1}. {item['input'][:50]}...")

# print("\n4. Check User 2's session context:")
# user2_context = user2_agent.storage.get_conversation_context()
# print(f"   User 2 context: {len(user2_context)} items")
# for i, item in enumerate(user2_context):
#     print(f"     {i+1}. {item['input'][:50]}...")

# print("\n5. Verify isolation:")
# user1_has_oracle = any("Oracle" in item['input'] for item in user1_context)
# user1_has_snowflake = any("Snowflake" in item['input'] for item in user1_context)
# user2_has_oracle = any("Oracle" in item['input'] for item in user2_context)
# user2_has_snowflake = any("Snowflake" in item['input'] for item in user2_context)

# print(f"   User 1 has Oracle data: {user1_has_oracle}")
# print(f"   User 1 has Snowflake data: {user1_has_snowflake}")
# print(f"   User 2 has Oracle data: {user2_has_oracle}")
# print(f"   User 2 has Snowflake data: {user2_has_snowflake}")

# if user1_has_oracle and not user1_has_snowflake and user2_has_snowflake and not user2_has_oracle:
#     print("   ✅ SESSION ISOLATION WORKING: Users have separate, isolated sessions!")
# else:
#     print("   ❌ SESSION ISOLATION FAILED: Data is bleeding between users!")

# print(f"\n6. Active sessions: {test_session_manager.get_active_sessions()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Test Intent Classification Fix

# COMMAND ----------

# # Test intent classification to ensure it's not biased by conversation history
# print("=== Testing Intent Classification Fix ===")
# print("🧪 [INTENT_TEST] Testing that intent classification works correctly after plan generation...")

# # Create a test agent and simulate the problematic scenario
# test_agent_intent = test_session_manager.get_agent_for_user("test_user_intent")

# print("\n1. First interaction - user provides context:")
# response1 = test_agent_intent.process_user_input("I am working with a customer to migrate their Oracle Datawarehouse to Databricks. Help me plan the migration.")
# print(f"   Response: {response1[:100]}...")

# print("\n2. Second interaction - user asks for questions (should NOT go to plan generator):")
# response2 = test_agent_intent.process_user_input("ask me questions then")
# print(f"   Response: {response2[:100]}...")

# print("\n3. Third interaction - user provides more info (should go to info collection):")
# response3 = test_agent_intent.process_user_input("The customer has 5 team members with different roles")
# print(f"   Response: {response3[:100]}...")

# print("\n4. Fourth interaction - user asks for status (should go to feedback):")
# response4 = test_agent_intent.process_user_input("How's the information collection going?")
# print(f"   Response: {response4[:100]}...")

# print("\n✅ Intent classification test completed!")
# print("   - Each interaction should be classified based on the current input only")
# print("   - Conversation history should not bias the intent classification")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Test Information Accumulation Fix

# COMMAND ----------

# Test information accumulation to ensure user responses are being stored
print("=== Testing Information Accumulation Fix ===")
print("🧪 [INFO_ACCUMULATION_TEST] Testing that user information is properly accumulated...")

# Create a test agent and simulate the information collection flow
test_agent_info = test_session_manager.get_agent_for_user("test_user_info")

print("\n1. First interaction - user provides initial context:")
response1 = test_agent_info.process_user_input("I am working with a customer for a Oracle Datawarehouse to Databricks migration. It's a large instance with multiple database with around 50TB data.")
print(f"   Response: {response1[:100]}...")

print("\n2. Second interaction - user answers questions:")
response2 = test_agent_info.process_user_input("There are around 10 Databases with around 5 schemas in each database, roughly around 1000 tables, views, stored procedures etc... we need to export data from salesforce. using PL/SQL procedures for regular ETL and data processing. the customer is regulated GxP validation will be required for the data products.")
print(f"   Response: {response2[:100]}...")

print("\n3. Third interaction - user provides more details:")
response3 = test_agent_info.process_user_input("The customer has 5 team members with different roles - 2 data engineers, 1 data architect, 1 business analyst, and 1 project manager.")
print(f"   Response: {response3[:100]}...")

print("\n4. Check status to see accumulated information:")
response4 = test_agent_info.process_user_input("/status")
print(f"   Response: {response4[:200]}...")

print("\n✅ Information accumulation test completed!")
print("   - User responses should be properly extracted and stored")
print("   - Status should show accumulated information, not empty summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. MLflow Model Registration

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment(mlflow_experiment_name)

# Disable MLflow autologging to avoid Pydantic serialization warnings
mlflow.autolog(disable=True)
print("MLflow autologging disabled to avoid Pydantic serialization warnings")

# COMMAND ----------

# Create ResponsesAgent wrapper for Databricks Agent Framework compatibility
# Following the working pattern from migration_planning_agent_v2.py
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

class MigrationPlanningResponsesAgent(ResponsesAgent):
    """ResponsesAgent wrapper for Databricks Agent Framework compatibility with session management."""
    
    def __init__(self, vector_search_endpoint, vector_index):
        super().__init__()
        # Create session manager instead of single agent
        self.session_manager = SessionManager(vector_search_endpoint, vector_index)
        # Configure DSPy LM for the agent
        self._configure_dspy_lm()
    
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
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction with session management."""
        # Ensure DSPy LM is configured
        self._ensure_dspy_lm_configured()
        
        # Extract user input and user ID from the request
        user_input = self._extract_user_input(request)
        user_id = self._extract_user_id(request)
        
        # Get or create agent for this user
        agent = self.session_manager.get_agent_for_user(user_id)
        
        # Call the DSPy agent for this user's session
        response = agent.process_user_input(user_input)
        
        # Convert to ResponsesAgent format
        output_item = self.create_text_output_item(
            text=str(response), 
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
    
    def predict_stream(self, request: ResponsesAgentRequest):
        """Simple streaming - returns single response as stream."""
        # Get the response from regular predict
        response = self.predict(request)
        
        # Return as single stream event (no actual streaming)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=response.output[0]
        )
    
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
        """Extract user ID from ResponsesAgentRequest."""
        # Try to get user ID from request context or headers
        if hasattr(request, 'context') and request.context:
            if 'user_id' in request.context:
                return str(request.context['user_id'])
            if 'session_id' in request.context:
                return str(request.context['session_id'])
        
        # Try to get from request headers if available
        if hasattr(request, 'headers') and request.headers:
            if 'user-id' in request.headers:
                return str(request.headers['user-id'])
            if 'session-id' in request.headers:
                return str(request.headers['session-id'])
        
        # Try to get from the first message if it contains user info
        if request.input and len(request.input) > 0:
            first_message = request.input[0]
            if hasattr(first_message, 'metadata') and first_message.metadata:
                if 'user_id' in first_message.metadata:
                    return str(first_message.metadata['user_id'])
        
        # Fallback: generate a unique user ID based on request
        # This is not ideal for production but works for MVP
        import hashlib
        request_str = str(request.input) if request.input else "default"
        user_id = hashlib.md5(request_str.encode()).hexdigest()[:8]
        print(f"🔐 [USER_ID] Generated fallback user ID: {user_id}")
        return user_id

# Create the ResponsesAgent wrapper with session management
responses_agent = MigrationPlanningResponsesAgent(vector_search_endpoint, vector_index_name)
print("ResponsesAgent wrapper created successfully with session management!")

# COMMAND ----------

# Input example for the model - using the same format as v2 notebook
input_example = {
    "input": [
        {
            "role": "user", 
            "content": "I need to migrate our Oracle data warehouse to Databricks"
        }
    ]
}

# Log the ResponsesAgent using MLflow (Databricks Agent Framework compatible)
# Following the working pattern from migration_planning_agent_v2.py
with mlflow.start_run() as run:
    # Log the ResponsesAgent - this automatically handles the correct schema for Agent Framework
    model_info = mlflow.pyfunc.log_model(
        python_model=responses_agent,
        name="usecase-planning-agent",
        input_example=input_example,
        # Note: Question categories are now passed dynamically to the question generator
        # No need to include them as static files
    )
    
    print(f"MigrationPlanningAgent MVP (ResponsesAgent) logged to MLflow: {model_info.model_uri}")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# Register the model in Unity Catalog
catalog_name = "vbdemos"
schema_name = "usecase_agent"
model_name = f"{catalog_name}.{schema_name}.usecase-planning-agent"
uc_model_info = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)

print(f"Model registered in Unity Catalog: {uc_model_info.name}")
print(f"   Version: {uc_model_info.version}")
print(f"Model ready for deployment!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Test Registered Model

# COMMAND ----------

# # Test the registered model with different inputs
# test_inputs = [
#     "Hi, what can you help with?",  # greeting
#     "I'm working with a customer who wants to migrate from Snowflake to Databricks",  # providing_context
#     "The customer has 5 team members with different roles",  # answering_questions
#     "How's the information collection going?",  # feedback_request
#     "/plan"  # plan_generation
# ]

# print("=== Testing Registered Model ===")
# for i, test_input in enumerate(test_inputs, 1):
#     print(f"\n--- Test {i}: {test_input} ---")
#     try:
#         # Test the model using ResponsesAgent format (same as v2 notebook)
#         test_request = {
#             "input": [
#                 {
#                     "role": "user", 
#                     "content": test_input
#                 }
#             ]
#         }
#         result = responses_agent.predict(test_request)
#         print(f"Response: {result.output[0].text[:200]}...")
#     except Exception as e:
#         print(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Deploy Agent to Model Serving

# COMMAND ----------

# Import required libraries for deployment
from databricks import agents
import mlflow
import time

# Check MLflow version compatibility
try:
    import mlflow
    mlflow_version = mlflow.__version__
    print(f"MLflow version: {mlflow_version}")
    if mlflow_version >= "2.8.0":
        print("MLflow version is compatible with Agent Framework.")
    else:
        print("Warning: MLflow version might not be fully compatible with Agent Framework.")
except ImportError:
    print("MLflow not available.")

# COMMAND ----------

# Deploy the agent to Model Serving
print("Deploying agent to Model Serving...")
print(f"Model Name: {uc_model_info.name}")
print(f"Model Version: {uc_model_info.version}")

# Deploy the agent using the Agent Framework
# Use a static endpoint name to update the same endpoint
endpoint_name = "migration-planning-agent"
print(f"\nDeploying to endpoint: {endpoint_name}")

deployment = agents.deploy(uc_model_info.name, uc_model_info.version, endpoint_name=endpoint_name)
print(f"Agent deployed successfully to endpoint: {endpoint_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Summary

# COMMAND ----------

print("Migration Planning Agent MVP - Complete!")
print("=" * 60)
print(f"Model registered: {uc_model_info.name}")
print(f"Model version: {uc_model_info.version}")
print(f"Agent deployed to endpoint: {endpoint_name}")
print(f"Query endpoint: {deployment.query_endpoint}")
print(f"Deployment ID: {deployment.deployment_id}")
print("\nNext Steps:")
print("1. Test the agent via the Review App in Databricks UI")
print("2. Monitor agent performance using inference tables")
print("3. Provide feedback using the feedback API")
print("4. Scale the endpoint as needed for production traffic")
print("\nUseful Links:")
print(f"- Model in Unity Catalog: {uc_model_info.name}")
print(f"- Query Endpoint: {deployment.query_endpoint}")
print("- Review App: Available in Databricks UI under Model Serving")
print("- Inference Tables: Monitor logs and performance")
print("\nNote: Agent will update the same endpoint 'migration-planning-agent' on future deployments")

# COMMAND ----------
