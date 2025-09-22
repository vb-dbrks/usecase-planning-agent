# Databricks notebook source
# MAGIC %md
# MAGIC # Migration Planning Agent V2 - Simplified Design
# MAGIC 
# MAGIC This notebook implements a simplified DSPy-based migration planning agent with 4 specialized agents and 7 signatures.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Install required packages with version pinning for DBR 16.4 LTS compatibility
%pip install dspy databricks-vectorsearch mlflow flask

# Restart Python to ensure packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries
import dspy
import mlflow
import mlflow.dspy
from databricks.vector_search.client import VectorSearchClient
# MLflow DSPy handles schemas automatically
import json
import pandas as pd
from typing import List, Dict, Any
import pydantic
import numpy as np
from copy import copy

# Check versions for DBR 16.4 LTS compatibility
print(f"DSPy version: {dspy.__version__}")
print(f"Pydantic version: {pydantic.__version__}")
print(f"MLflow version: {mlflow.__version__}")
print("Version compatibility check completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Get Parameters

# COMMAND ----------

# vector_search_endpoint = dbutils.widgets.text("vector_search_endpoint", "usecase-agent")
# vector_index_name = dbutils.widgets.text("vector_index_name", "vbdemos.usecase_agent.migration_planning_documents")
# migration_documents_table = dbutils.widgets.text("migration_documents_table", "vbdemos.usecase_agent.migration_documents")
# agent_model = dbutils.widgets.text("agent_model", "databricks-claude-3-7-sonnet")
# temperature = dbutils.widgets.text("temperature", "0.1")
# max_tokens = dbutils.widgets.text("max_tokens", "5000")
# mlflow_experiment_name = dbutils.widgets.text("mlflow_experiment_name", "/Users/varun.bhandary@databricks.com/usecase-agent")

# Get parameters from job configuration
vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint")
vector_index_name = dbutils.widgets.get("vector_index_name")
migration_documents_table = dbutils.widgets.get("migration_documents_table")
agent_model = dbutils.widgets.get("agent_model")
temperature = float(dbutils.widgets.get("temperature"))
max_tokens = int(dbutils.widgets.get("max_tokens"))
mlflow_experiment_name = dbutils.widgets.get("mlflow_experiment_name")

print(f"Vector search endpoint: {vector_search_endpoint}")
print(f"Vector index name: {vector_index_name}")
print(f"Migration documents table: {migration_documents_table}")
print(f"Agent model: {agent_model}")
print(f"Temperature: {temperature}")
print(f"Max tokens: {max_tokens}")
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

# Initialize Vector Search client
vsc = VectorSearchClient()
print("DSPy and Vector Search configured successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. DSPy Signatures (7 Total)

# COMMAND ----------

# Question Management Signatures
class QuestionSelector(dspy.Signature):
    """Select the next question based on conversation progress and project context. Focus on gathering key migration details efficiently."""
    project_context: str = dspy.InputField(desc="Current project context and objectives")
    answered_questions: str = dspy.InputField(desc="Previously answered questions and responses")
    current_category: str = dspy.InputField(desc="Current planning category")
    conversation_stage: str = dspy.InputField(desc="Current stage: initial, questioning, ready_for_plan")
    next_question: str = dspy.OutputField(desc="The next specific question to ask the user. Avoid repeating similar questions. Focus on gathering unique information.")
    category: str = dspy.OutputField(desc="The category this question belongs to")
    priority: str = dspy.OutputField(desc="Priority level: high, medium, low")
    completion_status: str = dspy.OutputField(desc="Status: complete, in_progress, needs_more_info")

class DataAccumulator(dspy.Signature):
    """Structure and accumulate user responses into organized data."""
    question: str = dspy.InputField(desc="The question that was asked")
    user_answer: str = dspy.InputField(desc="The user's answer")
    project_context: str = dspy.InputField(desc="Project context and objectives")
    existing_data: str = dspy.InputField(desc="Previously accumulated structured data")
    structured_data: str = dspy.OutputField(desc="Updated structured data in JSON format")
    updated_context: str = dspy.OutputField(desc="Updated project context")
    data_completeness: str = dspy.OutputField(desc="Completeness percentage and missing elements")

# Knowledge Retrieval Signatures
class DocumentSearcher(dspy.Signature):
    """Search for relevant migration documents based on query and context."""
    query: str = dspy.InputField(desc="Search query for relevant documents")
    project_context: str = dspy.InputField(desc="Project context and objectives")
    search_type: str = dspy.InputField(desc="Type of search: general, timeline, resources, technical, risks")
    relevant_documents: str = dspy.OutputField(desc="List of relevant document titles and summaries")
    search_confidence: str = dspy.OutputField(desc="Confidence level in search results")
    document_categories: str = dspy.OutputField(desc="Categories of found documents")

class ReferenceSummarizer(dspy.Signature):
    """Summarize documents for specific plan sections."""
    documents: str = dspy.InputField(desc="Relevant documents to summarize")
    plan_section: str = dspy.InputField(desc="Plan section: timeline, resources, technical_approach, risks, considerations")
    project_context: str = dspy.InputField(desc="Project context and objectives")
    summarized_references: str = dspy.OutputField(desc="Summarized references for this plan section")
    key_insights: str = dspy.OutputField(desc="Key insights and best practices")
    best_practices: str = dspy.OutputField(desc="Specific best practices and recommendations")

# Plan Generation Signatures
class PlanGenerator(dspy.Signature):
    """Generate comprehensive migration plan using structured data and references."""
    project_context: str = dspy.InputField(desc="Project context and objectives")
    structured_data: str = dspy.InputField(desc="Accumulated structured data from user responses")
    references: str = dspy.InputField(desc="Summarized references for each plan section")
    plan_sections: str = dspy.InputField(desc="Required plan sections: timeline, resources, technical_approach, risks, considerations")
    migration_plan: str = dspy.OutputField(desc="Comprehensive migration plan")
    timeline: str = dspy.OutputField(desc="Detailed timeline with phases and milestones")
    resource_requirements: str = dspy.OutputField(desc="Resource requirements and team structure")
    risks: str = dspy.OutputField(desc="Identified risks and mitigation strategies")

class PlanFormatter(dspy.Signature):
    """Format migration plan into structured tables and additional information."""
    migration_plan: str = dspy.InputField(desc="Generated migration plan")
    format_requirements: str = dspy.InputField(desc="Format requirements: tabular_structure")
    formatted_plan: str = dspy.OutputField(desc="Formatted plan with tables and structure")
    tables: str = dspy.OutputField(desc="Structured tables for timeline, resources, risks")
    additional_info: str = dspy.OutputField(desc="Additional information and considerations")

# Plan Evaluation Signature
class PlanEvaluator(dspy.Signature):
    """Evaluate migration plan completeness and quality."""
    generated_plan: str = dspy.InputField(desc="Generated migration plan")
    project_context: str = dspy.InputField(desc="Project context and objectives")
    structured_data: str = dspy.InputField(desc="Accumulated structured data")
    evaluation_criteria: str = dspy.InputField(desc="Evaluation criteria: completeness_and_quality")
    completeness_score: str = dspy.OutputField(desc="Completeness score (0-100)")
    quality_score: str = dspy.OutputField(desc="Quality score (0-100)")
    missing_elements: str = dspy.OutputField(desc="Missing elements and gaps")
    recommendations: str = dspy.OutputField(desc="Recommendations for improvement")

print("✅ All 7 DSPy signatures defined successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Planning Categories (Institutional Knowledge)

# COMMAND ----------

# PLANNING_CATEGORIES - Institutional knowledge from Databricks Professional Services
PLANNING_CATEGORIES = {
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

print("✅ Planning categories defined successfully!")
print(f"Categories: {list(PLANNING_CATEGORIES.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Agent Implementations

# COMMAND ----------

class QuestionManagementAgent(dspy.Module):
    """Manages question flow, tracks progress, maintains structured data."""
    
    def __init__(self):
        super().__init__()
        self.question_selector = dspy.ChainOfThought(QuestionSelector)
        self.data_accumulator = dspy.ChainOfThought(DataAccumulator)
        
        # State management
        self.answered_questions = {}
        self.categories_progress = {}
        self.structured_data = {}
        self.current_category = "Resource & Team"
        self.categories = list(PLANNING_CATEGORIES.keys())
        self.current_question = ""
        self.question_count = 0
        self.max_questions = 10  # Limit total questions
        self.category_questions = {}  # Track questions per category
    
    def get_next_question(self, user_input, project_context):
        """Get the next question based on progress."""
        # Check if we have enough information
        if self.question_count >= self.max_questions or self._has_sufficient_data():
            return dspy.Prediction(
                next_question="",
                category="complete",
                priority="low",
                completion_status="complete"
            )
        
        # Move to next category if current one is complete
        if self._is_current_category_complete():
            self._move_to_next_category()
        
        result = self.question_selector(
            project_context=project_context,
            answered_questions=json.dumps(self.answered_questions),
            current_category=self.current_category,
            conversation_stage=self._get_conversation_stage()
        )
        
        self.current_question = result.next_question
        self.question_count += 1
        
        # Track questions per category
        if self.current_category not in self.category_questions:
            self.category_questions[self.current_category] = 0
        self.category_questions[self.current_category] += 1
        
        return dspy.Prediction(
            next_question=result.next_question,
            category=result.category,
            priority=result.priority,
            completion_status=result.completion_status
        )
    
    def accumulate_data(self, question, user_answer, project_context):
        """Structure and store user response."""
        result = self.data_accumulator(
            question=question,
            user_answer=user_answer,
            project_context=project_context,
            existing_data=json.dumps(self.structured_data)
        )
        
        # Update state
        self.answered_questions[question] = user_answer
        try:
            new_data = json.loads(result.structured_data)
            self.structured_data.update(new_data)
        except:
            pass
        
        return dspy.Prediction(
            structured_data=result.structured_data,
            updated_context=result.updated_context,
            data_completeness=result.data_completeness
        )
    
    def _get_conversation_stage(self):
        """Determine current conversation stage."""
        if self.question_count == 0:
            return "initial"
        elif self.question_count < 6:
            return "questioning"
        else:
            return "ready_for_plan"
    
    def _has_sufficient_data(self):
        """Check if we have enough data to generate a plan."""
        required_fields = [
            "team", "data", "timeline", "business", "technical", "architecture"
        ]
        data_str = str(self.structured_data).lower()
        return sum(1 for field in required_fields if field in data_str) >= 4
    
    def _is_current_category_complete(self):
        """Check if current category has enough questions answered."""
        return self.category_questions.get(self.current_category, 0) >= 2
    
    def _move_to_next_category(self):
        """Move to the next planning category."""
        current_index = self.categories.index(self.current_category)
        if current_index < len(self.categories) - 1:
            self.current_category = self.categories[current_index + 1]
        else:
            self.current_category = "complete"

class KnowledgeRetrievalAgent(dspy.Module):
    """Searches vector DB for relevant migration documents and references."""
    
    def __init__(self, vector_search_endpoint, vector_index):
        super().__init__()
        self.document_searcher = dspy.ChainOfThought(DocumentSearcher)
        self.reference_summarizer = dspy.ChainOfThought(ReferenceSummarizer)
        self.vector_search = VectorSearchClient()
        self.endpoint = vector_search_endpoint
        self.index = vector_index
    
    def search_documents(self, query, project_context, search_type="general"):
        """Search for relevant documents."""
        try:
            # Search vector database
            search_results = self.vector_search.get_index(
                endpoint_name=self.endpoint,
                index_name=self.index
            ).query(
                query_text=query,
                columns=["content", "title", "category", "source"],
                num_results=5
            )
            
            # Format results for AI processing
            formatted_docs = []
            for doc in search_results.get('result', {}).get('data_array', []):
                formatted_docs.append(f"Title: {doc[1]}\nContent: {doc[0][:500]}...")
            
            docs_text = "\n\n".join(formatted_docs)
        except Exception as e:
            docs_text = f"Error searching documents: {str(e)}"
        
        result = self.document_searcher(
            query=query,
            project_context=project_context,
            search_type=search_type
        )
        
        return dspy.Prediction(
            relevant_documents=docs_text,
            search_confidence=result.search_confidence,
            document_categories=result.document_categories
        )
    
    def summarize_for_plan_section(self, documents, plan_section, project_context):
        """Summarize documents for specific plan sections."""
        result = self.reference_summarizer(
            documents=documents,
            plan_section=plan_section,
            project_context=project_context
        )
        
        return dspy.Prediction(
            summarized_references=result.summarized_references,
            key_insights=result.key_insights,
            best_practices=result.best_practices
        )

class PlanGenerationAgent(dspy.Module):
    """Generates structured migration plan using accumulated data and references."""
    
    def __init__(self, knowledge_agent):
        super().__init__()
        self.plan_generator = dspy.ChainOfThought(PlanGenerator)
        self.plan_formatter = dspy.ChainOfThought(PlanFormatter)
        self.knowledge_agent = knowledge_agent
    
    def generate_plan(self, project_context, structured_data):
        """Generate comprehensive migration plan."""
        # Search for references for each plan section
        plan_sections = ["timeline", "resources", "technical_approach", "risks", "considerations"]
        references = {}
        
        for section in plan_sections:
            # Search for relevant documents for this section
            search_query = f"{project_context} {section} migration best practices"
            docs = self.knowledge_agent.search_documents(search_query, project_context, section)
            
            # Summarize references for this section
            refs = self.knowledge_agent.summarize_for_plan_section(
                docs.relevant_documents, section, project_context
            )
            references[section] = refs
        
        # Generate the plan
        result = self.plan_generator(
            project_context=project_context,
            structured_data=json.dumps(structured_data),
            references=json.dumps(references),
            plan_sections=", ".join(plan_sections)
        )
        
        # Format the plan
        formatted = self.plan_formatter(
            migration_plan=result.migration_plan,
            format_requirements="tabular_structure"
        )
        
        return dspy.Prediction(
            migration_plan=result.migration_plan,
            timeline=result.timeline,
            resource_requirements=result.resource_requirements,
            risks=result.risks,
            formatted_plan=formatted.formatted_plan,
            tables=formatted.tables,
            additional_info=formatted.additional_info
        )

class PlanEvaluationAgent(dspy.Module):
    """Evaluates migration plan completeness and quality."""
    
    def __init__(self):
        super().__init__()
        self.plan_evaluator = dspy.ChainOfThought(PlanEvaluator)
    
    def evaluate_plan(self, generated_plan, project_context, structured_data):
        """Evaluate plan completeness and quality."""
        result = self.plan_evaluator(
            generated_plan=generated_plan,
            project_context=project_context,
            structured_data=json.dumps(structured_data),
            evaluation_criteria="completeness_and_quality"
        )
        
        return dspy.Prediction(
            completeness_score=result.completeness_score,
            quality_score=result.quality_score,
            missing_elements=result.missing_elements,
            recommendations=result.recommendations
        )

print("✅ All 4 agents implemented successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Main Migration Planning Agent

# COMMAND ----------

class MigrationPlanningAgent(dspy.Module):
    """Main agent that orchestrates the entire migration planning process."""
    
    def __init__(self, vector_search_endpoint, vector_index):
        super().__init__()
        self.question_agent = QuestionManagementAgent()
        self.knowledge_agent = KnowledgeRetrievalAgent(vector_search_endpoint, vector_index)
        self.plan_agent = PlanGenerationAgent(self.knowledge_agent)
        self.evaluation_agent = PlanEvaluationAgent()
        
        # State
        self.project_context = ""
        self.conversation_stage = "initial"
        self.plan_generated = False
    
    def forward(self, user_input=None, **kwargs):
        """Main forward method for MLflow deployment."""
        # Handle different input formats from MLflow PyFunc
        if user_input is None:
            # If no positional argument, check kwargs
            if "user_input" in kwargs:
                user_input = kwargs["user_input"]
            elif "inputs" in kwargs:
                user_input = kwargs["inputs"]
            else:
                # Try to get the first value from kwargs
                user_input = next(iter(kwargs.values())) if kwargs else ""
        
        # Handle MLflow serving format: {"inputs": "string"}
        if isinstance(user_input, dict) and "inputs" in user_input:
            user_input = user_input["inputs"]
        elif isinstance(user_input, dict):
            # Handle other dictionary formats
            user_input = str(user_input)
        else:
            user_input = str(user_input).strip()
        
        # Handle special commands
        if user_input.lower() == "/plan":
            return self._generate_plan()
        elif user_input.lower() == "/status":
            return self._get_status()
        elif user_input.lower() == "/help":
            return self._get_help()
        
        # Handle conversation
        if self.conversation_stage == "initial":
            return self._handle_initial_input(user_input)
        else:
            return self._handle_question_answer(user_input)
    
    def _handle_initial_input(self, user_input):
        """Handle initial user input."""
        self.project_context = user_input
        self.conversation_stage = "questioning"
        
        # Get first question
        question_result = self.question_agent.get_next_question(user_input, self.project_context)
        
        return f"I'll help you plan this migration. Let me ask some questions to understand your project better.\n\n{question_result.next_question}"
    
    def _handle_question_answer(self, user_input):
        """Handle user answers to questions."""
        # Accumulate the answer
        data_result = self.question_agent.accumulate_data(
            question=self.question_agent.current_question,
            user_answer=user_input,
            project_context=self.project_context
        )
        
        # Get next question
        question_result = self.question_agent.get_next_question(user_input, self.project_context)
        
        if question_result.completion_status == "complete":
            return "Thanks! I have enough information to create a comprehensive migration plan. Use /plan to generate your detailed migration plan."
        else:
            # Provide more context about what we're learning
            category_info = f" (Category: {question_result.category})" if question_result.category != "complete" else ""
            return f"Thanks! {question_result.next_question}{category_info}"
    
    def _generate_plan(self):
        """Generate migration plan."""
        # Generate the plan
        plan_result = self.plan_agent.generate_plan(
            self.project_context,
            self.question_agent.structured_data
        )
        
        # Evaluate the plan
        evaluation = self.evaluation_agent.evaluate_plan(
            plan_result.migration_plan,
            self.project_context,
            self.question_agent.structured_data
        )
        
        self.plan_generated = True
        
        return f"Here's your migration plan:\n\n{plan_result.formatted_plan}\n\nCompleteness Score: {evaluation.completeness_score}/100\nQuality Score: {evaluation.quality_score}/100"
    
    def _get_status(self):
        """Get current status."""
        return f"""Status: {self.conversation_stage}
Questions answered: {self.question_agent.question_count}/{self.question_agent.max_questions}
Current category: {self.question_agent.current_category}
Categories covered: {list(self.question_agent.category_questions.keys())}
Data completeness: {len(self.question_agent.structured_data)} fields captured
Ready for plan: {'Yes' if self.question_agent.question_count >= 6 else 'No'}"""
    
    def _get_help(self):
        """Get help information."""
        return "Available commands:\n/plan - Generate migration plan\n/status - Show current status\n/help - Show this help\n\nJust answer the questions to continue planning!"

print("✅ Main MigrationPlanningAgent implemented successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. DSPy Model Compilation

# COMMAND ----------

# Create the MigrationPlanningAgent instance
migration_agent = MigrationPlanningAgent(
    vector_search_endpoint=vector_search_endpoint,
    vector_index=vector_index_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compile the DSPy Model
# MAGIC 
# MAGIC Following the MLflow DSPy documentation, we need to compile our DSPy model using an optimizer to improve its performance before logging to MLflow.

# COMMAND ----------

# Create training examples for compilation
# Need some examples to help the optimizer learn better prompts
training_examples = [
    {
        "user_input": "I need to migrate our Oracle data warehouse to Databricks",
        "expected_response": "I'll help you plan this migration. Let me ask some questions to understand your project better.\n\nWhat is the primary business driver for this migration?"
    },
    {
        "user_input": "We want to improve performance and reduce costs",
        "expected_response": "Thanks! What is the current data volume and processing requirements?"
    },
    {
        "user_input": "We have 50TB of data and need real-time analytics",
        "expected_response": "Thanks! What is your current team structure and technical expertise with cloud platforms?"
    },
    {
        "user_input": "We have 5 data engineers and some AWS experience",
        "expected_response": "Thanks! What is your current data architecture and integration patterns?"
    },
    {
        "user_input": "We use ETL processes with batch processing",
        "expected_response": "Thanks! I have enough information. Use /plan to generate your migration plan."
    }
]

# Convert to DSPy examples format
dspy_examples = []
for example in training_examples:
    dspy_examples.append(dspy.Example(
        user_input=example["user_input"],
        response=example["expected_response"]
    ).with_inputs("user_input"))

print(f"Created {len(dspy_examples)} training examples for DSPy compilation")

# COMMAND ----------

# Compile the DSPy model using BootstrapFewShotWithRandomSearch optimizer
# Following the MLflow DSPy documentation pattern
print("Compiling DSPy model with BootstrapFewShotWithRandomSearch...")

# Create a simple wrapper class for compilation
class MigrationPlanningWrapper(dspy.Module):
    def __init__(self, base_agent):
        super().__init__()
        self.base_agent = base_agent
    
    def forward(self, user_input):
        # Use the proper DSPy calling convention: module(...) instead of module.forward(...)
        return self.base_agent(user_input)

# Wrap the agent for compilation
wrapped_agent = MigrationPlanningWrapper(migration_agent)

# Define a metric function for evaluation
def accuracy_metric(example, prediction, trace=None):
    """Simple accuracy metric for DSPy compilation."""
    # Using string matching for the conversational agent
    expected = example.response.lower().strip()
    actual = prediction.lower().strip()
    
    # Check if the prediction contains key phrases from the expected response
    if expected in actual or actual in expected:
        return 1.0
    
    # Check for partial matches based on word overlap
    expected_words = set(expected.split())
    actual_words = set(actual.split())
    overlap = len(expected_words.intersection(actual_words))
    
    # Return a score based on word overlap
    if len(expected_words) > 0:
        return overlap / len(expected_words)
    else:
        return 0.0

# Use BootstrapFewShotWithRandomSearch optimizer
# Same optimizer used in the MLflow DSPy documentation
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=accuracy_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=8,
    num_threads=1
)

# Compile the model
try:
    compiled_agent = optimizer.compile(
        wrapped_agent,
        trainset=dspy_examples[:3],  # Use first 3 examples for training
        valset=dspy_examples[3:]     # Use remaining examples for validation
    )
    print("DSPy model compiled successfully!")
    
    # Extract the compiled base agent
    compiled_migration_agent = compiled_agent.base_agent
    
except Exception as e:
    print(f"Compilation failed: {e}")
    print("Using uncompiled model for MLflow logging...")
    compiled_migration_agent = migration_agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. MLflow Model Registration

# COMMAND ----------

# Input example for the model - plain string as per MLflow DSPy documentation
input_example = "I need to migrate our Oracle data warehouse to Databricks"

# Import os for environment variables
import os

# Create a ChatModel wrapper following the reference code pattern
from dataclasses import dataclass
from typing import Optional, Dict, List, Generator
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    ChatChoice,
    ChatParams,
    ChatChoiceDelta,
    ChatChunkChoice,
)

class MigrationPlanningChatModel(ChatModel):
    def __init__(self):
        # Store only serializable parameters
        self.agent_model = agent_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Store the token and URL from the notebook context (these are strings, so serializable)
        self.token = token
        self.url = url
    
    def load_context(self, context):
        """Configure DSPy when the model is loaded in the serving environment."""
        # Configure DSPy with the stored token and URL
        lm = dspy.LM(
            model=f"databricks/{self.agent_model}",
            api_key=self.token,
            api_base=self.url,
        )
        
        dspy.configure(lm=lm)
    
    def _prepare_messages(self, messages: List[ChatMessage]):
        return {"messages": [m.to_dict() for m in messages]}
      
    def predict(self, context, messages: list[ChatMessage], params=None) -> ChatCompletionResponse:
        # Extract the user's question from the last message
        question = messages[-1].content
        print(f"User question: {question}")
        
        # Configure DSPy for this prediction if not already configured
        try:
            # Check if DSPy is already configured
            dspy.settings.lm
        except:
            # Configure DSPy with the stored token and URL
            lm = dspy.LM(
                model=f"databricks/{self.agent_model}",
                api_key=self.token,
                api_base=self.url,
            )
            
            dspy.configure(lm=lm)
        
        # Create a new instance of the migration agent for this prediction
        # This avoids serialization issues
        migration_agent = MigrationPlanningAgent(
            vector_search_endpoint=vector_search_endpoint,
            vector_index=vector_index_name
        )
        
        # Use the agent to get response
        response = migration_agent.forward(question)
        
        # Create response message
        response_message = ChatMessage(
            role="assistant",
            content=response
        )
        
        return ChatCompletionResponse(
            choices=[ChatChoice(message=response_message)]
        )

# Log the MigrationPlanningAgent using MLflow PyFunc with ChatModel
# Following the reference code pattern for model serving
with mlflow.start_run() as run:
    # Log using pyfunc with ChatModel for proper serving
    model_info = mlflow.pyfunc.log_model(
        python_model=MigrationPlanningChatModel(),
        name="usecase-planning-agent",
        input_example={
            "messages": [{"role": "user", "content": "I need to migrate our Oracle data warehouse to Databricks"}]
        }
    )
    
    print(f"MigrationPlanningAgent V2 logged to MLflow: {model_info.model_uri}")
    print(f"Run ID: {run.info.run_id}")

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
# MAGIC ### Compare Compiled vs Uncompiled Model Performance
# MAGIC 
# MAGIC Following the MLflow DSPy documentation pattern, let's compare the performance of our compiled vs uncompiled model.

# COMMAND ----------

# Test both models with the same input
test_input = "I need to migrate our Oracle data warehouse to Databricks"

print("Testing Uncompiled Model:")
print("-" * 50)
uncompiled_response = migration_agent.forward(test_input)
print(f"Response: {uncompiled_response}")

print("\nTesting Compiled Model:")
print("-" * 50)
compiled_response = compiled_migration_agent.forward(test_input)
print(f"Response: {compiled_response}")

print("\nModel comparison completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test MLflow Model Loading
# MAGIC 
# MAGIC Test loading the logged model using both DSPy and PyFunc APIs

# COMMAND ----------

# Test the ChatModel following the reference pattern
print("Testing MigrationPlanningChatModel:")
print("-" * 50)

# Test the ChatModel directly
agent = MigrationPlanningChatModel()
model_input = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content=test_input)]
)
response = agent.predict(context=None, messages=model_input.messages)
print(f"ChatModel Response: {response.choices[0].message.content}")

print("\nTesting Loaded PyFunc Model:")
print("-" * 50)
loaded_pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
# Test with proper ChatModel input format (dictionary with messages key)
test_input_dict = {"messages": [{"role": "user", "content": test_input}]}
pyfunc_response = loaded_pyfunc_model.predict(test_input_dict)
print(f"PyFunc Response: {pyfunc_response}")

print("\nMLflow model loading test completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Interactive Testing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Migration Planning Agent
# MAGIC 
# MAGIC Use this cell to interactively test the agent. Try these examples:
# MAGIC 
# MAGIC 1. **Start a migration project**: "I need to migrate Oracle to Databricks for a customer"
# MAGIC 2. **Answer questions**: Respond to the questions the agent asks
# MAGIC 3. **Check status**: "/status"
# MAGIC 4. **Generate plan**: "/plan"
# MAGIC 5. **Get help**: "/help"

# COMMAND ----------

# Interactive testing cell - modify this to test different inputs
def test_agent(user_input):
    """Test the migration planning agent with user input."""
    print(f"Testing with input: '{user_input}'")
    print("=" * 60)
    
    # Test the compiled agent directly
    result = compiled_migration_agent.forward(user_input)
    
    print(f"Response: {result}")
    print("=" * 60)
    return result

def test_chat_model(user_input):
    """Test the ChatModel with proper message format following reference pattern."""
    print(f"Testing ChatModel with input: '{user_input}'")
    print("=" * 60)
    
    # Test the ChatModel following the reference pattern
    agent = MigrationPlanningChatModel()
    model_input = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=user_input)]
    )
    response = agent.predict(context=None, messages=model_input.messages)
    
    print(f"ChatModel Response: {response.choices[0].message.content}")
    print("=" * 60)
    return response

def test_pyfunc_model(user_input):
    """Test the PyFunc model with proper input format."""
    print(f"Testing PyFunc model with input: '{user_input}'")
    print("=" * 60)
    
    # Test the PyFunc model with proper ChatModel input format
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    test_input_dict = {"messages": [{"role": "user", "content": user_input}]}
    response = loaded_pyfunc_model.predict(test_input_dict)
    
    print(f"PyFunc Response: {response}")
    print("=" * 60)
    return response

# Test examples - uncomment and modify as needed
# test_agent("I need to migrate Oracle to Databricks for a customer")
# test_agent("We have 5 developers, 2 data engineers, and 1 DBA")
# test_agent("Our Oracle database is 500GB with 1000 tables")
# test_agent("/status")
# test_agent("/plan")

# Test ChatModel examples - uncomment and modify as needed
# test_chat_model("I need to migrate Oracle to Databricks for a customer")
# test_chat_model("We have 5 developers, 2 data engineers, and 1 DBA")
# test_chat_model("Our Oracle database is 500GB with 1000 tables")

# Test PyFunc model examples - uncomment and modify as needed
# test_pyfunc_model("I need to migrate Oracle to Databricks for a customer")
# test_pyfunc_model("We have 5 developers, 2 data engineers, and 1 DBA")
# test_pyfunc_model("Our Oracle database is 500GB with 1000 tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Test MLflow Model Loading

# COMMAND ----------

# Test the logged ChatModel following the reference pattern
print("Testing the logged ChatModel...")

# Test the ChatModel following the reference pattern
agent = MigrationPlanningChatModel()
model_input = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="I need to migrate Oracle to Databricks for a customer")]
)
response = agent.predict(context=None, messages=model_input.messages)

print(f"Model loaded successfully!")
print(f"Response: {response.choices[0].message.content}")

# Test the PyFunc model as well
print("\nTesting the logged PyFunc model...")
loaded_pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
test_input_dict = {"messages": [{"role": "user", "content": "I need to migrate Oracle to Databricks for a customer"}]}
pyfunc_response = loaded_pyfunc_model.predict(test_input_dict)
print(f"PyFunc Response: {pyfunc_response}")

print("All tests completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Improved Agent Behavior

# COMMAND ----------

# Test the improved agent with a conversation flow
def test_improved_agent():
    """Test the improved agent behavior with a realistic conversation."""
    print("Testing Improved Migration Planning Agent")
    print("=" * 60)
    
    # Create a fresh agent instance
    test_agent = MigrationPlanningAgent(
        vector_search_endpoint=vector_search_endpoint,
        vector_index=vector_index_name
    )
    
    # Simulate a conversation
    test_inputs = [
        "I want to migrate a customer from Oracle to Databricks",
        "There's a central data platform team with 10 DBAs and 20 Data Engineers",
        "Around 5 Departments: Sales, Finance, Marketing, HR, Manufacturing with 2 DBAs each and 4 Data Engineers in each department",
        "Migrate their Oracle Exadata Platform to Databricks. They have around 50 reports in each department with overall size of 50TB. Need a plan to migrate this to Databricks in 6 months"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- User Input {i} ---")
        print(f"User: {user_input}")
        
        response = test_agent.forward(user_input)
        print(f"Agent: {response}")
        
        # Show status after each interaction
        status = test_agent._get_status()
        print(f"\nStatus: {status}")
        
        print("-" * 40)

# Run the test
test_improved_agent()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Deploy Agent to Model Serving

# COMMAND ----------

# Install the databricks-agents SDK
%pip install databricks-agents
dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries for deployment
from databricks import agents

# COMMAND ----------

# Deploy the agent to Model Serving
print("Deploying agent to Model Serving...")
print(f"Model Name: {uc_model_info.name}")
print(f"Model Version: {uc_model_info.version}")

# Deploy the agent using the Agent Framework
deployment = agents.deploy(uc_model_info.name, uc_model_info.version)

# Retrieve the query endpoint URL for making API requests
print(f"✅ Agent deployed successfully!")
print(f"Query Endpoint URL: {deployment.query_endpoint}")
print(f"Deployment ID: {deployment.deployment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Deployed Agent

# COMMAND ----------

# Test the deployed agent endpoint
import requests
import json

# Get the endpoint URL
endpoint_url = deployment.query_endpoint

# Test payload
test_payload = {
    "messages": [
        {
            "role": "user", 
            "content": "I need to migrate our Oracle data warehouse to Databricks for a customer"
        }
    ]
}

# Make request to the deployed endpoint
try:
    response = requests.post(
        endpoint_url,
        headers={"Content-Type": "application/json"},
        json=test_payload,
        auth=("token", token)  # Use the token we captured earlier
    )
    
    if response.status_code == 200:
        print("✅ Deployed agent test successful!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Test failed with status code: {response.status_code}")
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"❌ Test failed with error: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deployment Management

# COMMAND ----------

# List all current deployments
print("Current agent deployments:")
deployments = agents.list_deployments()
print(deployments)

# Get specific deployment details
deployment_details = agents.get_deployments(
    model_name=uc_model_info.name, 
    model_version=uc_model_info.version
)
print(f"\nDeployment details: {deployment_details}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary

# COMMAND ----------

print("🎉 Migration Planning Agent V2 - Complete!")
print("=" * 60)
print(f"✅ Model registered: {uc_model_info.name}")
print(f"✅ Model version: {uc_model_info.version}")
print(f"✅ Agent deployed: {deployment.query_endpoint}")
print(f"✅ Deployment ID: {deployment.deployment_id}")
print("\n📋 Next Steps:")
print("1. Test the agent via the Review App in Databricks UI")
print("2. Monitor agent performance using inference tables")
print("3. Provide feedback using the feedback API")
print("4. Scale the endpoint as needed for production traffic")
print("\n🔗 Useful Links:")
print(f"- Model in Unity Catalog: {uc_model_info.name}")
print(f"- Query Endpoint: {deployment.query_endpoint}")
print("- Review App: Available in Databricks UI under Model Serving")
print("- Inference Tables: Monitor logs and performance")
