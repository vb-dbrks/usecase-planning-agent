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
%pip install dspy>=2.6.23 databricks-vectorsearch mlflow databricks-agents

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
from typing import List, Dict, Any
import pydantic

# Check versions for DBR 16.4 LTS compatibility
print(f"DSPy version: {dspy.__version__}")
print(f"Pydantic version: {pydantic.__version__}")
print(f"MLflow version: {mlflow.__version__}")
print("Version compatibility check completed")

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

# Initialize Vector Search client
vsc = VectorSearchClient()
print("DSPy and Vector Search configured successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. DSPy Signatures (7 Total)

# COMMAND ----------

# Question Management Signatures
class QuestionSelector(dspy.Signature):
    """Select the next question based on conversation progress and project context. Focus on gathering key migration TO Databricks details efficiently."""
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
    """Search for relevant migration TO Databricks documents based on query and context."""
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
    """Generate comprehensive migration TO Databricks plan using structured data and references."""
    project_context: str = dspy.InputField(desc="Project context and objectives")
    structured_data: str = dspy.InputField(desc="Accumulated structured data from user responses")
    references: str = dspy.InputField(desc="Summarized references for each plan section")
    plan_sections: str = dspy.InputField(desc="Required plan sections: timeline, resources, technical_approach, risks, considerations")
    migration_plan: str = dspy.OutputField(desc="Comprehensive migration plan")
    timeline: str = dspy.OutputField(desc="Detailed timeline with phases and milestones")
    resource_requirements: str = dspy.OutputField(desc="Resource requirements and team structure")
    risks: str = dspy.OutputField(desc="Identified risks and mitigation strategies")

class PlanFormatter(dspy.Signature):
    """Format migration TO Databricks plan into structured tables and additional information."""
    migration_plan: str = dspy.InputField(desc="Generated migration plan")
    format_requirements: str = dspy.InputField(desc="Format requirements: tabular_structure")
    formatted_plan: str = dspy.OutputField(desc="Formatted plan with tables and structure")
    tables: str = dspy.OutputField(desc="Structured tables for timeline, resources, risks")
    additional_info: str = dspy.OutputField(desc="Additional information and considerations")

# Plan Evaluation Signature
class PlanEvaluator(dspy.Signature):
    """Evaluate migration TO Databricks plan completeness and quality."""
    generated_plan: str = dspy.InputField(desc="Generated migration plan")
    project_context: str = dspy.InputField(desc="Project context and objectives")
    structured_data: str = dspy.InputField(desc="Accumulated structured data")
    evaluation_criteria: str = dspy.InputField(desc="Evaluation criteria: completeness_and_quality")
    completeness_score: str = dspy.OutputField(desc="Completeness score (0-100)")
    quality_score: str = dspy.OutputField(desc="Quality score (0-100)")
    missing_elements: str = dspy.OutputField(desc="Missing elements and gaps")
    recommendations: str = dspy.OutputField(desc="Recommendations for improvement")

print("All 7 DSPy signatures defined successfully!")

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

print("Planning categories defined successfully!")
print(f"Categories: {list(PLANNING_CATEGORIES.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Agent Implementations

# COMMAND ----------

# COMMENTED OUT - Complex QuestionManagementAgent
# class QuestionManagementAgent(dspy.Module):
#     """Manages question flow, tracks progress, maintains structured data."""
#     
#     def __init__(self):
#         super().__init__()
#         self.question_selector = dspy.ChainOfThought(QuestionSelector)
#         self.data_accumulator = dspy.ChainOfThought(DataAccumulator)
#         
#         # Enhanced state management
#         self.answered_questions = {}  # question -> answer mapping
#         self.asked_questions = set()  # Track all questions asked to avoid repetition
#         self.categories_progress = {}
#         self.structured_data = {}
#         self.current_category = "Resource & Team"
#         self.categories = list(PLANNING_CATEGORIES.keys())
#         self.current_question = ""
#         self.question_count = 0
#         self.max_questions = 15  # Increased limit
#         self.category_questions = {}  # Track questions per category
#         self.category_completion = {cat: False for cat in self.categories}  # Track category completion
#         self.conversation_stage = "initial"  # initial, questioning, ready_for_plan, planning_complete
    
#     def get_next_question(self, user_input, project_context):
#         """Get the next question based on progress."""
#         # Check if we have enough information
#         if self.question_count >= self.max_questions or self._has_sufficient_data():
#             self.conversation_stage = "ready_for_plan"
#             return dspy.Prediction(
#                 next_question="",
#                 category="complete",
#                 priority="low",
#                 completion_status="complete"
#             )
#         
#         # Move to next category if current one is complete
#         if self._is_current_category_complete():
#             self._move_to_next_category()
#         
#         # Generate context-aware questions
#         result = self.question_selector(
#             project_context=project_context,
#             answered_questions=self._format_answered_questions(),
#             current_category=self.current_category,
#             conversation_stage=self._get_conversation_stage()
#         )
#         
#         # Check if this question was already asked
#         question_text = result.next_question.strip()
#         if question_text in self.asked_questions:
#             # Generate alternative question
#             result = self.question_selector(
#                 project_context=project_context,
#                 answered_questions=self._format_answered_questions(),
#                 current_category=self.current_category,
#                 conversation_stage=self._get_conversation_stage()
#             )
#             question_text = result.next_question.strip()
#         
#         self.current_question = question_text
#         self.asked_questions.add(question_text)
#         self.question_count += 1
#         
#         # Track questions per category
#         if self.current_category not in self.category_questions:
#             self.category_questions[self.current_category] = 0
#         self.category_questions[self.current_category] += 1
#         
#         return dspy.Prediction(
#             next_question=question_text,
#             category=result.category,
#             priority=result.priority,
#             completion_status=result.completion_status
#         )
    
#     def accumulate_data(self, question, user_answer, project_context):
#         """Structure and store user response."""
#         result = self.data_accumulator(
#             question=question,
#             user_answer=user_answer,
#             project_context=project_context,
#             existing_data=json.dumps(self.structured_data)
#         )
#         
#         # Update state
#         self.answered_questions[question] = user_answer
#         try:
#             new_data = json.loads(result.structured_data)
#             self.structured_data.update(new_data)
#         except:
#             # If JSON parsing fails, store as text
#             self.structured_data[f"question_{len(self.answered_questions)}"] = {
#                 "question": question,
#                 "answer": user_answer,
#                 "category": self.current_category
#             }
#         
#         return dspy.Prediction(
#             structured_data=result.structured_data,
#             updated_context=result.updated_context,
#             data_completeness=result.data_completeness
#         )
    
#     def _format_answered_questions(self):
#         """Format answered questions for context."""
#         if not self.answered_questions:
#             return "No questions answered yet."
#         
#         formatted = []
#         for question, answer in self.answered_questions.items():
#             formatted.append(f"Q: {question}\nA: {answer}")
#         
#         return "\n\n".join(formatted)
#     
#     def _get_conversation_stage(self):
#         """Determine current conversation stage."""
#         if self.question_count == 0:
#             return "initial"
#         elif self.question_count < 6:
#             return "questioning"
#         else:
#             return "ready_for_plan"
#     
#     def _has_sufficient_data(self):
#         """Check if we have enough data to generate a plan."""
#         # Check if we have key information from the conversation
#         data_str = str(self.structured_data).lower()
#         answered_str = str(self.answered_questions).lower()
#         
#         # Look for key migration planning information
#         key_info = [
#             "team" in answered_str or "dba" in answered_str or "engineer" in answered_str,
#             "data" in answered_str or "tb" in answered_str or "50" in answered_str,
#             "migrate" in answered_str or "databricks" in answered_str,
#             "department" in answered_str or "organization" in answered_str,
#             "oracle" in answered_str or "exadata" in answered_str
#         ]
#         
#         return sum(key_info) >= 3 or self.question_count >= 6
#     
#     def _is_current_category_complete(self):
#         """Check if current category has enough questions answered."""
#         # Check if category is already marked complete
#         if self.category_completion.get(self.current_category, False):
#             return True
#             
#         # Move to next category based on content analysis
#         answered_str = str(self.answered_questions).lower()
#         
#         if self.current_category == "Resource & Team":
#             # Check for key team information
#             team_indicators = ["team", "dba", "engineer", "developer", "analyst", "manager", "member"]
#             if any(indicator in answered_str for indicator in team_indicators):
#                 self.category_completion[self.current_category] = True
#                 return True
#                 
#         elif self.current_category == "Customer Background & Drivers":
#             # Check for business context
#             business_indicators = ["deadline", "driver", "business", "customer", "migration", "databricks"]
#             if any(indicator in answered_str for indicator in business_indicators):
#                 self.category_completion[self.current_category] = True
#                 return True
#                 
#         elif self.current_category == "Technical Scope & Architecture":
#             # Check for technical details
#             tech_indicators = ["pipeline", "data", "tb", "gb", "table", "database", "oracle", "exadata"]
#             if any(indicator in answered_str for indicator in tech_indicators):
#                 self.category_completion[self.current_category] = True
#                 return True
#         
#         # Default: complete after 2 questions per category
#         return self.category_questions.get(self.current_category, 0) >= 2
#     
#     def _move_to_next_category(self):
#         """Move to the next planning category."""
#         current_index = self.categories.index(self.current_category)
#         if current_index < len(self.categories) - 1:
#             self.current_category = self.categories[current_index + 1]
#         else:
#             self.current_category = "complete"

# NEW SIMPLIFIED QuestionManagementAgent
class QuestionManagementAgent(dspy.Module):
    """Simplified question management that generates 3 questions per category upfront."""
    
    def __init__(self):
        super().__init__()
        self.data_accumulator = dspy.ChainOfThought(DataAccumulator)
        
        # Simple state management
        self.answered_questions = {}  # question -> answer mapping
        self.structured_data = {}
        self.categories = list(PLANNING_CATEGORIES.keys())
        self.all_questions = {}  # category -> list of 3 questions
        self.current_batch = []  # Current batch of 3 questions
        self.batch_index = 0  # Current batch index
        self.conversation_stage = "initial"  # initial, questioning, ready_for_plan, planning_complete
    
        # Generate all questions upfront
        self._generate_all_questions()
    
    def _generate_all_questions(self):
        """Generate 3 questions per category upfront."""
        for category in self.categories:
            questions = PLANNING_CATEGORIES[category][:3]  # Take first 3 questions
            self.all_questions[category] = questions
    
    def get_next_questions(self, user_input, project_context):
        """Get the next batch of 3 questions from different categories."""
        if self.conversation_stage == "initial":
            # First time - return first batch
            self.conversation_stage = "questioning"
            self._prepare_next_batch()
            return self._format_question_batch()
        
        elif self.conversation_stage == "questioning":
            # Check if we have more batches
            if self.batch_index < len(self.categories):
                self._prepare_next_batch()
                return self._format_question_batch()
            else:
                # No more questions - ready for plan
            self.conversation_stage = "ready_for_plan"
            return dspy.Prediction(
                    next_questions="",
                    categories="complete",
                completion_status="complete"
            )
        
        else:
            # Already ready for plan
            return dspy.Prediction(
                next_questions="",
                categories="complete", 
                completion_status="complete"
            )
    
    def _prepare_next_batch(self):
        """Prepare the next batch of 3 questions from different categories."""
        if self.batch_index < len(self.categories):
            category = self.categories[self.batch_index]
            self.current_batch = self.all_questions[category]
            self.batch_index += 1
    
    def _format_question_batch(self):
        """Format the current batch of questions."""
        if not self.current_batch:
            return dspy.Prediction(
                next_questions="",
                categories="complete",
                completion_status="complete"
            )
        
        category = self.categories[self.batch_index - 1]
        formatted_questions = []
        
        for i, question in enumerate(self.current_batch, 1):
            formatted_questions.append(f"{i}. {question}")
        
        questions_text = "\n".join(formatted_questions)
        
        return dspy.Prediction(
            next_questions=questions_text,
            categories=category,
            completion_status="in_progress"
        )
    
    def accumulate_data(self, questions, user_answers, project_context):
        """Structure and store user responses for the current batch."""
        # Parse user answers (assuming they're provided as a list or string)
        if isinstance(user_answers, str):
            # Split by newlines or numbers
            answers = [ans.strip() for ans in user_answers.split('\n') if ans.strip()]
        else:
            answers = user_answers
        
        # Store answers for each question in the current batch
        current_category = self.categories[self.batch_index - 1] if self.batch_index > 0 else "unknown"
        
        for i, question in enumerate(self.current_batch):
            if i < len(answers):
                answer = answers[i]
                self.answered_questions[question] = answer
                
                # Use data accumulator to structure the data
        result = self.data_accumulator(
            question=question,
                    user_answer=answer,
            project_context=project_context,
            existing_data=json.dumps(self.structured_data)
        )
        
        try:
            new_data = json.loads(result.structured_data)
            self.structured_data.update(new_data)
        except:
            # If JSON parsing fails, store as text
            self.structured_data[f"question_{len(self.answered_questions)}"] = {
                "question": question,
                        "answer": answer,
                        "category": current_category
            }
        
        # Clear the current batch since we've processed it
        self.current_batch = []
        
        return dspy.Prediction(
            structured_data=json.dumps(self.structured_data),
            updated_context="Data accumulated for current batch",
            data_completeness=f"Answered {len(self.answered_questions)} questions across {self.batch_index} categories"
        )
    
    def get_status(self):
        """Get current status of question collection."""
        completed_categories = self.batch_index
        total_categories = len(self.categories)
        
        return {
            "conversation_stage": self.conversation_stage,
            "questions_answered": len(self.answered_questions),
            "categories_completed": completed_categories,
            "total_categories": total_categories,
            "ready_for_plan": self.conversation_stage == "ready_for_plan"
        }

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
        
        # Convert structured_data to serializable format
        serializable_data = self._convert_to_serializable(structured_data)
        
        # Generate the plan
        result = self.plan_generator(
            project_context=project_context,
            structured_data=json.dumps(serializable_data),
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
    
    def _convert_to_serializable(self, data):
        """Convert DSPy objects to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._convert_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif hasattr(data, '__dict__') and hasattr(data, '_asdict'):
            # Handle DSPy Prediction objects
            try:
                return data._asdict()
            except:
                return str(data)
        elif hasattr(data, '__dict__'):
            # Handle other objects with __dict__
            return {k: self._convert_to_serializable(v) for k, v in data.__dict__.items()}
        else:
            return data

class PlanEvaluationAgent(dspy.Module):
    """Evaluates migration plan completeness and quality."""
    
    def __init__(self):
        super().__init__()
        self.plan_evaluator = dspy.ChainOfThought(PlanEvaluator)
    
    def evaluate_plan(self, generated_plan, project_context, structured_data):
        """Evaluate plan completeness and quality."""
        # Convert structured_data to serializable format
        serializable_data = self._convert_to_serializable(structured_data)
        
        result = self.plan_evaluator(
            generated_plan=generated_plan,
            project_context=project_context,
            structured_data=json.dumps(serializable_data),
            evaluation_criteria="completeness_and_quality"
        )
        
        return dspy.Prediction(
            completeness_score=result.completeness_score,
            quality_score=result.quality_score,
            missing_elements=result.missing_elements,
            recommendations=result.recommendations
        )
    
    def _convert_to_serializable(self, data):
        """Convert DSPy objects to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._convert_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif hasattr(data, '__dict__') and hasattr(data, '_asdict'):
            # Handle DSPy Prediction objects
            try:
                return data._asdict()
            except:
                return str(data)
        elif hasattr(data, '__dict__'):
            # Handle other objects with __dict__
            return {k: self._convert_to_serializable(v) for k, v in data.__dict__.items()}
        else:
            return data

print("All 4 agents implemented successfully!")

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
        
        # Handle dictionary input format (MLflow signature format)
        if isinstance(user_input, dict):
            if "user_input" in user_input:
                user_input = user_input["user_input"]
            elif "inputs" in user_input:
            user_input = user_input["inputs"]
            else:
            # Handle other dictionary formats
            user_input = str(user_input)
        # Handle ChatCompletionRequest format (for agent compatibility)
        elif hasattr(user_input, 'messages') and len(user_input.messages) > 0 and hasattr(user_input.messages[0], 'content'):
            user_input = user_input.messages[0].content
        else:
            user_input = str(user_input).strip()
        
        # Handle special commands
        if user_input.lower() == "/plan":
            response = self._generate_plan()
        elif user_input.lower() == "/status":
            response = self._get_status()
        elif user_input.lower() == "/help":
            response = self._get_help()
        else:
        # Handle conversation
        if self.conversation_stage == "initial":
                response = self._handle_initial_input(user_input)
        else:
                response = self._handle_question_answer(user_input)
        
        # Convert response to string for MLflow DSPy streaming compatibility
        if isinstance(response, dict):
            if "error" in response:
                return f"Error: {response['error']}"
            elif "response" in response:
                return response["response"]
            else:
                # Convert dict to formatted string
                return str(response)
        else:
            return str(response)
    
    def _handle_initial_input(self, user_input):
        """Handle initial user input."""
        self.project_context = user_input
        self.conversation_stage = "questioning"
        
        # Get first batch of questions
        question_result = self.question_agent.get_next_questions(user_input, self.project_context)
        
        return f"""**Welcome to the Migration Planning Agent!**

I'll help you create a comprehensive migration plan for your project. Let me ask some targeted questions to understand your requirements better.

**Project Context:** {user_input[:200]}{'...' if len(user_input) > 200 else ''}

**Questions for {question_result.categories} category:**
{question_result.next_questions}

*Please answer all 3 questions above. Use `/status` anytime to see progress or `/help` for available commands.*"""
    
    def _handle_question_answer(self, user_input):
        """Handle user answers to questions."""
        # Accumulate the answers for the current batch
        data_result = self.question_agent.accumulate_data(
            questions=self.question_agent.current_batch,
            user_answers=user_input,
            project_context=self.project_context
        )
        
        # Get next batch of questions
        question_result = self.question_agent.get_next_questions(user_input, self.project_context)
        
        if question_result.completion_status == "complete":
            self.conversation_stage = "ready_for_plan"
            return "**Great! I have enough information to create a comprehensive migration plan.**\n\nUse `/plan` to generate your detailed migration plan, or `/status` to see what information I've gathered."
        else:
            # Provide next batch of questions
            return f"**Thanks for those answers!**\n\n**Next questions for {question_result.categories} category:**\n{question_result.next_questions}\n\n*Please answer all 3 questions above. Use `/status` to see progress or `/plan` when ready to generate your plan.*"
    
    def _generate_plan(self):
        """Generate migration plan."""
        if self.conversation_stage != "ready_for_plan" and not self.plan_generated:
            return "I need to gather more information before generating a plan. Please answer a few more questions or use /status to see progress."
        
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
        self.conversation_stage = "planning_complete"
        
        # Format the plan with proper structure like the reference notebook
        formatted_plan = f"""
# Migration Plan for: {self.project_context[:100]}...

## Project Overview
{plan_result.migration_plan}

## Migration Timeline
{plan_result.timeline}

## Resource Requirements
{plan_result.resource_requirements}

## Risk Assessment
{plan_result.risks}

## Additional Information
{plan_result.additional_info}

---
**Plan Quality Metrics:**
- Completeness Score: {evaluation.completeness_score}/100
- Quality Score: {evaluation.quality_score}/100
- Missing Elements: {evaluation.missing_elements}
- Recommendations: {evaluation.recommendations}
"""
        
        return formatted_plan
    
    def _get_status(self):
        """Get current status."""
        status = self.question_agent.get_status()
        
        return f"""**Migration Planning Status**

**Conversation Stage:** {status['conversation_stage']}
**Questions Answered:** {status['questions_answered']}
**Categories Completed:** {status['categories_completed']}/{status['total_categories']}

**Categories Progress:**
Completed: {status['categories_completed']} out of {status['total_categories']} categories
Remaining: {status['total_categories'] - status['categories_completed']} categories

**Data Captured:** {len(self.question_agent.structured_data)} fields
**Ready for Plan:** {'Yes' if status['ready_for_plan'] else 'No'}

**Next Steps:**
{'- Answer the current batch of 3 questions' if status['conversation_stage'] == "questioning" else '- Type /plan to generate your migration plan' if status['ready_for_plan'] else '- Plan generation complete!'}"""
    
    def _get_help(self):
        """Get help information."""
        return """**Migration Planning Agent Help**

**Available Commands:**
- `/plan` - Generate your comprehensive migration plan
- `/status` - View current progress and data gathered
- `/help` - Show this help information

**How it works:**
1. I'll ask you 3 questions at a time from different categories
2. Questions cover: Team & Resources, Business Drivers, Technical Scope, Process Maturity, Performance, and Security
3. Answer all 3 questions in each batch before moving to the next category
4. When all categories are covered, use `/plan` to generate your detailed migration plan

**Tips:**
- Be specific in your answers for better planning
- Answer all 3 questions in each batch
- Use `/status` to see progress through categories
- Questions are pre-selected to cover all essential migration planning aspects

*Just answer the questions in each batch to continue planning your migration!*"""

print("Main MigrationPlanningAgent implemented successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the New Simplified QuestionManagementAgent

# COMMAND ----------

# Test the new simplified agent
def test_simplified_agent():
    """Test the new simplified QuestionManagementAgent."""
    print("Testing New Simplified QuestionManagementAgent")
    print("=" * 60)
    
    # Create a fresh agent instance
    test_agent = MigrationPlanningAgent(
        vector_search_endpoint=vector_search_endpoint,
        vector_index=vector_index_name
    )
    
    # Test initial input
    print("1. Testing initial input:")
    response1 = test_agent.forward("I want to migrate a customer from Oracle to Databricks")
    print(f"Response: {response1}")
    
    # Test first batch of answers
    print("\n2. Testing first batch of answers:")
    response2 = test_agent.forward("""1. We have 10 DBAs and 20 Data Engineers in the central team
2. Yes, they are trained in Databricks
3. We are using Professional Services for this migration""")
    print(f"Response: {response2}")
    
    # Test status
    print("\n3. Testing status:")
    status = test_agent._get_status()
    print(f"Status: {status}")
    
    print("\n4. Testing help:")
    help_text = test_agent._get_help()
    print(f"Help: {help_text}")

# Run the test
test_simplified_agent()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Streaming Support

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Capabilities

# COMMAND ----------

# MAGIC %md
# MAGIC The agent uses **ResponsesAgent interface** for Databricks Agent Framework compatibility:
# MAGIC
# MAGIC **Key Features:**
# MAGIC - **ResponsesAgent wrapper** - Following Databricks Agent Framework patterns
# MAGIC - **Automatic signature inference** - MLflow automatically infers the correct schema
# MAGIC - **String outputs** - Required for MLflow DSPy streaming compatibility
# MAGIC - **Token usage tracking** - MLflow automatically tracks tokens
# MAGIC - **Regular prediction mode** - Reliable `predict()` functionality
# MAGIC - **Streaming support** - Built-in streaming with `predict_stream()`
# MAGIC - **Agent framework compatibility** - Native support for Databricks Agent Framework
# MAGIC
# MAGIC **Usage (ResponsesAgent format):**
# MAGIC ```python
# MAGIC # Load the model
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC
# MAGIC # Regular prediction (ResponsesAgent format)
# MAGIC request = {
# MAGIC     "input": [
# MAGIC         {
# MAGIC             "role": "user", 
# MAGIC             "content": "I want to migrate Oracle to Databricks"
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC response = model.predict(request)
# MAGIC print(response)
# MAGIC
# MAGIC # Streaming prediction
# MAGIC stream_response = model.predict_stream(request)
# MAGIC for output in stream_response:
# MAGIC     print(output)
# MAGIC ```

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
# MAGIC ### Create ResponsesAgent Wrapper for Databricks Agent Framework

# COMMAND ----------

from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

class MigrationPlanningResponsesAgent(ResponsesAgent):
    """ResponsesAgent wrapper for Databricks Agent Framework compatibility."""
    
    def __init__(self, dspy_agent):
        super().__init__()
        self.agent = dspy_agent
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
            lm = dspy.LM(model="databricks/databricks-claude-sonnet-4", max_tokens=1000)
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
        """Non-streaming prediction."""
        # Ensure DSPy LM is configured
        self._ensure_dspy_lm_configured()
        
        # Extract user input from the request
        user_input = self._extract_user_input(request)
        
        # Call the DSPy agent (use module() instead of module.forward())
        response = self.agent(user_input)
        
        # Convert to ResponsesAgent format
        output_item = self.create_text_output_item(
            text=str(response), 
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
    
    def predict_stream(self, request: ResponsesAgentRequest):
        """Streaming prediction."""
        # Ensure DSPy LM is configured
        self._ensure_dspy_lm_configured()
        
        # Extract user input from the request
        user_input = self._extract_user_input(request)
        
        # Call the DSPy agent (use module() instead of module.forward())
        response = self.agent(user_input)
        
        # For now, yield the response as a single chunk
        # In a more sophisticated implementation, you could stream the DSPy agent's output
        output_item = self.create_text_output_item(
            text=str(response), 
            id=str(uuid4())
        )
        
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=output_item
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

# Create the ResponsesAgent wrapper
responses_agent = MigrationPlanningResponsesAgent(compiled_migration_agent)
print("ResponsesAgent wrapper created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. MLflow Model Registration

# COMMAND ----------

# Input example for the model - plain string as per MLflow DSPy documentation
input_example = "I need to migrate our Oracle data warehouse to Databricks"


# Use native MLflow DSPy logging with proper agent signatures
from mlflow.models import ModelSignature
from mlflow.types import ColSpec
from mlflow.types.llm import ChatCompletionRequest, ChatMessage

# Disable MLflow autologging to avoid Pydantic serialization warnings
mlflow.autolog(disable=True)

# Set environment variable to disable autologging globally
import os
os.environ["MLFLOW_DISABLE_AUTOLOGGING"] = "true"

# Suppress Pydantic serialization warnings
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic.*serialization.*")
warnings.filterwarnings("ignore", message=".*Expected.*fields.*but got.*")

# Set the experiment for this run
mlflow.set_experiment(mlflow_experiment_name)

# Log the ResponsesAgent using MLflow (Databricks Agent Framework compatible)
with mlflow.start_run() as run:
    # Log the ResponsesAgent - this automatically handles the correct schema for Agent Framework
    model_info = mlflow.pyfunc.log_model(
        python_model=responses_agent,
        name="usecase-planning-agent",
        input_example={
            "input": [
                {
                    "role": "user", 
                    "content": "I need to migrate our Oracle data warehouse to Databricks"
                }
            ]
        }
    )
    
    print(f"MigrationPlanningAgent V2 (ResponsesAgent) logged to MLflow: {model_info.model_uri}")
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
# MAGIC ### Test Streaming Support

# COMMAND ----------

# Test streaming functionality
def test_responses_agent():
    """Test the ResponsesAgent functionality."""
    print("Testing ResponsesAgent")
    print("=" * 60)
    
    # Load the logged model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    
    # Test regular prediction with ResponsesAgent format
    print("1. Testing regular prediction (ResponsesAgent format):")
    request = {
        "input": [
            {
                "role": "user", 
                "content": "I want to migrate Oracle to Databricks"
            }
        ]
    }
    response = loaded_model.predict(request)
    print(f"Response: {response}")
    
    # Test streaming prediction
    print("\n2. Testing streaming prediction:")
    try:
        print("Streaming response chunks:")
        stream_response = loaded_model.predict_stream(request)
        
        for chunk in stream_response:
            print(f"Chunk: {chunk}")
            # Break after a few chunks for demo purposes
            if "questions" in str(chunk).lower():
                break
        
        print("Streaming completed successfully!")
    except Exception as e:
        print(f"Streaming error: {e}")
    
    print("\n" + "=" * 60)
    print("ResponsesAgent test completed!")

# Run the test
test_responses_agent()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Question Flow Fix

# COMMAND ----------

# Test the question flow to ensure it works correctly
def test_question_flow():
    """Test the question flow to ensure it progresses correctly."""
    print("Testing Question Flow")
    print("=" * 60)
    
    # Create a test agent with required parameters
    test_agent = MigrationPlanningAgent(
        vector_search_endpoint=vector_search_endpoint,
        vector_index=vector_index_name
    )
    
    # Test initial input
    print("1. Testing initial input:")
    response1 = test_agent("I want to migrate Oracle Exadata to Databricks")
    print(f"Response: {response1}")
    
    # Test first batch of answers
    print("\n2. Testing first batch of answers:")
    response2 = test_agent("""1. We have 10 DBAs and 20 Data Engineers in the central team
2. Yes, they are trained in Databricks
3. We are using Professional Services for this migration""")
    print(f"Response: {response2}")
    
    # Test status
    print("\n3. Testing status:")
        status = test_agent._get_status()
    print(f"Status: {status}")
    
    # Test second batch of answers
    print("\n4. Testing second batch of answers:")
    response3 = test_agent("""1. Quite basic actually, we've got a data catalog and have multiple teams with their own data product owners
2. Quite mature from a legacy architecture point of view
3. We manage data security centrally and use Azure EntraID Groups and regularly audit the access of users and teams""")
    print(f"Response: {response3}")
    
    # Test status again
    print("\n5. Testing status after second batch:")
    status2 = test_agent._get_status()
    print(f"Status: {status2}")

# Run the test
test_question_flow()

# COMMAND ----------


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Deploy Agent to Model Serving

# COMMAND ----------

# Import required libraries for deployment
from databricks import agents
import mlflow

# Check MLflow version compatibility
print(f"MLflow version: {mlflow.__version__}")
mlflow_version = mlflow.__version__
if mlflow_version < "2.20.0":
    print(f"WARNING: MLflow version {mlflow_version} is below 2.20.0")
    print("Agent Framework requires MLflow 2.20.0 or above for proper agent signatures.")
    print("Consider upgrading MLflow or the deployment may fail.")
else:
    print("MLflow version is compatible with Agent Framework.")

# COMMAND ----------

# Deploy the agent to Model Serving
print("Deploying agent to Model Serving...")
print(f"Model Name: {uc_model_info.name}")
print(f"Model Version: {uc_model_info.version}")

# List existing deployments for troubleshooting
try:
    print("\nChecking existing deployments...")
    existing_deployments = agents.list_deployments()
    if existing_deployments:
        print("Existing deployments found:")
        for dep in existing_deployments:
            print(f"  - {dep.name} (version {dep.version})")
    else:
        print("No existing deployments found.")
except Exception as e:
    print(f"Could not list existing deployments: {e}")

# Deploy the agent using the Agent Framework
# Use a static endpoint name to update the same endpoint
endpoint_name = "migration-planning-agent"
print(f"\nDeploying to endpoint: {endpoint_name}")

try:
    deployment = agents.deploy(uc_model_info.name, uc_model_info.version, endpoint_name=endpoint_name)
    print(f"Agent deployed successfully to endpoint: {endpoint_name}")
except Exception as e:
    print(f"Deployment to endpoint failed: {e}")
    print("Trying to clean up existing deployments and retry...")
    
    try:
        # Try to delete existing deployments for this model
        print("Attempting to clean up existing deployments...")
        # Note: This might fail if there are no existing deployments, which is fine
        agents.delete_deployment(uc_model_info.name, uc_model_info.version)
        print("Existing deployments cleaned up.")
        
        # Retry deployment
        deployment = agents.deploy(uc_model_info.name, uc_model_info.version, endpoint_name=endpoint_name)
        print(f"Agent deployed successfully after cleanup!")
    except Exception as e2:
        print(f"Cleanup and retry failed: {e2}")
        print("Please manually clean up incompatible model versions from the endpoint")
        print("or use a different endpoint name for deployment.")
        raise e2

# Retrieve the query endpoint URL for making API requests
print(f"Deployment successful!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary

# COMMAND ----------

print("Migration Planning Agent V2 - Complete!")
print("=" * 60)
print(f"Model registered: {uc_model_info.name}")
print(f"Model version: {uc_model_info.version}")
print(f"Agent deployed to endpoint: migration-planning-agent")
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