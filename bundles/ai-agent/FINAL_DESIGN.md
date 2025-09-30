# Final Migration Planning Agent Design

## Core Flow
1. **User Input** → System asks structured questions from past project plans
2. **Knowledge Accumulation** → Maintain structured data of user responses
3. **Progressive Questioning** → Track what's asked/answered efficiently
4. **Plan Generation** → Vector search for each plan section + generate structured plan
5. **Plan Evaluation** → Rate completeness and quality (0-100)

## Required Agents & Signatures

### Agent 1: **Question Management Agent**
**Purpose**: Manages question flow, tracks progress, maintains structured data

**Signatures (2):**
1. **`QuestionSelector`** - Choose next question based on progress
   - Input: `project_context`, `answered_questions`, `current_category`, `conversation_stage`
   - Output: `next_question`, `category`, `priority`, `completion_status`

2. **`DataAccumulator`** - Structure and store user responses
   - Input: `question`, `user_answer`, `project_context`, `existing_data`
   - Output: `structured_data`, `updated_context`, `data_completeness`

### Agent 2: **Knowledge Retrieval Agent**
**Purpose**: Search vector DB for relevant migration documents and references

**Signatures (2):**
1. **`DocumentSearcher`** - Search for relevant documents
   - Input: `query`, `project_context`, `search_type`
   - Output: `relevant_documents`, `search_confidence`, `document_categories`

2. **`ReferenceSummarizer`** - Summarize documents for plan sections
   - Input: `documents`, `plan_section`, `project_context`
   - Output: `summarized_references`, `key_insights`, `best_practices`

### Agent 3: **Plan Generation Agent**
**Purpose**: Generate structured migration plan using accumulated data + references

**Signatures (2):**
1. **`PlanGenerator`** - Create structured migration plan
   - Input: `project_context`, `structured_data`, `references`, `plan_sections`
   - Output: `migration_plan`, `timeline`, `resource_requirements`, `risks`

2. **`PlanFormatter`** - Format plan into tabular structure
   - Input: `migration_plan`, `format_requirements`
   - Output: `formatted_plan`, `tables`, `additional_info`

### Agent 4: **Plan Evaluation Agent**
**Purpose**: Rate plan completeness and quality

**Signatures (1):**
1. **`PlanEvaluator`** - Evaluate plan quality and completeness
   - Input: `generated_plan`, `project_context`, `structured_data`, `evaluation_criteria`
   - Output: `completeness_score`, `quality_score`, `missing_elements`, `recommendations`

## Detailed Agent Structure

```python
class QuestionManagementAgent(dspy.Module):
    def __init__(self):
        self.question_selector = dspy.ChainOfThought(QuestionSelector)
        self.data_accumulator = dspy.ChainOfThought(DataAccumulator)
        
        # State management
        self.answered_questions = {}
        self.categories_progress = {}
        self.structured_data = {}
        self.current_category = "Resource & Team"
        
        # Use existing PLANNING_CATEGORIES
        self.categories = PLANNING_CATEGORIES
    
    def get_next_question(self, user_input, project_context):
        # Select next question based on progress
        result = self.question_selector(
            project_context=project_context,
            answered_questions=self.answered_questions,
            current_category=self.current_category,
            conversation_stage=self._get_conversation_stage()
        )
        
        return dspy.Prediction(
            next_question=result.next_question,
            category=result.category,
            priority=result.priority,
            completion_status=result.completion_status
        )
    
    def accumulate_data(self, question, user_answer, project_context):
        # Structure and store user response
        result = self.data_accumulator(
            question=question,
            user_answer=user_answer,
            project_context=project_context,
            existing_data=self.structured_data
        )
        
        # Update state
        self.answered_questions[question] = user_answer
        self.structured_data.update(result.structured_data)
        
        return dspy.Prediction(
            structured_data=result.structured_data,
            updated_context=result.updated_context,
            data_completeness=result.data_completeness
        )

class KnowledgeRetrievalAgent(dspy.Module):
    def __init__(self, vector_search_endpoint, vector_index):
        self.document_searcher = dspy.ChainOfThought(DocumentSearcher)
        self.reference_summarizer = dspy.ChainOfThought(ReferenceSummarizer)
        self.vector_search = VectorSearchClient()
        self.endpoint = vector_search_endpoint
        self.index = vector_index
    
    def search_documents(self, query, project_context, search_type="general"):
        # Search vector database
        search_results = self.vector_search.get_index(
            endpoint_name=self.endpoint,
            index_name=self.index
        ).query(
            query_text=query,
            columns=["content", "title", "category", "source"],
            num_results=5
        )
        
        result = self.document_searcher(
            query=query,
            project_context=project_context,
            search_type=search_type
        )
        
        return dspy.Prediction(
            relevant_documents=search_results,
            search_confidence=result.search_confidence,
            document_categories=result.document_categories
        )
    
    def summarize_for_plan_section(self, documents, plan_section, project_context):
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
    def __init__(self, knowledge_agent):
        self.plan_generator = dspy.ChainOfThought(PlanGenerator)
        self.plan_formatter = dspy.ChainOfThought(PlanFormatter)
        self.knowledge_agent = knowledge_agent
    
    def generate_plan(self, project_context, structured_data):
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
            structured_data=structured_data,
            references=references,
            plan_sections=plan_sections
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
    def __init__(self):
        self.plan_evaluator = dspy.ChainOfThought(PlanEvaluator)
    
    def evaluate_plan(self, generated_plan, project_context, structured_data):
        result = self.plan_evaluator(
            generated_plan=generated_plan,
            project_context=project_context,
            structured_data=structured_data,
            evaluation_criteria="completeness_and_quality"
        )
        
        return dspy.Prediction(
            completeness_score=result.completeness_score,
            quality_score=result.quality_score,
            missing_elements=result.missing_elements,
            recommendations=result.recommendations
        )

class MigrationPlanningAgent(dspy.Module):
    def __init__(self, vector_search_endpoint, vector_index):
        self.question_agent = QuestionManagementAgent()
        self.knowledge_agent = KnowledgeRetrievalAgent(vector_search_endpoint, vector_index)
        self.plan_agent = PlanGenerationAgent(self.knowledge_agent)
        self.evaluation_agent = PlanEvaluationAgent()
        
        # State
        self.project_context = ""
        self.conversation_stage = "initial"
        self.plan_generated = False
    
    def forward(self, user_input):
        user_input = user_input.strip()
        
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
        # Extract project context
        self.project_context = user_input
        self.conversation_stage = "questioning"
        
        # Get first question
        question_result = self.question_agent.get_next_question(user_input, self.project_context)
        
        return dspy.Prediction(
            response=f"I'll help you plan this migration. Let me ask some questions to understand your project better.\n\n{question_result.next_question}",
            next_action="waiting_for_answer",
            current_category=question_result.category,
            completion_status=question_result.completion_status
        )
    
    def _handle_question_answer(self, user_input):
        # Accumulate the answer
        data_result = self.question_agent.accumulate_data(
            question=self.question_agent.current_question,
            user_answer=user_input,
            project_context=self.project_context
        )
        
        # Get next question
        question_result = self.question_agent.get_next_question(user_input, self.project_context)
        
        if question_result.completion_status == "complete":
            return dspy.Prediction(
                response=f"Thanks! I have enough information. Use /plan to generate your migration plan.",
                next_action="ready_for_plan",
                completion_status="complete"
            )
        else:
            return dspy.Prediction(
                response=f"Thanks! {question_result.next_question}",
                next_action="waiting_for_answer",
                current_category=question_result.category,
                completion_status=question_result.completion_status
            )
    
    def _generate_plan(self):
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
        
        return dspy.Prediction(
            response=f"Here's your migration plan:\n\n{plan_result.formatted_plan}",
            migration_plan=plan_result.migration_plan,
            completeness_score=evaluation.completeness_score,
            quality_score=evaluation.quality_score,
            missing_elements=evaluation.missing_elements,
            recommendations=evaluation.recommendations
        )
```

## Total Architecture Summary

**4 Agents, 7 Signatures:**
- **Question Management Agent**: 2 signatures
- **Knowledge Retrieval Agent**: 2 signatures  
- **Plan Generation Agent**: 2 signatures
- **Plan Evaluation Agent**: 1 signature

**Flow:**
1. User input → Question Management → Structured questions
2. User answers → Data Accumulation → Structured data
3. /plan command → Knowledge Retrieval → References → Plan Generation → Evaluation
4. Response with plan + scores + recommendations

This design is much cleaner, follows your exact flow, and leverages existing assets while being maintainable and scalable.
