import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum

# Core LangGraph imports (minimal dependencies)
from langgraph.graph import StateGraph, END

# Your existing imports
from llm.openai_client import get_azure_client, chat_completion
from prompts.code_generation_prompt import generate_prompt_code
from prompts.rule_validation_prompt import generate_prompt_rule
from code_generation import rule_evaluate_batch
from utils.utils import load_data, get_calculated_value

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    # Input data
    target_field: str
    client_name: str
    tdd_document: str
    source_schema: Dict[str, Any]
    cbms_data: Dict[str, Any]
    
    # Processing state
    current_step: str
    iteration_count: int
    max_iterations: int
    
    # Generated outputs
    tdd_analysis: Optional[Dict[str, Any]]
    business_rules: Optional[Dict[str, Any]]
    generated_code: Optional[str]
    validation_results: Optional[Dict[str, Any]]
    improvement_suggestions: Optional[List[Dict[str, Any]]]
    
    # Messages and logs
    messages: List[str]
    error_log: List[str]
    
    # Final outputs
    final_rules: Optional[Dict[str, Any]]
    final_code: Optional[str]
    confidence_score: Optional[float]
    
    # Additional context
    client_to_id: Optional[Dict[str, Any]]
    client_to_conformity_csv: Optional[Dict[str, Any]]
    txt_files: Optional[List[str]]
    parquet_dir: Optional[str]
    error_log_dir: Optional[str]
    output_dir: Optional[str]
    table_cache: Optional[Dict[str, Any]]

class ProcessingStep(Enum):
    TDD_ANALYSIS = "tdd_analysis"
    RULE_GENERATION = "rule_generation"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    IMPROVEMENT = "improvement"
    FINALIZATION = "finalization"

@dataclass
class AgentConfig:
    """Configuration for the LangGraph agent"""
    max_iterations: int = 3
    confidence_threshold: float = 0.85
    enable_auto_improvement: bool = True
    api_version: str = "2024-02-15-preview"
    max_tokens: int = 10000
    validation_sample_size: int = 10  # Number of records to validate

class SimpleRuleAgent:
    """Simplified LangGraph-based agent for rule generation and validation"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.client = get_azure_client()
        self.logger = logging.getLogger(__name__)
        
        # Build and compile the graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each processing step
        workflow.add_node("analyze_tdd", self._analyze_tdd)
        workflow.add_node("generate_rules", self._generate_rules)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("validate_solution", self._validate_solution)
        workflow.add_node("improve_solution", self._improve_solution)
        workflow.add_node("finalize_results", self._finalize_results)
        
        # Define the workflow edges
        workflow.set_entry_point("analyze_tdd")
        
        # Linear flow: TDD → Rules → Code → Validation
        workflow.add_edge("analyze_tdd", "generate_rules")
        workflow.add_edge("generate_rules", "generate_code")
        workflow.add_edge("generate_code", "validate_solution")
        
        # Conditional: Validation → Improve OR Finalize
        workflow.add_conditional_edges(
            "validate_solution",
            self._decide_next_step,
            {
                "improve": "improve_solution",
                "finalize": "finalize_results"
            }
        )
        
        # Conditional: Improve → Rules (iterate) OR Finalize
        workflow.add_conditional_edges(
            "improve_solution", 
            self._decide_iteration,
            {
                "iterate": "generate_rules",
                "finalize": "finalize_results"
            }
        )
        
        # End workflow
        workflow.add_edge("finalize_results", END)
        
        return workflow
    
    def _analyze_tdd(self, state: AgentState) -> AgentState:
        """Step 1: Analyze TDD document for relevant information"""
        self.logger.info(f"[TDD_ANALYSIS] Analyzing TDD for field: {state['target_field']}")
        
        tdd_prompt = f"""
        Analyze the Technical Design Document for field transformation requirements.
        
        TARGET FIELD: {state['target_field']}
        CLIENT: {state['client_name']}
        
        TDD DOCUMENT:
        {state['tdd_document'][:5000]}  # Limit for token efficiency
        
        Extract and return JSON with:
        {{
            "relevant_sections": ["section numbers/names that mention the target field"],
            "business_logic": "detailed description of transformation logic",
            "dependencies": ["list of fields this transformation depends on"],
            "validation_rules": ["list of validation requirements"],
            "edge_cases": ["special conditions or exceptions"],
            "complexity": "simple|moderate|complex",
            "data_sources": ["source systems or tables mentioned"]
        }}
        
        Focus only on information relevant to {state['target_field']}.
        """
        
        try:
            response = chat_completion(
                client=self.client,
                deployment=self.config.api_version,
                user_question=tdd_prompt,
                max_tokens=self.config.max_tokens
            )
            
            # Clean and parse response
            clean_response = response.replace("```json", "").replace("```", "").strip()
            tdd_analysis = json.loads(clean_response)
            
            state["tdd_analysis"] = tdd_analysis
            state["current_step"] = ProcessingStep.TDD_ANALYSIS.value
            state["messages"].append(f"✅ TDD analysis completed for {state['target_field']}")
            
        except Exception as e:
            error_msg = f"❌ TDD analysis failed: {str(e)}"
            state["error_log"].append(error_msg)
            state["messages"].append(error_msg)
            self.logger.error(error_msg)
            
            # Provide fallback analysis
            state["tdd_analysis"] = {
                "business_logic": f"Transform {state['target_field']} according to standard mapping rules",
                "complexity": "moderate",
                "dependencies": [],
                "validation_rules": [],
                "edge_cases": []
            }
        
        return state
    
    def _generate_rules(self, state: AgentState) -> AgentState:
        """Step 2: Generate business rules based on TDD analysis"""
        self.logger.info(f"[RULE_GENERATION] Iteration {state.get('iteration_count', 0) + 1}")
        
        # Get base rule prompt
        base_prompt = generate_prompt_rule(state['target_field'])
        
        # Enhance with TDD context and previous feedback
        enhanced_prompt = f"""
        {base_prompt}
        
        TDD ANALYSIS CONTEXT:
        {json.dumps(state.get('tdd_analysis', {}), indent=2)}
        
        PREVIOUS ITERATION FEEDBACK:
        {json.dumps(state.get('improvement_suggestions', {}), indent=2) if state.get('improvement_suggestions') else 'None - First iteration'}
        
        GENERATE BUSINESS RULES following Map2Common standards for field: {state['target_field']}
        
        Return JSON format:
        {{
            "target_field": "{state['target_field']}",
            "business_rule": "Detailed explanation of transformation logic",
            "input_fields": ["list", "of", "source", "fields"],
            "transformation_steps": [
                "Step 1: description",
                "Step 2: description"
            ],
            "validation_criteria": ["criteria1", "criteria2"],
            "error_handling": ["error condition 1", "error condition 2"]
        }}
        """
        
        try:
            response = chat_completion(
                client=self.client,
                deployment=self.config.api_version,
                user_question=enhanced_prompt,
                max_tokens=self.config.max_tokens
            )
            
            clean_response = response.replace("```json", "").replace("```", "").strip()
            business_rules = json.loads(clean_response)
            
            state["business_rules"] = business_rules
            state["current_step"] = ProcessingStep.RULE_GENERATION.value
            state["messages"].append(f"✅ Business rules generated for {state['target_field']}")
            
        except Exception as e:
            error_msg = f"❌ Rule generation failed: {str(e)}"
            state["error_log"].append(error_msg)
            state["messages"].append(error_msg)
            self.logger.error(error_msg)
        
        return state
    
    def _generate_code(self, state: AgentState) -> AgentState:
        """Step 3: Generate Python transformation code"""
        self.logger.info(f"[CODE_GENERATION] Generating code for: {state['target_field']}")
        
        # Prepare inputs for code generation prompt
        business_rule_str = json.dumps(state.get('business_rules', {}), indent=2)
        fields_used_from_cbms = state.get('cbms_data', {})
        input_cols = str(list(state.get('source_schema', {}).keys()))
        
        # Get base code prompt
        code_prompt = generate_prompt_code(
            business_rule=business_rule_str,
            fields_used_from_cbms=fields_used_from_cbms,
            input_cols=input_cols
        )
        
        # Add improvement context if available
        improvement_context = ""
        if state.get('improvement_suggestions'):
            improvement_context = f"""
            
            PREVIOUS CODE ISSUES TO ADDRESS:
            {json.dumps(state.get('improvement_suggestions', {}), indent=2)}
            
            Please fix the above issues in the new code generation.
            """
        
        enhanced_code_prompt = f"""
        {code_prompt}
        {improvement_context}
        
        Generate production-ready Python code for field: {state['target_field']}
        
        Requirements:
        1. Follow the business rules exactly
        2. Include comprehensive error handling
        3. Add detailed logging statements
        4. Handle edge cases mentioned in TDD analysis
        5. Return code in JSON format with 'python_code' key
        """
        
        try:
            response = chat_completion(
                client=self.client,
                deployment=self.config.api_version,
                user_question=enhanced_code_prompt,
                system_prompt="You are a Python code generation expert specializing in data transformation.",
                max_tokens=self.config.max_tokens,
                prefix="TDD is: "
            )
            
            # Parse response and extract code
            clean_response = response.replace("```json", "").replace("```", "").strip()
            code_result = json.loads(clean_response)
            generated_code = code_result.get("python_code", "")
            
            state["generated_code"] = generated_code
            state["current_step"] = ProcessingStep.CODE_GENERATION.value
            state["messages"].append(f"✅ Python code generated for {state['target_field']}")
            
        except Exception as e:
            error_msg = f"❌ Code generation failed: {str(e)}"
            state["error_log"].append(error_msg)
            state["messages"].append(error_msg)
            self.logger.error(error_msg)
        
        return state
    
    def _validate_solution(self, state: AgentState) -> AgentState:
        """Step 4: Validate the generated solution"""
        self.logger.info(f"[VALIDATION] Validating solution for: {state['target_field']}")
        
        try:
            # Write generated code to file
            code_file = f"./transformation_codes/{state['target_field']}.py"
            import os
            os.makedirs("./transformation_codes", exist_ok=True)
            
            with open(code_file, 'w') as f:
                f.write(state.get('generated_code', ''))
            
            # Load data using your existing utilities
            inmemory_data = load_data(
                client_name=state['client_name'],
                client_to_id=state.get('client_to_id', {}),
                txt_files=state.get('txt_files', []),
                client_to_json={},  # Add if needed
                client_to_conformity_csv=state.get('client_to_conformity_csv', {})
            )
            
            # Run evaluation with small sample for efficiency
            all_eval_dfs, all_error_dfs = rule_evaluate_batch(
                target_fields=[state['target_field']],
                inmemory_data=inmemory_data,
                n_random=self.config.validation_sample_size,
                parquet_dir=state.get('parquet_dir', './data'),
                error_log_dir=state.get('error_log_dir', './errors'), 
                output_dir=state.get('output_dir', './outputs'),
                table_cache=state.get('table_cache', {})
            )
            
            # Calculate validation metrics
            validation_results = self._calculate_metrics(all_eval_dfs, all_error_dfs, state['target_field'])
            
            state["validation_results"] = validation_results
            state["current_step"] = ProcessingStep.VALIDATION.value
            
            accuracy = validation_results.get('accuracy', 0)
            state["messages"].append(f"✅ Validation completed. Accuracy: {accuracy:.1%}")
            
        except Exception as e:
            error_msg = f"❌ Validation failed: {str(e)}"
            state["error_log"].append(error_msg)
            state["messages"].append(error_msg)
            self.logger.error(error_msg)
            
            # Set default validation results
            state["validation_results"] = {
                "accuracy": 0.0,
                "total_records": 0,
                "error_count": 1,
                "validation_error": str(e)
            }
        
        return state
    
    def _improve_solution(self, state: AgentState) -> AgentState:
        """Step 5: Analyze results and suggest improvements"""
        self.logger.info(f"[IMPROVEMENT] Analyzing issues for iteration {state.get('iteration_count', 0) + 1}")
        
        validation_results = state.get('validation_results', {})
        current_rules = state.get('business_rules', {})
        current_code = state.get('generated_code', '')
        
        improvement_prompt = f"""
        Analyze the validation results and provide specific improvements for {state['target_field']}.
        
        CURRENT PERFORMANCE:
        - Accuracy: {validation_results.get('accuracy', 0):.1%}
        - Total Records: {validation_results.get('total_records', 0)}
        - Errors: {validation_results.get('error_count', 0)}
        
        CURRENT BUSINESS RULES:
        {json.dumps(current_rules, indent=2)}
        
        VALIDATION DETAILS:
        {json.dumps(validation_results, indent=2)}
        
        ANALYZE and provide specific improvements in JSON format:
        {{
            "issue_analysis": "What went wrong and why",
            "rule_improvements": [
                "Specific rule modification 1",
                "Specific rule modification 2"
            ],
            "code_improvements": [
                "Specific code fix 1", 
                "Specific code fix 2"
            ],
            "priority_fixes": ["Most critical issue to address"],
            "expected_impact": "Expected improvement percentage"
        }}
        
        Focus on actionable, specific improvements.
        """
        
        try:
            response = chat_completion(
                client=self.client,
                deployment=self.config.api_version,
                user_question=improvement_prompt,
                max_tokens=self.config.max_tokens
            )
            
            clean_response = response.replace("```json", "").replace("```", "").strip()
            improvements = json.loads(clean_response)
            
            state["improvement_suggestions"] = improvements
            state["current_step"] = ProcessingStep.IMPROVEMENT.value
            state["messages"].append(f"✅ Improvement analysis completed")
            
        except Exception as e:
            error_msg = f"❌ Improvement analysis failed: {str(e)}"
            state["error_log"].append(error_msg)
            state["messages"].append(error_msg)
            self.logger.error(error_msg)
        
        return state
    
    def _finalize_results(self, state: AgentState) -> AgentState:
        """Step 6: Finalize and package results"""
        self.logger.info(f"[FINALIZATION] Finalizing results for: {state['target_field']}")
        
        # Calculate final confidence score
        validation_results = state.get('validation_results', {})
        accuracy = validation_results.get('accuracy', 0)
        iteration_count = state.get('iteration_count', 0)
        
        # Penalize for too many iterations
        iteration_penalty = max(0, (iteration_count - 1) * 0.05)
        confidence_score = max(0, accuracy - iteration_penalty)
        
        # Package final results
        state["final_rules"] = state.get('business_rules', {})
        state["final_code"] = state.get('generated_code', '')
        state["confidence_score"] = confidence_score
        state["current_step"] = ProcessingStep.FINALIZATION.value
        
        final_message = f"🎯 Processing completed for {state['target_field']}. "
        final_message += f"Final confidence: {confidence_score:.1%} (Iterations: {iteration_count})"
        state["messages"].append(final_message)
        
        return state
    
    def _decide_next_step(self, state: AgentState) -> str:
        """Decision logic: improve or finalize based on validation results"""
        validation_results = state.get('validation_results', {})
        accuracy = validation_results.get('accuracy', 0)
        iteration_count = state.get('iteration_count', 0)
        
        # Check if we should try to improve
        should_improve = (
            accuracy < self.config.confidence_threshold and
            iteration_count < self.config.max_iterations and
            self.config.enable_auto_improvement and
            not validation_results.get('validation_error')  # Don't improve if there was a validation error
        )
        
        return "improve" if should_improve else "finalize"
    
    def _decide_iteration(self, state: AgentState) -> str:
        """Decision logic: iterate again or finalize"""
        iteration_count = state.get('iteration_count', 0)
        
        # Increment iteration count
        state["iteration_count"] = iteration_count + 1
        
        # Check if we should continue iterating
        if state["iteration_count"] < self.config.max_iterations:
            return "iterate"
        else:
            return "finalize"
    
    def _calculate_metrics(self, all_eval_dfs: Dict, all_error_dfs: Dict, target_field: str) -> Dict[str, Any]:
        """Calculate validation metrics from evaluation results"""
        try:
            eval_df = all_eval_dfs.get(target_field)
            error_df = all_error_dfs.get(target_field)
            
            if eval_df is None or eval_df.empty:
                return {
                    "accuracy": 0.0,
                    "total_records": 0,
                    "correct_predictions": 0,
                    "error_count": 0,
                    "status": "no_data"
                }
            
            total_records = len(eval_df)
            error_count = len(error_df) if error_df is not None and not error_df.empty else 0
            
            # Calculate accuracy (assuming rule_agent_value vs eagle_value comparison)
            if 'rule_agent_value' in eval_df.columns and 'eagle_value' in eval_df.columns:
                correct_predictions = (eval_df['rule_agent_value'] == eval_df['eagle_value']).sum()
                accuracy = correct_predictions / total_records if total_records > 0 else 0
            else:
                correct_predictions = 0
                accuracy = 0.0
            
            return {
                "accuracy": accuracy,
                "total_records": total_records,
                "correct_predictions": int(correct_predictions),
                "error_count": error_count,
                "success_rate": (total_records - error_count) / total_records if total_records > 0 else 0,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "accuracy": 0.0,
                "total_records": 0,
                "error_count": 1,
                "status": "error",
                "error_details": str(e)
            }
    
    def run(self, **initial_state) -> Dict[str, Any]:
        """Run the complete agent workflow"""
        
        # Ensure required fields are present
        required_fields = ['target_field', 'client_name', 'tdd_document']
        for field in required_fields:
            if field not in initial_state:
                return {
                    "success": False,
                    "error": f"Missing required field: {field}",
                    "required_fields": required_fields
                }
        
        # Initialize state with defaults
        state = AgentState(
            # Required inputs
            target_field=initial_state['target_field'],
            client_name=initial_state['client_name'],
            tdd_document=initial_state['tdd_document'],
            source_schema=initial_state.get('source_schema', {}),
            cbms_data=initial_state.get('cbms_data', {}),
            
            # Processing state
            current_step="initialized",
            iteration_count=0,
            max_iterations=self.config.max_iterations,
            
            # Initialize empty outputs
            tdd_analysis=None,
            business_rules=None,
            generated_code=None,
            validation_results=None,
            improvement_suggestions=None,
            messages=[],
            error_log=[],
            final_rules=None,
            final_code=None,
            confidence_score=None,
            
            # Additional context
            client_to_id=initial_state.get('client_to_id', {}),
            client_to_conformity_csv=initial_state.get('client_to_conformity_csv', {}),
            txt_files=initial_state.get('txt_files', []),
            parquet_dir=initial_state.get('parquet_dir', './data'),
            error_log_dir=initial_state.get('error_log_dir', './errors'),
            output_dir=initial_state.get('output_dir', './outputs'),
            table_cache=initial_state.get('table_cache', {})
        )
        
        try:
            # Run the workflow
            final_state = self.app.invoke(state)
            
            return {
                "success": True,
                "target_field": final_state.get("target_field"),
                "final_rules": final_state.get("final_rules"),
                "final_code": final_state.get("final_code"),
                "confidence_score": final_state.get("confidence_score"),
                "validation_results": final_state.get("validation_results"),
                "iteration_count": final_state.get("iteration_count"),
                "processing_steps": final_state.get("messages"),
                "errors": final_state.get("error_log"),
                "tdd_analysis": final_state.get("tdd_analysis")
            }
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "target_field": initial_state.get('target_field', 'unknown')
            }

# Usage Helper Class
class AgentRunner:
    """Helper to run the agent with your existing setup"""
    
    def __init__(self, config: AgentConfig = None):
        self.agent = SimpleRuleAgent(config)
    
    def run_for_field(self, target_field: str, tdd_document: str, **kwargs) -> Dict[str, Any]:
        """Run agent for a specific target field"""
        
        # Default client setup
        default_client_to_id = {
            "Oscar HealthFirst": {
                "client_id": "33019", 
                "file_id": [162014]
            }
        }
        
        # Prepare agent inputs
        agent_inputs = {
            "target_field": target_field,
            "client_name": kwargs.get('client_name', 'Oscar HealthFirst'),
            "tdd_document": tdd_document,
            "source_schema": kwargs.get('source_schema', {}),
            "cbms_data": kwargs.get('cbms_data', {}),
            "client_to_id": kwargs.get('client_to_id', default_client_to_id),
            "client_to_conformity_csv": kwargs.get('client_to_conformity_csv', {}),
            "txt_files": kwargs.get('txt_files', []),
            "parquet_dir": kwargs.get('parquet_dir', './data'),
            "error_log_dir": kwargs.get('error_log_dir', './errors'),
            "output_dir": kwargs.get('output_dir', './outputs'),
            "table_cache": kwargs.get('table_cache', {})
        }
        
        return self.agent.run(**agent_inputs)

# Example Usage
if __name__ == "__main__":
    # Configure agent
    config = AgentConfig(
        max_iterations=2,
        confidence_threshold=0.80,
        enable_auto_improvement=True,
        validation_sample_size=5  # Small sample for testing
    )
    
    # Create runner
    runner = AgentRunner(config)
    
    # Example TDD document (replace with your actual TDD)
    sample_tdd = """
    Field: grgr_id
    Description: Group ID field transformation
    Business Rule: Map source group identifier to standardized format
    Validation: Must be non-null and follow pattern [A-Z]{3}[0-9]{4}
    """
    
    # Run the agent
    result = runner.run_for_field(
        target_field="grgr_id",
        tdd_document=sample_tdd,
        client_name="Oscar HealthFirst"
    )
    
    # Print results
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Confidence: {result['confidence_score']:.1%}")
        print(f"Iterations: {result['iteration_count']}")
        print("\nProcessing Steps:")
        for step in result['processing_steps']:
            print(f"  {step}")
        if result['errors']:
            print(f"\nErrors: {result['errors']}")
    else:
        print(f"Error: {result['error']}")
