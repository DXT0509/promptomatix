"""
Core module for prompt optimization functionality.
"""

import dspy
import ast
import json
import litellm
from typing import Dict, List, Type, Optional, Union, Tuple
from datetime import datetime
from dspy.evaluate import Evaluate
import nltk
import os
import logging
from pathlib import Path
from pathlib import Path
import requests

from ..utils.parsing import parse_dict_strings
from ..utils.paths import OPTIMIZER_LOGS_DIR
from ..core.config import Config
from ..core.session import OptimizationSession
from ..metrics.metrics import MetricsManager
from .prompts import (
    generate_synthetic_data_prompt, 
    generate_synthetic_data_validation_prompt,
    generate_meta_prompt,
    generate_meta_prompt_6,
    generate_meta_prompt_7,
    validate_synthetic_data,
    generate_meta_prompt_2,
    generate_classification_type_prompt,
    generate_evaluate_prompt
)

# Setup module logger
logger = logging.getLogger(__name__)
# Directory for optimizer logs (JSONL). Falls back to project-relative path.
OPTIMIZER_LOGS_DIR = Path("logs/optimizer")

class PromptOptimizer:
    """
    Handles the optimization of prompts using either DSPy or meta-prompt backend.
    
    Attributes:
        config (Config): Configuration for optimization
        lm: Language model instance
        optimized_prompt (str): Latest optimized prompt
        data_template (Dict): Template for data structure
        backend (str): Optimization backend ('dspy' or 'simple_meta_prompt')
    """
    
    def __init__(self, config: Config):
        """
        Initialize the optimizer with configuration.
        
        Args:
            config (Config): Configuration object containing optimization parameters
        """
        self.config = config
        # --- FORCE PROVIDER TO OPENROUTER IF USING LLAMA 4 MAVERICK ---
        # If model_name or config_model_name contains 'llama-4-maverick', force provider to openrouter
        if (
            hasattr(self.config, 'model_name') and self.config.model_name and 'llama-4-maverick' in self.config.model_name
        ) or (
            hasattr(self.config, 'config_model_name') and self.config.config_model_name and 'llama-4-maverick' in self.config.config_model_name
        ):
            self.config.model_provider = 'openrouter'
            self.config.config_model_provider = 'openrouter'
        self.lm = None
        self.llm_cost = 0
        # Initialize logger
        setup_optimizer_logger()
        self.logger = logger
        self.logger.info("PromptOptimizer initialized")
        
        # Set backend (default to DSPy for backward compatibility)
        self.backend = getattr(config, 'backend', 'simple_meta_prompt')
        if self.backend not in ['dspy', 'simple_meta_prompt']:
            raise ValueError(f"Unsupported backend: {self.backend}. Must be 'dspy' or 'simple_meta_prompt'")
        
        self.logger.info(f"Using backend: {self.backend}")

    def create_signature(self, name: str, input_fields: List[str], 
                        output_fields: List[str]) -> Type[dspy.Signature]:
        """
        Create a DSPy signature for the optimization task.
        
        Args:
            name (str): Name of the signature
            input_fields (List[str]): List of input field names
            output_fields (List[str]): List of output field names
            
        Returns:
            Type[dspy.Signature]: DSPy signature class
        """
        cleaned_name = name.strip('_').strip()
        
        # Parse fields if they're strings
        input_fields = self._parse_fields(input_fields)
        output_fields = self._parse_fields(output_fields)
        
        # Create signature class
        class_body = {
            '__annotations__': {},
            '__doc__': self.config.task
        }
        
        # Add input and output fields
        for field in input_fields:
            field = field.strip('"\'')
            class_body['__annotations__'][field] = str
            class_body[field] = dspy.InputField()
            
        for field in output_fields:
            field = field.strip('"\'')
            class_body['__annotations__'][field] = str
            class_body[field] = dspy.OutputField()
        
        return type(cleaned_name, (dspy.Signature,), class_body)
    
    def _parse_fields(self, fields: Union[List[str], str]) -> List[str]:
        """Parse field definitions from string or list."""
        if isinstance(fields, str):
            fields = fields.strip()
            if not (fields.startswith('[') and fields.endswith(']')):
                fields = f"[{fields}]"
            return ast.literal_eval(fields)
        return fields

    def generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic training data based on sample data in batches."""
        try:
            # If train_data already provided (e.g., loaded from local files), skip synthetic generation
            if getattr(self.config, 'train_data', None):
                print("[INFO] train_data already provided (skipping synthetic generation)")
                return self.config.train_data

            # If user requested loading local data but train_data is empty, try to load it now
            if getattr(self.config, 'load_data_local', False) and getattr(self.config, 'local_train_data_path', None):
                try:
                    import pandas as pd
                    from pathlib import Path as _Path
                    p = _Path(self.config.local_train_data_path)
                    if p.exists():
                        df = pd.read_csv(p)
                        self.config.train_data = df.to_dict('records')
                        print(f"[INFO] Loaded {len(self.config.train_data)} samples from local_train_data_path; skipping synthetic generation.")
                        return self.config.train_data
                except Exception as _e:
                    print(f"[WARN] Failed to load local train data in generate_synthetic_data(): {_e}")

            sample_data, sample_data_group = self._prepare_sample_data()
            template = {key: '...' for key in sample_data.keys()}

            # On average, 4 characters make up a token
            no_of_toks_in_sample_data = len(str(sample_data))/4
            
            # Ensure we generate at least as many synthetic samples as requested for training
            desired_samples = int(self.config.synthetic_data_size or 0)
            train_req = int(getattr(self.config, 'train_data_size', 0) or 0)
            if desired_samples < train_req and train_req > 0:
                print(f"[DEBUG] synthetic_data_size ({desired_samples}) < train_data_size ({train_req}); increasing synthetic_data_size to {train_req}")
                self.config.synthetic_data_size = train_req
                desired_samples = train_req

            # Calculate batch size based on token limits (assuming 16k token limit)
            max_samples_per_batch = min(50, max(1, int(8000 / no_of_toks_in_sample_data)))

            all_synthetic_data = []
            remaining_samples = self.config.synthetic_data_size
            max_retries = 3  # Maximum number of retries per batch
            validation_feedback = []  # Store feedback for failed attempts
            consecutive_failures = 0
            
            print(f"📊 Generating {self.config.synthetic_data_size} synthetic samples...")
            
            # Initialize LLM once for all batches
            with dspy.settings.context():
                try:
                    provider = getattr(self.config, 'config_model_provider', 'openai')
                    if hasattr(provider, 'value'):
                        provider = provider.value
                    if provider.lower() == 'openrouter':
                        api_key = os.environ.get("OPENROUTER_API_KEY") or self.config.model_api_key
                        api_base = "https://openrouter.ai/api/v1"
                        tmp_lm = dspy.LM(
                            model=self.config.model_name,
                            api_key=api_key,
                            api_base=api_base,
                            custom_llm_provider="openrouter",
                            max_tokens=self.config.config_max_tokens,
                            cache=False
                        )
                    else:
                        tmp_lm = dspy.LM(
                            self.config.config_model_name,
                            api_key=self.config.config_model_api_key,
                            api_base=self.config.config_model_api_base,
                            max_tokens=self.config.config_max_tokens,
                            cache=False
                        )
                    
                    batch_num = 1
                    while remaining_samples > 0:
                        batch_size = min(max_samples_per_batch, remaining_samples)
                        
                        valid_batch_data = []
                        retry_count = 0
                        
                        while len(valid_batch_data) < batch_size and retry_count < max_retries:
                            try:
                                # Include previous validation feedback in the prompt
                                feedback_section = ""
                                if validation_feedback:
                                    feedback_section = "\n### Previous Validation Feedback:\n" + "\n".join(
                                        f"- Attempt {i+1}: {feedback}" 
                                        for i, feedback in enumerate(validation_feedback[-3:])  # Show last 3 feedbacks
                                    )
                                
                                prompt = self._create_synthetic_data_prompt(
                                    sample_data_group, 
                                    template, 
                                    batch_size - len(valid_batch_data),
                                    feedback_section
                                )
                                response = tmp_lm(prompt)[0]
                                response = self._clean_llm_response(response)
                                try:
                                    batch_data = json.loads(response)
                                    print(f"\n📝 Generated {len(batch_data)} samples from LLM:")
                                    for idx, sample in enumerate(batch_data, 1):
                                        print(f"   {idx}. {json.dumps(sample, ensure_ascii=False)}")
                                    
                                    # Validate each sample in the batch
                                    print(f"\n🔍 Validating samples...")
                                    for sample in batch_data:
                                        is_valid, feedback_msg = self._validate_synthetic_data(sample, self.config.task)
                                        if is_valid:
                                            valid_batch_data.append(sample)
                                            print(f"   ✓ Valid: {json.dumps(sample, ensure_ascii=False)[:80]}...")
                                            if len(valid_batch_data) >= batch_size:
                                                break
                                        else:
                                            validation_feedback.append(feedback_msg)
                                            print(f"   ✗ Invalid: {feedback_msg[:80]}...")
                                except json.JSONDecodeError as e:
                                    validation_feedback.append(f"Failed to parse JSON response: {str(e)}")
                                
                                retry_count += 1
                            except Exception as e:
                                self.logger.error(f"Error in batch generation: {str(e)}")
                                retry_count += 1
                                continue
                        
                        if valid_batch_data:
                            all_synthetic_data.extend(valid_batch_data)
                            remaining_samples -= len(valid_batch_data)
                            consecutive_failures = 0
                            print(f"  ✓ Batch {batch_num}: {len(valid_batch_data)} samples ({len(all_synthetic_data)}/{self.config.synthetic_data_size})")
                        else:
                            consecutive_failures += 1
                            print(f"  ⚠️  Batch {batch_num} failed after {max_retries} retries (consecutive failures={consecutive_failures})")
                            # If we hit several consecutive failures, reduce the batch size and retry
                            if consecutive_failures >= 3:
                                print(f"[WARN] Consecutive batch failures reached {consecutive_failures}. Filling remaining samples deterministically as fallback.")
                                try:
                                    import copy
                                    seeds = sample_data_group if isinstance(sample_data_group, list) else [sample_data]
                                    fallback = []
                                    for i in range(remaining_samples):
                                        base = seeds[i % len(seeds)]
                                        fb = copy.deepcopy(base)
                                        fb['_fallback'] = True
                                        fallback.append(fb)
                                    all_synthetic_data.extend(fallback)
                                    remaining_samples = 0
                                except Exception as _e:
                                    print(f"[WARN] Failed to build deterministic fallback samples: {_e}")
                                    # If fallback fails, break to avoid infinite loop
                                    break
                            else:
                                # Reduce batch size to try to get at least some valid samples
                                max_samples_per_batch = max(1, int(max_samples_per_batch / 2))
                                print(f"[INFO] Reducing batch size to {max_samples_per_batch} and retrying")
                        
                        batch_num += 1
                    
                    self.llm_cost += sum([x['cost'] for x in getattr(tmp_lm, 'history', []) if x.get('cost') is not None])
                finally:
                    if 'tmp_lm' in locals():
                        del tmp_lm
                
            print(f"✅ Generated {len(all_synthetic_data)} synthetic samples")
            
            # If none were generated, build a deterministic fallback from sample_data_group
            try:
                expected = int(self.config.synthetic_data_size or 0)
            except Exception:
                expected = 0
            if expected > 0 and len(all_synthetic_data) == 0:
                try:
                    import copy
                    # Ensure sample_data_group is a list we can cycle through
                    seeds = sample_data_group if isinstance(sample_data_group, list) else [sample_data]
                    fallback = []
                    for i in range(expected):
                        base = seeds[i % len(seeds)]
                        fb = copy.deepcopy(base)
                        fb['_fallback'] = True
                        fallback.append(fb)
                    all_synthetic_data = fallback
                    print(f"[INFO] Using deterministic fallback synthetic data: {len(all_synthetic_data)} samples")
                except Exception as _e:
                    print(f"[WARN] Failed to build deterministic fallback samples: {_e}")
            
            # Print final synthetic data
            if all_synthetic_data:
                print(f"\n📋 Final Synthetic Data ({len(all_synthetic_data)} samples):")
                for idx, sample in enumerate(all_synthetic_data, 1):
                    print(f"   {idx}. {json.dumps(sample, ensure_ascii=False)}")
            
            # Warning if very few samples were generated
            try:
                expected = int(self.config.synthetic_data_size or expected)
            except Exception:
                expected = 0
            if expected > 0 and len(all_synthetic_data) == 0:
                print("[WARN] No synthetic samples were generated. Check LLM responses and rate limits.")
                self.logger.warning("No synthetic samples were generated.")
            elif expected > 0 and len(all_synthetic_data) < max(1, int(expected * 0.5)):
                print(f"[WARN] Generated fewer than expected synthetic samples ({len(all_synthetic_data)}/{expected}).")
                self.logger.warning(f"Generated fewer than expected synthetic samples: {len(all_synthetic_data)}/{expected}")
            return all_synthetic_data

        except Exception as e:
            print(f"❌ Failed to generate synthetic data: {str(e)}")
            self.logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def _prepare_sample_data(self) -> Dict:
        """Prepare sample data for synthetic data generation."""
        # If no sample_data provided, create a default based on task
        if self.config.sample_data is None or (isinstance(self.config.sample_data, str) and not self.config.sample_data.strip()):
            task = getattr(self.config, 'task', '') or ''
            default_sample = {
                "question": f"Example question based on: {task[:100]}",
                "answer": "Example answer"
            }
            return default_sample, [default_sample]
        
        if isinstance(self.config.sample_data, str):
            try:
                # First try to parse as JSON
                try:
                    data = json.loads(self.config.sample_data)
                    if isinstance(data, list):
                        return data[0], data
                    else:
                        return data, [data]
                except json.JSONDecodeError:
                    # If JSON parsing fails, try ast.literal_eval
                    data = ast.literal_eval(self.config.sample_data)
                    if isinstance(data, list):
                        return data[0], data
                    else:
                        return data, [data]
            except (SyntaxError, ValueError, json.JSONDecodeError) as e:
                self.logger.error(f"Error parsing sample data: {str(e)}")
                # Fallback: create default sample
                task = getattr(self.config, 'task', '') or ''
                default_sample = {
                    "question": f"Example question based on: {task[:100]}",
                    "answer": "Example answer"
                }
                print(f"[WARN] Failed to parse sample_data, using default sample")
                return default_sample, [default_sample]
        elif isinstance(self.config.sample_data, list):
            return self.config.sample_data[0], self.config.sample_data
        elif isinstance(self.config.sample_data, dict):
            return self.config.sample_data, [self.config.sample_data]
        else:
            # Fallback for unexpected types
            task = getattr(self.config, 'task', '') or ''
            default_sample = {
                "question": f"Example question based on: {task[:100]}",
                "answer": "Example answer"
            }
            return default_sample, [default_sample]

    def _create_synthetic_data_prompt(self, sample_data: Dict, template: Dict, batch_size: int, feedback_section: str = "") -> str:
        """Generate a high-quality prompt for synthetic data creation with specified batch size."""
        
        return generate_synthetic_data_prompt(
            task=self.config.task,
            batch_size=batch_size,
            example_data=json.dumps(sample_data, indent=2),
            template=json.dumps([template], indent=2),
            feedback_section=feedback_section
        )

    def _clean_llm_response(self, response: str) -> str:
        """Clean and format LLM response."""
        if "```json" in response:
            response = response.split("```json")[1].strip()
        if "```" in response:
            response = response.split("```")[0].strip()
        return response.strip()

    def run(self, initial_flag: bool = True) -> Dict:
        """
        Run the optimization process using the configured backend.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        if self.backend == 'dspy':
            return self._run_dspy_backend(initial_flag)
        elif self.backend == 'simple_meta_prompt':
            return self._run_meta_prompt_backend(initial_flag)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    
    def _run_dspy_backend(self, initial_flag: bool = True) -> Dict:
        """
        Run optimization using DSPy backend.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        try:
            print("Starting DSPy optimization...")
            
            # Create signature
            print("Creating DSPy signature...")
            signature = self.create_signature(
                name=f"{self.config.task_type.upper()}Signature",
                input_fields=self.config.input_fields,
                output_fields=self.config.output_fields
            )
            print("✓ Signature created")

            # Generate synthetic data if needed
            if not self.config.train_data:
                print("Generating synthetic training data...")
                synthetic_data = self.generate_synthetic_data()
                self.config.train_data = synthetic_data[:self.config.train_data_size]
                self.config.valid_data = synthetic_data[self.config.train_data_size:]
                print(f"✓ Generated {len(synthetic_data)} synthetic samples")

            # Prepare datasets
            print("Preparing datasets...")
            trainset, validset = self._prepare_datasets()
            validset_full = (self._prepare_full_validation_dataset() 
                           if self.config.valid_data_full else validset)
            print("✓ Datasets prepared")

            # Initialize trainer
            print("Initializing DSPy trainer...")
            trainer = self._initialize_trainer()
            print("✓ Trainer initialized")

            # Compile program
            print("Creating DSPy program...")
            if self.config.dspy_module == dspy.ReAct:
                program = self.config.dspy_module(signature, tools=self.config.tools)
            else:
                program = self.config.dspy_module(signature)
            print("✓ Program created")
            
            # Get evaluation metrics
            print("Setting up evaluation metrics...")
            eval_metrics = self.get_final_eval_metrics()
            print("✓ Evaluation metrics ready")
            
            # Evaluate initial prompt
            print("Evaluating initial prompt...")
            provider = getattr(self.config, 'config_model_provider', 'openai')
            if hasattr(provider, 'value'):
                provider = provider.value
            if provider.lower() == 'openrouter':
                api_key = os.environ.get("OPENROUTER_API_KEY") or self.config.model_api_key
                api_base = "https://openrouter.ai/api/v1"
                evaluator = Evaluate(devset=validset_full, metric=eval_metrics, lm=dspy.LM(
                    model=self.config.model_name,
                    api_key=api_key,
                    api_base=api_base,
                    custom_llm_provider="openrouter",
                    max_tokens=self.config.config_max_tokens,
                    cache=False
                ))
            else:
                evaluator = Evaluate(devset=validset_full, metric=eval_metrics)
            initial_score = evaluator(program=program)
            try:
                _init_score_val = getattr(initial_score, 'score', initial_score)
                print(f"✓ Initial score: {float(_init_score_val):.4f}")
            except Exception:
                print(f"✓ Initial score: {initial_score}")
            
            # Compile optimized program
            print("Compiling optimized program (this may take a while)...")
            compiled_program = self._compile_program(trainer, program, trainset, validset)
            print("✓ Program compilation complete")
            
            # Evaluate optimized prompt
            print("Evaluating optimized prompt...")
            if provider.lower() == 'openrouter':
                optimized_score = evaluator(
                    program=compiled_program
                )
            else:
                optimized_score = evaluator(
                    program=compiled_program
                )
            try:
                _opt_score_val = getattr(optimized_score, 'score', optimized_score)
                print(f"✓ Optimized score: {float(_opt_score_val):.4f}")
            except Exception:
                print(f"✓ Optimized score: {optimized_score}")
            
            try:
                opt_instructions = compiled_program.signature.instructions
            except:
                opt_instructions = compiled_program.predict.signature.instructions
            
            # Prepare and return results
            print("Preparing final results...")
            result = self._prepare_results(
                self.config.task,
                opt_instructions,
                initial_score,
                optimized_score
            )
            
            # Calculate LLM cost from optimizer's LM if available
            if hasattr(self, 'lm') and self.lm is not None:
                self.llm_cost += sum([x['cost'] for x in getattr(self.lm, 'history', []) if x.get('cost') is not None])
            
            print("✓ DSPy optimization complete!")
            return result

        except Exception as e:
            print(f"❌ Error in DSPy optimization: {str(e)}")
            self.logger.error(f"Error in DSPy optimization run: {str(e)}")
            return {'error': str(e), 'session_id': self.config.session_id}

    def _run_meta_prompt_backend(self, initial_flag: bool = True) -> Dict:
        """
        Run optimization using meta-prompt backend with direct API calls.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        try:
            # Classification gate: decide if we should proceed with meta-prompt optimization
            try:
                classification_prompt = generate_classification_type_prompt(self.config.raw_input)
                gate_resp = self._call_llm_api_directly(classification_prompt)
                gate_val = (gate_resp or "").strip().strip('"').strip("'").lower()
                is_gate_true = (gate_val == 'true')
                print(f"[DEBUG] Classification gate response: {gate_resp} -> {is_gate_true}")
                if not is_gate_true:
                    return self._handle_non_classification_case()
            except Exception as _e:
                # If gate check fails, continue with normal flow
                self.logger.warning(f"Classification gate check failed: {_e}; proceeding normally")
            
            # Generate synthetic data if needed (same as DSPy backend)
            if not self.config.train_data:
                synthetic_data = self.generate_synthetic_data()
                self.config.train_data = synthetic_data[:self.config.train_data_size]
                self.config.valid_data = synthetic_data[self.config.train_data_size:]
            
            # Auto-detect input/output fields from synthetic data if not set
            if (not self.config.input_fields or not self.config.output_fields):
                all_data = self.config.train_data + (self.config.valid_data or [])
                if all_data:
                    self._auto_detect_fields(all_data)
                    print(f"📋 Auto-detected fields - Input: {self.config.input_fields}, Output: {self.config.output_fields}")
            
            # Evaluate initial prompt first
            print("🔧 Evaluating initial prompt...")
            x = self.config.raw_input
            y = self.config.original_raw_input
            if (initial_flag == False):
                provider = getattr(self.config, 'config_model_provider', 'openai')
                if hasattr(provider, 'value'):
                    provider = provider.value
                if provider.lower() == 'openrouter':
                    api_key = os.environ.get("OPENROUTER_API_KEY") or self.config.model_api_key
                    api_base = "https://openrouter.ai/api/v1"
                    def call_llm(prompt, model=None):
                        response = litellm.completion(
                            model=self.config.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            api_base=api_base,
                            api_key=api_key,
                            custom_llm_provider="openrouter"
                        )
                        return response.choices[0].message.content
                    self._call_llm_api_directly = call_llm
            else :
                self.config.raw_input = y
            
            initial_score = self._evaluate_prompt_meta_backend(self.config.raw_input)
            
            print(f"  Initial score: {initial_score:.4f}")
            # Generate meta-prompt using the function from prompts.py
            # Prefer examples from a local CSV specified by `local_train_data_path`.
            # If not found, fall back to any `self.config.train_data` already loaded.
            examples = None
            try:
                local_path = getattr(self.config, 'local_train_data_path', None)
                if local_path:
                    p = Path(local_path).expanduser()
                    if p.exists():
                        try:
                            import pandas as _pd
                            df = _pd.read_csv(p)
                            examples = df.to_dict('records')
                        except Exception:
                            # pandas unavailable or failed; fallback to csv module
                            import csv as _csv
                            with open(p, 'r', encoding="latin-1") as _f:
                                reader = _csv.DictReader(_f)
                                examples = [row for row in reader]

                if not examples and getattr(self.config, 'train_data', None):
                    examples = self.config.train_data
            except Exception as _e:
                self.logger.warning(f"Could not load local examples for meta-prompt: {_e}")

            # If generate_meta_prompt_7 accepts examples, pass them; otherwise call old signature
            try:
                import inspect as _inspect
                sig = _inspect.signature(generate_meta_prompt_7)
                if 'examples' in sig.parameters:
                    meta_prompt = generate_meta_prompt_7(self.config.raw_input, examples=examples)
                else:
                    meta_prompt = generate_meta_prompt_7(self.config.raw_input)
            except Exception:
                meta_prompt = generate_meta_prompt_7(self.config.raw_input)
            # print("~"*100)
            # print(meta_prompt)
            # print("~"*100)
            # Get optimized prompt from LLM using direct API calls
            # By default, send the generated meta-prompt to the model to obtain an optimized prompt.
            # Allow skipping this behavior via config flag `skip_meta_generation` for debugging or
            # when you explicitly want to keep the original prompt.
            optimized_prompt = x
            print("DEBUGMETAPROMPT7: ", meta_prompt)
            if (initial_flag == False):
                
                optimized_prompt = self._call_llm_api_directly(meta_prompt, self.config.model_name)
            
            # Evaluate optimized prompt
            print("📊 Evaluating optimized prompt...")
            optimized_score = self._evaluate_prompt_meta_backend(optimized_prompt)
            print(f"  Optimized score: {optimized_score:.4f}")
            
            # Prepare and return results
            result = self._prepare_results(
                self.config.raw_input,
                optimized_prompt,
                initial_score,
                optimized_score
            )
            
            print("✅ Prompt optimization complete!")
            return result
                    
        except Exception as e:
            print(f"❌ Prompt optimization failed: {str(e)}")
            self.logger.error(f"Error in meta-prompt optimization run: {str(e)}")
            return {'error': str(e), 'session_id': self.config.session_id}

    def _handle_non_classification_case(self) -> Dict:
        """Handle prompts classified as evaluable directly without meta-optimization.

        Strategy:
        - Build a meta-prompt from the initial raw input.
        - Call the LLM to obtain an optimized prompt.
        - Query the LLM with both the initial and optimized prompts to get their responses.
        - Return structured results containing the optimized prompt and both responses.
        """
        try:
            # Debug: show which optimizer file is being executed to detect site-packages vs local src
            # 1) Create meta-prompt from the initial input
            x= self.config.raw_input
            y= self.config.original_raw_input
            
            meta_prompt = generate_meta_prompt_7(self.config.raw_input)

            # 2) Call LLM to get the optimized prompt (model from config)
            optimized_prompt_raw = self._call_llm_api_directly(meta_prompt, self.config.model_name)
            self.config.raw_input = y
            # 3) Extract final optimized prompt text if wrapped in <optimized_prompt> tags
            optimized_prompt_text = optimized_prompt_raw.strip()
            try:
                start_tag = "<optimized_prompt>"
                end_tag = "</optimized_prompt>"
                if start_tag in optimized_prompt_text and end_tag in optimized_prompt_text:
                    start_idx = optimized_prompt_text.find(start_tag) + len(start_tag)
                    end_idx = optimized_prompt_text.find(end_tag)
                    optimized_prompt_text = optimized_prompt_text[start_idx:end_idx].strip()
            except Exception:
                # If parsing fails, fall back to the raw content
                pass

            # 4) Get responses for both the initial prompt and the optimized prompt
            print("DEBUGINITIALPROMPT: ", self.config.raw_input)
            print("DEBUGOPTIMIZED PROMPT: ", optimized_prompt_text)
            initial_response = self._call_llm_api_directly(self.config.raw_input)
            optimized_response = self._call_llm_api_directly(optimized_prompt_text)
            print("DEBUG: Initial response:", initial_response)
            print("DEBUG: Optimized response:", optimized_response)
            # 5) Evaluate both responses using evaluation prompt and average metrics
            def _safe_average_score(metrics_json: Dict) -> float:
                if not isinstance(metrics_json, dict):
                    return 0.0

                weights = {
                    "relevance": 0.30,
                    "accuracy": 0.30,
                    "consistency": 0.10,
                    "readability_coherence": 0.15,
                    "efficiency": 0.15,
                }

                total = 0.0
                for k, w in weights.items():
                    v = metrics_json.get(k)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = None
                    if v is not None:
                        total += v * w

                return total

            def _evaluate_pair(prompt_text: str, answer_text: str) -> Tuple[float, Dict]:
                try:                   
                    eval_prompt = generate_evaluate_prompt(prompt_text, answer_text)
                    # Use dedicated judge model/provider if configured
                    try:
                        eval_raw = self._call_judge_llm(eval_prompt)
                    except Exception:
                        # Fallback to the main LLM if judge call fails or not configured
                        eval_raw = self._call_llm_api_directly(eval_prompt)
                    # Attempt to isolate JSON portion
                    json_start = eval_raw.find('{')
                    json_end = eval_raw.rfind('}')
                    if json_start != -1 and json_end != -1:
                        candidate = eval_raw[json_start:json_end+1]
                    else:
                        candidate = eval_raw
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        parsed = {}
                    score = _safe_average_score(parsed)
                    return score, parsed
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for pair: {str(e)}")
                    return 0.0, {}

            print("🔍 Evaluating initial prompt response...")
            initial_score, initial_metrics_json = _evaluate_pair(self.config.raw_input, initial_response)
            print(f"  Initial evaluation score: {initial_score:.4f}")
            print("📊 Evaluating optimized prompt response...")
            optimized_score, optimized_metrics_json = _evaluate_pair(self.config.raw_input, optimized_response)
            print(f"  Optimized evaluation score: {optimized_score:.4f}")
            if optimized_score < 0.5:
                meta_prompt = generate_meta_prompt_6(self.config.raw_input)
                optimized_prompt_raw = self._call_llm_api_directly(meta_prompt, self.config.model_name)
                optimized_prompt_text = optimized_prompt_raw.strip()
                optimized_response = self._call_llm_api_directly(optimized_prompt_text)
                print("DEBUGOPTIMIZED RESPONSE RETRY: ", optimized_response)
                optimized_score, optimized_metrics_json = _evaluate_pair(self.config.raw_input, optimized_response)
                print(f"  Re-evaluated optimized score: {optimized_score:.4f}")
            print("DXT: ", initial_metrics_json, "\n", optimized_metrics_json)
            # 6) Prepare result with computed scores
            result = self._prepare_results(
                self.config.raw_input,
                optimized_prompt_text,
                initial_score,
                optimized_score
            )

            # 7) Attach extra details
            result['classification_gate'] = False
            result['initial_response'] = initial_response
            result['optimized_response'] = optimized_response
            result['initial_evaluation_json'] = initial_metrics_json
            result['optimized_evaluation_json'] = optimized_metrics_json
            print("[INFO] Non-classification handling complete")
            return result
        except Exception as e:
            self.logger.error(f"Error in non-classification handling: {str(e)}")
            return {'error': str(e), 'session_id': self.config.session_id, 'classification_gate': False}

    def _evaluate_prompt_meta_backend(self, prompt: str) -> float:
        """
        Evaluate a prompt using the meta-prompt backend by testing it against synthetic data.
        
        Args:
            prompt (str): The prompt to evaluate
            
        Returns:
            float: Average score across all synthetic data samples
        """
        try:
            # Ensure MetricsManager is configured with output fields
            if self.config.output_fields:
                from ..metrics.metrics import MetricsManager
                MetricsManager.configure(self.config.output_fields)
            
            # Get the appropriate evaluation metric for the task type
            eval_metric = self.get_final_eval_metrics()
            
            # Use all available synthetic data for evaluation
            all_data = (self.config.valid_data or [])
            
            if not all_data:
                return 0.0
            
            total_score = 0.0
            valid_evaluations = 0
            failures: list[tuple[int, str]] = []
            
            for i, sample in enumerate(all_data):
                try:
                    # Create a test prompt by combining the prompt with the sample input
                    test_input = self._create_test_input_from_sample(sample)
                    full_test_prompt = f"{prompt}\n\n{test_input}"
                    print(f"[DEBUG] Evaluating sample {i+1}/{len(all_data)} with prompt:\n{full_test_prompt}\n")
                    # Get prediction from LLM
                    prediction_text = self._call_llm_api_directly(full_test_prompt)
                    print(f"[DEBUG] Prediction for sample {i+1}: {prediction_text}\n")
                    # Create prediction object with the same structure as expected
                    prediction = self._create_prediction_object(prediction_text, sample)
                    
                    # Evaluate using the appropriate metric
                    score = eval_metric(sample, prediction, prompt)
                    total_score += score
                    print(f"[DEBUG] Score for sample {i+1}: {score}\n")
                    valid_evaluations += 1
                    
                except Exception as e:
                    # Capture per-sample evaluation failures for better diagnostics
                    err_msg = str(e)
                    self.logger.warning(f"Sample {i} evaluation failed: {err_msg}")
                    failures.append((i, err_msg))
                    continue
            
            # Emit a concise summary of evaluation coverage
            try:
                total_samples = len(all_data)
            except Exception:
                total_samples = 0
            self.logger.info(
                f"Prompt eval summary: {valid_evaluations}/{total_samples} samples succeeded; "
                f"{len(failures)} failed"
            )

            if valid_evaluations == 0:
                # When all evaluations fail, return 0.0 but include actionable diagnostics in logs
                if failures:
                    first_err = failures[0][1]
                else:
                    first_err = "unknown"
                self.logger.error(
                    "All evaluations failed for this prompt. Common causes: missing/invalid API keys, "
                    "provider outages, or output_fields mismatch. First error: %s", first_err
                )
                return 0.0
            
            average_score = total_score / valid_evaluations
            return average_score
            
        except Exception as e:
            self.logger.error(f"Error in prompt evaluation: {str(e)}")
            return 0.0

    def _create_test_input_from_sample(self, sample: Dict) -> str:
        """
        Create a test input string from a sample data dictionary.
        
        Args:
            sample (Dict): Sample data containing input fields
            
        Returns:
            str: Formatted test input string
        """
        try:
            # Parse input fields
            input_fields = self._parse_input_fields()
            
            if isinstance(input_fields, (list, tuple)):
                # Multiple input fields
                input_parts = []
                for field in input_fields:
                    if field in sample:
                        input_parts.append(f"{field}: {sample[field]}")
                    else:
                        input_parts.append(f"{field}: [MISSING]")
                return "\n".join(input_parts)
            else:
                # Single input field
                if input_fields in sample:
                    return f"{input_fields}: {sample[input_fields]}"
                else:
                    return f"{input_fields}: [MISSING]"
                
        except Exception as e:
            self.logger.error(f"Error creating test input: {str(e)}")
            # Fallback: return the sample as a simple string
            return str(sample)

    def _create_prediction_object(self, prediction_text: str, sample: Dict) -> Dict:
        """
        Create a prediction object with the expected structure for evaluation.
        
        Args:
            prediction_text (str): Raw prediction text from LLM
            sample (Dict): Original sample for reference
            
        Returns:
            Dict: Prediction object with output fields populated
        """
        try:
            # Parse output fields
            if isinstance(self.config.output_fields, str):
                output_fields = ast.literal_eval(self.config.output_fields)
            else:
                output_fields = self.config.output_fields
            
            # Create prediction object
            prediction = {}
            
            if isinstance(output_fields, (list, tuple)):
                # Multiple output fields - try to extract them from the prediction text
                if len(output_fields) == 1:
                    # Single output field, use the entire prediction text
                    prediction[output_fields[0]] = prediction_text.strip()
                else:
                    # Multiple output fields - this is more complex
                    # For now, assign the prediction text to the first field
                    # In a more sophisticated implementation, you might parse the text
                    prediction[output_fields[0]] = prediction_text.strip()
                    for field in output_fields[1:]:
                        prediction[field] = ""  # Empty for additional fields
            else:
                # Single output field
                prediction[output_fields] = prediction_text.strip()
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error creating prediction object: {str(e)}")
            # Fallback: return a simple object with the prediction text
            return {"output": prediction_text.strip()}

    def _call_llm_api_directly(self, prompt: str, model: str = "") -> str:
        """
        Call LLM API directly based on the configured provider.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The LLM response
        """
        try:
            provider = getattr(self.config, 'config_model_provider', 'openai')
            if hasattr(provider, 'value'):
                provider = provider.value
            model_name = model if model else self.config.model_name
            if provider.lower() == 'openrouter':
                api_key = os.environ.get("OPENROUTER_API_KEY") or self.config.model_api_key
                api_base = "https://openrouter.ai/api/v1"
                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=api_base,
                    api_key=api_key,
                    custom_llm_provider="openrouter"
                )
                return response.choices[0].message.content
            elif provider.lower() == 'openai':
                return self._call_openai_api(prompt, model)
            elif provider.lower() == 'anthropic':
                return self._call_anthropic_api(prompt)
            else:
                raise ValueError(f"Unsupported provider for direct API calls: {provider}")
        except Exception as e:
            self.logger.error(f"Error calling LLM API directly: {str(e)}")
            raise

    def _call_judge_llm(self, prompt: str) -> str:
        """Call a dedicated judge LLM if configured, otherwise raise to allow fallback.

        Uses optional config fields:
        - judge_model_name
        - judge_model_provider ('openrouter' | 'openai' | 'anthropic')
        - judge_model_api_key
        - judge_model_api_base
        Also supports environment variables:
        - JUDGE_MODEL_PROVIDER, JUDGE_MODEL_NAME, JUDGE_API_KEY, JUDGE_API_BASE
        """
        # Resolve provider and model from config or environment
        provider = (
            getattr(self.config, 'judge_model_provider', None)
            or os.environ.get('JUDGE_MODEL_PROVIDER')
            or getattr(self.config, 'model_provider', None)
        )
        model_name = (
            getattr(self.config, 'judge_model_name', None)
            or os.environ.get('JUDGE_MODEL_NAME')
        )
        if not provider or not model_name:
            raise ValueError('Judge LLM not configured (provider/model missing)')
        # Normalize enum-like values
        if hasattr(provider, 'value'):
            provider = provider.value
        provider = provider.lower()
        print(f"[DEBUG] Using judge LLM provider: {provider}, model: {model_name}", flush=True)
        if provider == 'openrouter':
            api_key_env_judge = os.environ.get('JUDGE_API_KEY')
            api_key_env_openrouter = os.environ.get('OPENROUTER_API_KEY')
            api_key_config = getattr(self.config, 'judge_model_api_key', None)
            api_key = api_key_env_judge or api_key_config or api_key_env_openrouter
            api_base = (
                os.environ.get('JUDGE_API_BASE')
                or getattr(self.config, 'judge_model_api_base', None)
                or 'https://openrouter.ai/api/v1'
            )
            # Debug the source of the API key without revealing it
            key_source = (
                'env:JUDGE_API_KEY' if api_key_env_judge else (
                    'config:judge_model_api_key' if api_key_config else (
                        'env:OPENROUTER_API_KEY' if api_key_env_openrouter else 'none'
                    )
                )
            )
            print(f"[DEBUG] Judge API key source: {key_source}; base: {api_base}", flush=True)
            if not api_key:
                raise ValueError("Missing API key for judge OpenRouter (expect JUDGE_API_KEY or OPENROUTER_API_KEY)")
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                api_key=api_key,
                custom_llm_provider="openrouter"
            )
            return response.choices[0].message.content
        elif provider == 'openai':
            # Prefer dedicated judge API key/base from env, else reuse config
            prev_key = self.config.config_model_api_key
            prev_base = self.config.config_model_api_base
            try:
                if os.environ.get('JUDGE_API_KEY') or os.environ.get('JUDGE_API_BASE'):
                    # Temporarily override config to reuse _call_openai_api
                    self.config.config_model_api_key = os.environ.get('JUDGE_API_KEY', prev_key)
                    self.config.config_model_api_base = os.environ.get('JUDGE_API_BASE', prev_base)
                print(f"[DEBUG] Judge OpenAI key source: {'env:JUDGE_API_KEY' if os.environ.get('JUDGE_API_KEY') else 'config'}; base: {self.config.config_model_api_base}", flush=True)
                return self._call_openai_api(prompt, model_name)
            finally:
                # Restore
                self.config.config_model_api_key = prev_key
                self.config.config_model_api_base = prev_base
        elif provider == 'anthropic':
            # Prefer dedicated judge API key/base from env, else reuse config
            prev_key = self.config.config_model_api_key
            prev_base = self.config.config_model_api_base
            try:
                if os.environ.get('JUDGE_API_KEY') or os.environ.get('JUDGE_API_BASE'):
                    self.config.config_model_api_key = os.environ.get('JUDGE_API_KEY', prev_key)
                    self.config.config_model_api_base = os.environ.get('JUDGE_API_BASE', prev_base)
                print(f"[DEBUG] Judge Anthropic key source: {'env:JUDGE_API_KEY' if os.environ.get('JUDGE_API_KEY') else 'config'}; base: {self.config.config_model_api_base}", flush=True)
                return self._call_anthropic_api(prompt)
            finally:
                self.config.config_model_api_key = prev_key
                self.config.config_model_api_base = prev_base
        else:
            raise ValueError(f"Unsupported judge provider: {provider}")

    def _call_openai_api(self, prompt: str, model: str = "") -> str:
        """
        Call OpenAI API directly.
        
        Args:
            prompt (str): The prompt to send
            
        Returns:
            str: The API response
        """
        if model == "":
            model = self.config.config_model_name
        from openai import OpenAI
        
        # Configure OpenAI client
        client = OpenAI(
            api_key=self.config.config_model_api_key,
            base_url=self.config.config_model_api_base if self.config.config_model_api_base else None
        )


        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Calculate cost (approximate)
            # OpenAI pricing varies by model, this is a rough estimate
            try:
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                input_tokens = len(prompt_str.split()) * 1.3  # Rough token estimation
            except Exception:
                input_tokens = 0
            try:
                resp_str = response_text if isinstance(response_text, str) else str(response_text)
                output_tokens = len(resp_str.split()) * 1.3
            except Exception:
                output_tokens = 0
            
            # Estimate cost (this would need to be updated with actual pricing)
            estimated_cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)  # Rough estimate
            self.llm_cost += estimated_cost

            if model != "o3" and model != "gpt-4.1":
                return self._clean_llm_response(response_text)
            else:
                return response_text
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    def _call_anthropic_api(self, prompt: str) -> str:
        """
        Call Anthropic API directly.
        
        Args:
            prompt (str): The prompt to send
            
        Returns:
            str: The API response
        """
        import anthropic
        
        # Configure Anthropic client
        client = anthropic.Anthropic(
            api_key=self.config.config_model_api_key,
            base_url=self.config.config_model_api_base if self.config.config_model_api_base else None
        )
        
        try:
            response = client.messages.create(
                model=self.config.config_model_name,
                max_tokens=self.config.config_max_tokens,
                temperature=self.config.config_temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract response content
            response_text = response.content[0].text
            
            # Calculate cost (approximate)
            # Anthropic pricing varies by model, this is a rough estimate
            try:
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                input_tokens = len(prompt_str.split()) * 1.3  # Rough token estimation
            except Exception:
                input_tokens = 0
            try:
                resp_str = response_text if isinstance(response_text, str) else str(response_text)
                output_tokens = len(resp_str.split()) * 1.3
            except Exception:
                output_tokens = 0
            
            # Estimate cost (this would need to be updated with actual pricing)
            estimated_cost = (input_tokens * 0.000015) + (output_tokens * 0.000075)  # Rough estimate
            self.llm_cost += estimated_cost
            
            return self._clean_llm_response(response_text)
            
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            raise

    def _parse_input_fields(self) -> Union[str, List[str], Tuple[str, ...]]:
        """Parse input fields from config."""
        return (ast.literal_eval(self.config.input_fields) 
                if isinstance(self.config.input_fields, str) 
                else self.config.input_fields)

    def _prepare_dataset(self, data: List[Dict]) -> List[dspy.Example]:
        """Prepare a dataset from input data."""
        input_fields = self._parse_input_fields()
        
        if isinstance(input_fields, (list, tuple)):
            return [dspy.Example(**ex).with_inputs(*input_fields) for ex in data]
        return [dspy.Example(**ex).with_inputs(input_fields) for ex in data]

    def _prepare_datasets(self):
        """Prepare training and validation datasets."""
        return (
            self._prepare_dataset(self.config.train_data),
            self._prepare_dataset(self.config.valid_data)
        )

    def _prepare_full_validation_dataset(self):
        """Prepare full validation dataset if available."""
        return self._prepare_dataset(self.config.valid_data_full)

    def _initialize_trainer(self):
        """Initialize the DSPy trainer."""
        return dspy.MIPROv2(
            metric=self.get_eval_metrics(),
            init_temperature=0.3,
            auto=self.config.miprov2_init_auto,
            num_candidates=self.config.miprov2_init_num_candidates
        )

    def _compile_program(self, trainer, program, trainset, validset):
        """Compile the program using the trainer."""
        return trainer.compile(
            program,
            trainset=trainset,
            valset=validset,
            requires_permission_to_run=False,
            max_bootstrapped_demos=self.config.miprov2_compile_max_bootstrapped_demos,
            max_labeled_demos=self.config.miprov2_compile_max_labeled_demos,
            num_trials=self.config.miprov2_compile_num_trials,
            minibatch_size=self.config.miprov2_compile_minibatch_size
        )

    def _prepare_results(self, initial_prompt: str, optimized_prompt: str, 
                        initial_score: float, optimized_score: float) -> Dict:
        """Prepare the final results dictionary."""
        # Defensive: avoid float division by zero in any downstream display
        safe_initial_score = initial_score if initial_score is not None else 0.0
        safe_optimized_score = optimized_score if optimized_score is not None else 0.0
        # If both scores are zero, add a warning
        warning = None
        if safe_initial_score == 0.0 and safe_optimized_score == 0.0:
            warning = "Warning: Both initial and optimized scores are zero. This may indicate empty or invalid data, or a metric calculation issue."
        return {
            'result': optimized_prompt,
            'initial_prompt': initial_prompt,
            'session_id': self.config.session_id,
            'backend': self.backend,
            'metrics': {
                'initial_prompt_score': safe_initial_score,
                'optimized_prompt_score': safe_optimized_score
            },
            # Safely merge synthetic data pieces (may be None in non-classification path)
            'synthetic_data': (self.config.train_data or []) + (self.config.valid_data or []),
            'llm_cost': self.llm_cost,
            'warning': warning
        }

    def get_eval_metrics(self):
        """Get evaluation metrics for the task type."""
        if isinstance(self.config.output_fields, str):
            output_fields = ast.literal_eval(self.config.output_fields)
        else:
            output_fields = self.config.output_fields
        
        MetricsManager.configure(output_fields)
        return MetricsManager.get_metrics_for_task(self.config.task_type)
    
    def get_final_eval_metrics(self):
        """Get final evaluation metrics for the task type."""
        return MetricsManager.get_final_eval_metrics(self.config.task_type)

    
    def _validate_synthetic_data(self, data: Dict, task: str) -> Tuple[bool, str]:
        """
        Validate the generated synthetic data for quality and consistency.
        
        Args:
            data (Dict): The generated data sample to validate
            task (str): The task to validate the data against
            
        Returns:
            Tuple[bool, str]: (True if valid, feedback message)
        """        
        # 1) Fast local validation (schema-based) to reduce reliance on extra LLM calls
        try:
            def _nonempty(v):
                return isinstance(v, str) and v.strip() != ""

            # If input/output fields are explicitly configured, validate against them
            input_fields = getattr(self.config, 'input_fields', []) or []
            output_fields = getattr(self.config, 'output_fields', []) or []
            if input_fields or output_fields:
                has_any_input = any(_nonempty(data.get(k)) for k in input_fields)
                has_any_output = any(_nonempty(data.get(k)) for k in output_fields)
                if has_any_input and has_any_output:
                    return True, "Locally validated against configured input/output fields"

            # Otherwise, use common schema heuristics per typical tasks
            key_pairs = [
                ("question", "answer"),
                ("input", "output"),
                ("prompt", "response"),
                ("query", "answer"),
                ("text", "label"),
                ("context", "answer"),
            ]
            for a, b in key_pairs:
                if a in data and b in data and _nonempty(data[a]) and _nonempty(data[b]):
                    return True, f"Locally validated by schema: {a}/{b}"
        except Exception:
            # If local validation logic errors, fall back to LLM validation below
            pass

        # 2) Fallback to LLM-based validation
        try:
            prompt = validate_synthetic_data(task, data, self.config.input_fields, self.config.output_fields)

            response = self._call_llm_api_directly(prompt)

            # read and clean the response which is expected to be in json format
            response = self._clean_llm_response(response)
            response = json.loads(response)

            try:
                is_valid = response['is_valid']
                feedback = response['feedback']
            except Exception as e:
                self.logger.error(f"Error parsing validation response: {str(e)}")
                return False, "Invalid validation response"

            if not is_valid:
                return False, feedback
            else:
                return True, "Sample passed all validation checks"

        except Exception as e:
            error_msg = f"Error in data validation: {str(e)}"
            self.logger.error(error_msg)
            # As a safety net, if the sample superficially looks fine, accept it to avoid empty datasets
            try:
                if isinstance(data, dict) and any(isinstance(v, str) and v.strip() for v in data.values()):
                    return True, "Accepted by fallback after validation error"
            except Exception:
                pass
            return False, error_msg
    
    def _auto_detect_fields(self, synthetic_data: List[Dict]) -> None:
        """
        Auto-detect input and output fields from synthetic data.
        
        Heuristic: Common output field names appear last or are specific (answer, label, output, etc.)
        Common input field names appear first (question, input, text, context, etc.)
        
        Args:
            synthetic_data: List of synthetic data samples
        """
        if not synthetic_data:
            return
        
        # Get all keys from first sample
        sample_keys = list(synthetic_data[0].keys())
        
        # Common output field patterns (prioritized)
        output_patterns = ['answer', 'output', 'label', 'response', 'result', 'summary', 'translation', 'target']
        input_patterns = ['question', 'input', 'text', 'query', 'context', 'prompt', 'document', 'source']
        
        detected_output = []
        detected_input = []
        
        # First pass: match exact patterns
        for key in sample_keys:
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in output_patterns):
                detected_output.append(key)
            elif any(pattern in key_lower for pattern in input_patterns):
                detected_input.append(key)
        
        # If no clear output field found, assume last field is output
        if not detected_output and sample_keys:
            detected_output = [sample_keys[-1]]
        
        # If no clear input field found, assume all other fields are input
        if not detected_input:
            detected_input = [k for k in sample_keys if k not in detected_output]
        
        # Update config
        if not self.config.input_fields:
            self.config.input_fields = detected_input
        if not self.config.output_fields:
            self.config.output_fields = detected_output
        
        self.logger.info(f"Auto-detected fields - Input: {detected_input}, Output: {detected_output}")

def setup_optimizer_logger():
    """Set up dedicated logger for optimization steps and results."""
    logger.setLevel(logging.DEBUG)

    # Ensure the optimizer logs directory exists
    OPTIMIZER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create unique log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OPTIMIZER_LOGS_DIR / f"optimizer_{timestamp}.jsonl"

    # File Handler for JSON Lines format
    class JSONLinesHandler(logging.FileHandler):
        def emit(self, record):
            try:
                # Only try to parse JSON if it's an optimization step
                if hasattr(record, 'optimization_step'):
                    msg = record.optimization_step
                else:
                    # For regular log messages, create a simple JSON structure
                    msg = {
                        'timestamp': datetime.now().isoformat(),
                        'level': record.levelname,
                        'message': self.format(record)
                    }
                
                with open(self.baseFilename, 'a') as f:
                    json.dump(msg, f)
                    f.write('\n')
            except Exception as e:
                # Log error to stderr but don't raise to avoid logging loops
                import sys
                print(f"Error in JSONLinesHandler: {str(e)}", file=sys.stderr)
                # Ensure the file is closed even if an error occurs
                self.close()

    # Custom formatter for optimization steps
    class OptimizationStepFormatter(logging.Formatter):
        def format(self, record):
            if hasattr(record, 'optimization_step'):
                return json.dumps(record.optimization_step)
            return super().format(record)

    # Set up file handler with custom formatter
    file_handler = JSONLinesHandler(log_file)
    file_handler.setFormatter(OptimizationStepFormatter())
    logger.addHandler(file_handler)

    logger.info(f"Optimizer logger initialized. Log file: {log_file}")