#!/usr/bin/env python3
"""
A monitoring wrapper for MLX LoRA training that intercepts validation steps
and prints decoded model outputs to track learning progress.

COMPATIBILITY NOTES:
- Works well with Qwen models (e.g., mlx-community/Qwen3-0.6B-4bit)
- For Mistral models, use older versions like mistralai/Mistral-7B-Instruct-v0.1
- Newer Mistral models (e.g., Mistral-Small-3.2-24B-Instruct-2506) may not be 
  compatible due to Mistral3Config not being supported by current transformers library
- If model loading fails, training will continue but validation monitoring will be disabled
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import mlx.core as mx
import yaml
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

# Suppress tqdm progress bars
os.environ['TQDM_DISABLE'] = '0'


class ValidationMonitor:
    """Monitor validation steps and generate sample outputs."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        self.validation_samples = []
        self.current_adapter_path = None
        self._load_validation_samples()
        
    def _load_config(self):
        """Load training configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_validation_samples(self):
        """Load all validation samples for random selection."""
        val_path = Path(self.config["data"]) / "valid.jsonl"
        
        if val_path.exists():
            with open(val_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    self.validation_samples.append(sample)
                    
    def load_model_with_adapter(self, adapter_path: str = None):
        """Load or reload model with current adapter weights."""
        try:
            # Load base model if not already loaded or if no adapter specified
            if self.model is None or self.tokenizer is None or adapter_path is None:
                if adapter_path is None:
                    print(f"Loading base model without adapters: {self.config['model']}")
                    self.model, self.tokenizer = load(self.config["model"])
                else:
                    # MLX expects adapter_path to be a directory
                    if Path(adapter_path).is_dir():
                        print(f"Loading model with adapter directory: {adapter_path}")
                        self.model, self.tokenizer = load(
                            self.config["model"],
                            adapter_path=adapter_path
                        )
                    else:
                        print(f"Adapter path is not a directory, loading base model")
                        self.model, self.tokenizer = load(self.config["model"])
                        
            elif adapter_path and adapter_path != self.current_adapter_path:
                # Reload with new adapter weights
                if Path(adapter_path).is_dir():
                    print(f"Reloading with adapter directory: {adapter_path}")
                    self.model, self.tokenizer = load(
                        self.config["model"],
                        adapter_path=adapter_path
                    )
                else:
                    print(f"Adapter path is not a directory: {adapter_path}")
                    return False
                    
            self.current_adapter_path = adapter_path
            return True
        except KeyError as e:
            if "Mistral3Config" in str(e):
                print(f"❌ Error: The model '{self.config['model']}' uses Mistral3Config which is not supported by the current transformers library.")
                print("💡 Suggestions:")
                print("   1. Use an older Mistral model like: 'mistralai/Mistral-7B-v0.1' or 'mistralai/Mistral-7B-Instruct-v0.1'")
                print("   2. Update transformers library: pip install transformers --upgrade")
                print("   3. Use a different model that's compatible with MLX")
                print("   4. Training will continue but validation monitoring is disabled.")
                return False
            else:
                print(f"❌ KeyError loading model/tokenizer: {e}")
                print("💡 This might be a model compatibility issue. Training will continue but validation monitoring is disabled.")
                return False
        except Exception as e:
            print(f"❌ Error loading model/adapter: {e}")
            print("💡 Training will continue but validation monitoring is disabled.")
            return False
            
    def generate_outputs(self, iteration: int, val_loss: float, num_samples: int = 3):
        """Generate and print sample outputs."""
        print(f"\n{'='*80}")
        print(f"📊 VALIDATION OUTPUTS - Iteration {iteration} | Loss: {val_loss:.4f}")
        print(f"Using adapter: {self.current_adapter_path if self.current_adapter_path else 'Base model (no adapter)'}")
        print(f"{'='*80}")
        
        # Check if model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            print("❌ Validation monitoring is disabled due to model loading issues.")
            print("   Training will continue normally.")
            print(f"{'='*80}\n")
            return
        
        if not self.validation_samples:
            print("No validation samples loaded.")
            return
        
        # Randomly select samples
        selected_samples = random.sample(self.validation_samples, min(num_samples, len(self.validation_samples)))
        
        for i, sample in enumerate(selected_samples):
            prompt = sample.get("prompt", "").strip()
            expected = sample.get("completion", "").strip()

            # Support both completions and text-style datasets
            if (not prompt) and sample.get("text"):
                text_full = sample.get("text", "").strip()
                # Try to extract the expected JSON from end of text
                json_in_text = re.search(r"\{[^{}]*\"variety\"[^{}]*:[^{}]*\"[^\"]*\"[^{}]*\}(?!.*\{)", text_full, re.DOTALL)
                if json_in_text:
                    expected = json_in_text.group().strip()
                    prompt = text_full[: text_full.rfind(expected)].strip()
                else:
                    expected = ""
                    prompt = text_full

            if not prompt:
                continue

            print(f"\n[Sample {i+1}]")
            # print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            try:

                # Apply chat template only if tokenizer has template and prompt isn't already chat-formatted
                formatted_prompt = prompt
                has_template = bool(getattr(self.tokenizer, "chat_template", None))
                if has_template and hasattr(self.tokenizer, 'apply_chat_template'):
                    looks_chat_formatted = (
                        "<start_of_turn>" in prompt or "<|im_start|>" in prompt
                    )
                    if not looks_chat_formatted:
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                
                # Create sampler with requested generation parameters
                # Temperature: 1.0, Top-K: 64, Min-P: 0.0, Top-P: 0.95, Repetition Penalty: 1.0
                # Be robust to different mlx_lm versions by introspecting supported kwargs.
                try:
                    import inspect
                    supported = set()
                    try:
                        supported = set(inspect.signature(make_sampler).parameters.keys())
                    except Exception:
                        supported = set()
                    kwargs = {}
                    if 'temp' in supported:
                        kwargs['temp'] = 1.0
                    if 'top_p' in supported:
                        kwargs['top_p'] = 0.95
                    if 'top_k' in supported:
                        kwargs['top_k'] = 64
                    if 'min_p' in supported:
                        kwargs['min_p'] = 0.0
                    # repetition penalty key may vary by version
                    if 'repetition_penalty' in supported:
                        kwargs['repetition_penalty'] = 1.0
                    elif 'repeat_penalty' in supported:
                        kwargs['repeat_penalty'] = 1.0
                    sampler = make_sampler(**kwargs) if kwargs else make_sampler()
                except Exception:
                    # Fallback in case of unexpected issues
                    sampler = make_sampler(temp=1.0, top_p=0.95)
                
                # Generate completion using MLX generate function
                response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=50,
                    sampler=sampler,
                    verbose=False  # Disable verbose output
                )
                
                # Print raw response first (show what model actually generated)
                print(f"🔍 Raw output: {repr(response[:300])}{'...' if len(response) > 300 else ''}")
                
                # Extract just the generated part (remove the prompt)
                if response.startswith(formatted_prompt):
                    generated = response[len(formatted_prompt):].strip()
                elif response.startswith(prompt):
                    generated = response[len(prompt):].strip()
                else:
                    # Sometimes the model doesn't include the prompt in response
                    generated = response.strip()
                
                # Extract JSON from generated output (handle <think> tags)
                # Look for JSON pattern - match from { to the matching }
                json_match = re.search(r'\{[^{}]*"variety"[^{}]*:[^{}]*"[^"]*"[^{}]*\}', generated)
                
                if expected:
                    # First determine if the answer is correct
                    is_correct = False
                    try:
                        import json
                        expected_data = json.loads(expected)
                        expected_variety = expected_data.get("variety", "").lower()
                        
                        # Try to parse generated JSON too
                        if json_match:
                            generated_json = json_match.group()
                            try:
                                generated_data = json.loads(generated_json)
                                generated_variety = generated_data.get("variety", "").lower()
                            except:
                                generated_variety = generated.lower()
                        else:
                            generated_json = generated
                            generated_variety = generated.lower()
                        
                        # Check for exact match or substring match
                        if expected_variety == generated_variety:
                            is_correct = True
                        elif expected_variety in generated_variety or generated_variety in expected_variety:
                            is_correct = True
                    except:
                        # Fallback to simple string comparison
                        if expected.strip().lower() in generated.lower():
                            is_correct = True
                    
                    # Print with appropriate emoji
                    if json_match:
                        generated_json = json_match.group()
                        emoji = "✅" if is_correct else "❌"
                        print(f"{emoji} Generated: {generated_json}")
                    else:
                        emoji = "✅" if is_correct else "❌"
                        print(f"{emoji} Generated: {generated[:150]}{'...' if len(generated) > 150 else ''}")
                    
                    print(f"📋 Expected: {expected[:150]}{'...' if len(expected) > 150 else ''}")
                        
            except Exception as e:
                print(f"❌ Generation error: {e}")
                
        print(f"{'='*80}\n")


def _extract_user_text_from_prompt(prompt: str) -> str:
    """Extract the user portion from a chat-formatted prompt if present.

    Supports Gemma (<start_of_turn>user ... <end_of_turn>) and
    Qwen (<|im_start|>user ... <|im_end|>) formats. Falls back to the
    original string if no markers are found.
    """
    if "<start_of_turn>user" in prompt:
        m = re.search(r"<start_of_turn>user\s*(.*?)\s*<end_of_turn>", prompt, re.DOTALL)
        if m:
            return m.group(1).strip()
    if "<|im_start|>user" in prompt:
        m = re.search(r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>", prompt, re.DOTALL)
        if m:
            return m.group(1).strip()
    return prompt.strip()


def _extract_json_answer(text: str) -> str | None:
    """Extract a JSON object containing a "variety" field from text."""
    m = re.search(r"\{[^{}]*\"variety\"[^{}]*:[^{}]*\"[^\"]*\"[^{}]*\}", text, re.DOTALL)
    return m.group(0) if m else None


def ensure_config_and_data_compatibility(config_path: str) -> str:
    """If the model tokenizer lacks a chat_template, convert dataset to text-only.

    Returns the path to the config file to use (original or a patched copy).
    """
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ Failed to read config at {config_path}: {e}")
        return config_path

    model_id = cfg.get("model")
    data_dir = cfg.get("data")
    if not model_id or not data_dir:
        return config_path

    # Load only the tokenizer to check for chat template
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        has_template = bool(getattr(tokenizer, "chat_template", None))
    except Exception as e:
        print(f"⚠️ Could not load tokenizer for '{model_id}' to detect chat template: {e}")
        return config_path

    if has_template:
        # Instruct/chat model – existing prompt/completion format works
        return config_path

    # Base model: convert dataset to plain text samples
    src = Path(data_dir)
    if not src.exists():
        print(f"⚠️ Data directory not found: {src}")
        return config_path

    dst = src.parent / (src.name + "_text")
    dst.mkdir(parents=True, exist_ok=True)

    def convert_split(name: str):
        in_path = src / f"{name}.jsonl"
        out_path = dst / f"{name}.jsonl"
        if not in_path.exists():
            return
        try:
            with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
                for line in fin:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    prompt = rec.get("prompt", "")
                    completion = rec.get("completion", "")
                    if not prompt and rec.get("text"):
                        # Already text dataset – copy through
                        fout.write(json.dumps({"text": rec["text"]}) + "\n")
                        continue
                    user_text = _extract_user_text_from_prompt(str(prompt))
                    answer_json = _extract_json_answer(str(completion)) or str(completion).strip()
                    text = (user_text + "\n" + answer_json).strip()
                    fout.write(json.dumps({"text": text}) + "\n")
        except Exception as e:
            print(f"⚠️ Failed converting split '{name}': {e}")

    for split in ("train", "valid", "test"):
        convert_split(split)

    # Write a patched config referencing the text dataset
    patched_cfg = dict(cfg)
    patched_cfg["data"] = str(dst)
    patched_path = Path(config_path).with_suffix("")
    patched_path = patched_path.with_name(patched_path.name + ".base_text.yaml")
    try:
        with open(patched_path, 'w') as f:
            yaml.safe_dump(patched_cfg, f)
        print(f"🛠️ Base model detected. Using text dataset at: {dst}")
        return str(patched_path)
    except Exception as e:
        print(f"⚠️ Failed to write patched config, using original: {e}")
        return config_path

def monitor_training_output(config_path: str):
    """Monitor MLX training output and inject validation outputs."""
    
    # Ensure dataset and config are compatible with the model (base vs instruct)
    effective_config_path = ensure_config_and_data_compatibility(config_path)

    monitor = ValidationMonitor(effective_config_path)
    
    # Build MLX command with environment to suppress progress bars
    cmd = ["mlx_lm.lora", "--config", effective_config_path, "--train"]
    
    # Set up environment to suppress tqdm
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '0'
    
    print("Starting MLX LoRA training with validation monitoring...")
    print(f"Command: {' '.join(cmd)}")
    print("\nNote: Decoded outputs will be shown at each validation step.")
    print(f"Validation frequency: every {monitor.config.get('steps_per_eval', 200)} iterations\n")
    
    # Pattern to match validation output
    val_pattern = re.compile(r'Iter (\d+):.*Val loss ([\d.]+)')
    save_pattern = re.compile(r'Saved adapter (.*\.safetensors)')
    
    # Start training process with modified environment
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env
    )
    
    last_saved_adapter = None
    adapter_dir = Path(monitor.config.get("adapter_path", "adapters"))
    pending_validation = None  # Store (iteration, val_loss) until after save
    ran_initial_validation = False  # Track if we've run the initial validation
    
    try:
        for line in process.stdout:
            # Print original output
            print(line, end='', flush=True)
            
            # Check for validation line
            val_match = val_pattern.search(line)
            if val_match:
                iteration = int(val_match.group(1))
                val_loss = float(val_match.group(2))
                pending_validation = (iteration, val_loss)

                # Run initial validation after Iter 1, only once
                if not ran_initial_validation and iteration == 1:
                    ran_initial_validation = True
                    print(f"Initial validation (iteration {iteration}), using base model without adapters")
                    try:
                        if monitor.load_model_with_adapter(None):
                            monitor.generate_outputs(iteration, val_loss)
                    except Exception as e:
                        print(f"❌ Failed to run initial validation: {e}")
                        print("   Training will continue without validation monitoring.")

            # Check for saved adapter
            save_match = save_pattern.search(line)
            if save_match and pending_validation:
                last_saved_adapter = save_match.group(1)
                iteration, val_loss = pending_validation
                pending_validation = None

                # Try to find the most recent adapter file
                adapter_path = last_saved_adapter
                # Load model and generate outputs
                try:
                    if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
                        print(f"Using adapter directory: {adapter_dir}")
                        if monitor.load_model_with_adapter(str(adapter_dir)):
                            monitor.generate_outputs(iteration, val_loss)
                    else:
                        print(f"No adapter directory found at iteration {iteration}, using base model")
                        if monitor.load_model_with_adapter(None):
                            monitor.generate_outputs(iteration, val_loss)
                except Exception as e:
                    print(f"❌ Failed to run validation at iteration {iteration}: {e}")
                    print("   Training continues without validation monitoring.")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        process.terminate()
        
    # Wait for process to complete
    process.wait()
    
    if process.returncode == 0:
        print("\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed with return code: {process.returncode}")
        
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Monitor MLX LoRA training and display validation outputs"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1
        
    # Run monitoring
    return monitor_training_output(str(config_path))


if __name__ == "__main__":
    sys.exit(main())
