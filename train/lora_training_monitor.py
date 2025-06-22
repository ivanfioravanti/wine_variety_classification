#!/usr/bin/env python3
"""
A monitoring wrapper for MLX LoRA training that intercepts validation steps
and prints decoded model outputs to track learning progress.
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
        except Exception as e:
            print(f"Error loading model/adapter: {e}")
            return False
            
    def generate_outputs(self, iteration: int, val_loss: float, num_samples: int = 5):
        """Generate and print sample outputs."""
        print(f"\n{'='*80}")
        print(f"📊 VALIDATION OUTPUTS - Iteration {iteration} | Loss: {val_loss:.4f}")
        print(f"Using adapter: {self.current_adapter_path if self.current_adapter_path else 'Base model (no adapter)'}")
        print(f"{'='*80}")
        
        if not self.validation_samples:
            print("No validation samples loaded.")
            return
        
        # Randomly select samples
        selected_samples = random.sample(self.validation_samples, min(num_samples, len(self.validation_samples)))
        
        for i, sample in enumerate(selected_samples):
            prompt = sample.get("prompt", "").strip()
            expected = sample.get("completion", "").strip()
            
            if not prompt:
                continue
                
            print(f"\n[Sample {i+1}]")
            # print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            try:
                # Apply chat template if available
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt
                
                # Create sampler with low temperature for deterministic output
                sampler = make_sampler(temp=0.1, top_p=0.9)
                
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


def monitor_training_output(config_path: str):
    """Monitor MLX training output and inject validation outputs."""
    
    monitor = ValidationMonitor(config_path)
    
    # Build MLX command with environment to suppress progress bars
    cmd = ["mlx_lm.lora", "--config", config_path, "--train"]
    
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
                    if monitor.load_model_with_adapter(None):
                        monitor.generate_outputs(iteration, val_loss)

            # Check for saved adapter
            save_match = save_pattern.search(line)
            if save_match and pending_validation:
                last_saved_adapter = save_match.group(1)
                iteration, val_loss = pending_validation
                pending_validation = None

                # Try to find the most recent adapter file
                adapter_path = last_saved_adapter
                # Load model and generate outputs
                if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
                    print(f"Using adapter directory: {adapter_dir}")
                    if monitor.load_model_with_adapter(str(adapter_dir)):
                        monitor.generate_outputs(iteration, val_loss)
                else:
                    print(f"No adapter directory found at iteration {iteration}, using base model")
                    if monitor.load_model_with_adapter(None):
                        monitor.generate_outputs(iteration, val_loss)

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