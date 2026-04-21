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
from typing import List, Optional, Sequence

import mlx.core as mx
import yaml
from mlx_lm import batch_generate, load

from data_utils import SYSTEM_PROMPT

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
        self.monitor_batch_size = int(self.config.get("monitor_batch_size", 50))
        self.monitor_max_tokens = int(self.config.get("monitor_max_tokens", 64))
        self.monitor_sample_size = int(self.config.get("monitor_sample_size", 100))
        self.monitor_system_prompt = self.config.get(
            "monitor_system_prompt", SYSTEM_PROMPT
        )
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

    def _encode_prompts(self, prompts: Sequence[str]) -> List[List[int]]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before encoding prompts.")

        encoded: List[List[int]] = []
        has_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

        for prompt in prompts:
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Expected prompt to be a string, received {type(prompt)!r} instead."
                )

            if has_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                messages = []
                if self.monitor_system_prompt:
                    messages.append({"role": "system", "content": self.monitor_system_prompt})
                messages.append({"role": "user", "content": prompt})

                try:
                    rendered = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                except TypeError:
                    rendered = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                input_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            else:
                combined = prompt
                if self.monitor_system_prompt:
                    combined = f"{self.monitor_system_prompt}\n\n{prompt}"
                input_ids = self.tokenizer.encode(combined, add_special_tokens=True)

            if not isinstance(input_ids, list):
                raise ValueError("Tokenizer.encode must return a list of token ids.")

            encoded.append(input_ids)

        return encoded

    @staticmethod
    def _extract_variety(raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        candidate = raw_text.strip()

        try:
            value = json.loads(candidate).get("variety")
            if isinstance(value, str):
                return value.strip()
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = candidate[start : end + 1]
            try:
                value = json.loads(snippet).get("variety")
                if isinstance(value, str):
                    return value.strip()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                pass

        return None

    @staticmethod
    def _parse_expected_variety(expected: str) -> Optional[str]:
        if not expected:
            return None

        expected = expected.strip()
        if not expected:
            return None

        try:
            value = json.loads(expected).get("variety")
            if isinstance(value, str):
                return value.strip()
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass

        start = expected.find("{")
        end = expected.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = expected[start : end + 1]
            try:
                value = json.loads(snippet).get("variety")
                if isinstance(value, str):
                    return value.strip()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                pass

        return expected.strip().strip('"').strip('\'')
                    
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
            
    def generate_outputs(self, iteration: int, val_loss: float, num_samples: Optional[int] = None):
        """Generate and print sample outputs."""
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"📊 VALIDATION OUTPUTS - Iteration {iteration} | Loss: {val_loss:.4f}")
        print(f"Using adapter: {self.current_adapter_path if self.current_adapter_path else 'Base model (no adapter)'}")
        print(f"{'='*80}")
        
        if not self.validation_samples:
            print("No validation samples loaded.")
            return
        
        sample_count = min(
            num_samples if num_samples is not None else self.monitor_sample_size,
            len(self.validation_samples),
        )

        if sample_count == 0:
            print("No validation samples available for batch evaluation.")
            return

        selected_samples = random.sample(self.validation_samples, sample_count)

        prompts = []
        for sample in selected_samples:
            prompt = sample.get("prompt", "").strip()
            if not prompt and "messages" in sample:
                for msg in sample["messages"]:
                    if msg.get("role") == "user":
                        prompt = msg["content"].strip()
                        break
            prompts.append(prompt)
        valid_indices = [i for i, prompt in enumerate(prompts) if prompt]

        if not valid_indices:
            print("No valid prompts found in selected validation samples.")
            return

        prompts_to_run = [prompts[i] for i in valid_indices]
        samples_to_run = [selected_samples[i] for i in valid_indices]

        try:
            encoded_prompts = self._encode_prompts(prompts_to_run)

            batch_size = max(1, min(self.monitor_batch_size, len(encoded_prompts)))

            responses = batch_generate(
                self.model,
                self.tokenizer,
                encoded_prompts,
                max_tokens=self.monitor_max_tokens,
                verbose=False,
                completion_batch_size=min(batch_size, len(encoded_prompts)),
                prefill_batch_size=min(batch_size, len(encoded_prompts)),
            )
        except Exception as exc:
            print(f"❌ Batch generation failed: {exc}")
            return

        correct = 0
        total = len(samples_to_run)
        detailed_results = []

        for idx, (sample, response_text) in enumerate(zip(samples_to_run, responses.texts)):
            prompt = prompts_to_run[idx]
            expected_raw = sample.get("completion", "").strip()
            if not expected_raw and "messages" in sample:
                for msg in sample["messages"]:
                    if msg.get("role") == "assistant":
                        expected_raw = msg["content"].strip()
                        break
            expected_variety = self._parse_expected_variety(expected_raw) or ""

            candidate_text = response_text.strip() if response_text else ""
            json_match = re.search(r'\{[^{}]*"variety"[^{}]*:[^{}]*"[^"]*"[^{}]*\}', candidate_text)

            predicted_display = candidate_text
            predicted_variety = ""

            if json_match:
                predicted_display = json_match.group()
                try:
                    parsed = json.loads(predicted_display)
                    predicted_variety = str(parsed.get("variety", "")).strip()
                except (json.JSONDecodeError, TypeError, AttributeError):
                    predicted_variety = predicted_display
            else:
                predicted_variety = candidate_text

            expected_norm = expected_variety.lower()
            predicted_norm = predicted_variety.lower()

            is_correct = False
            if expected_norm and predicted_norm:
                if expected_norm == predicted_norm:
                    is_correct = True
                elif expected_norm in predicted_norm or predicted_norm in expected_norm:
                    is_correct = True

            if is_correct:
                correct += 1

            detailed_results.append(
                {
                    "prompt": prompt,
                    "expected": expected_variety or expected_raw,
                    "expected_raw": expected_raw,
                    "predicted_display": predicted_display,
                    "predicted_variety": predicted_variety,
                    "raw_response": candidate_text,
                    "is_correct": is_correct,
                }
            )

        accuracy = (correct / total) if total else 0.0
        total_time = time.time() - start_time

        print(
            f"✅ Accuracy: {accuracy * 100:.2f}% ({correct}/{total}) | "
            f"Batch size: {self.monitor_batch_size} | Max tokens: {self.monitor_max_tokens} | "
            f"Total time: {total_time:.2f}s"
        )
        
        print("\nSample predictions:")
        for res in detailed_results[:5]:
            expected_disp = res["expected"] or "<unknown>"
            predicted_disp = res["predicted_display"]
            status = "✅" if res["is_correct"] else "❌"
            print(
                f"{status} Expected: {expected_disp} | Predicted: {predicted_disp[:150]}"
                f"{'...' if len(predicted_disp) > 150 else ''}"
            )

        print(f"{'='*80}\n")

        return accuracy


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
    pending_validation = None
    ran_initial_validation = False
    best_accuracy = 0.0
    best_iteration = 0
    best_val_loss = float("inf")
    last_accuracy = 0.0
    last_val_loss = float("inf")
    
    try:
        suppress_next_warning_line = False
        for line in process.stdout:
            if 'UnsupportedFieldAttributeWarning' in line:
                suppress_next_warning_line = True
                continue
            if suppress_next_warning_line:
                if 'warnings.warn' in line:
                    suppress_next_warning_line = False
                    continue
                suppress_next_warning_line = False

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
                        acc = monitor.generate_outputs(iteration, val_loss)
                        if acc is not None:
                            last_accuracy = acc
                            last_val_loss = val_loss
                            if acc > best_accuracy:
                                best_accuracy = acc
                                best_iteration = iteration
                                best_val_loss = val_loss

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
                        acc = monitor.generate_outputs(iteration, val_loss)
                        if acc is not None:
                            last_accuracy = acc
                            last_val_loss = val_loss
                            if acc > best_accuracy:
                                best_accuracy = acc
                                best_iteration = iteration
                                best_val_loss = val_loss
                else:
                    print(f"No adapter directory found at iteration {iteration}, using base model")
                    if monitor.load_model_with_adapter(None):
                        acc = monitor.generate_outputs(iteration, val_loss)
                        if acc is not None:
                            last_accuracy = acc
                            last_val_loss = val_loss
                            if acc > best_accuracy:
                                best_accuracy = acc
                                best_iteration = iteration
                                best_val_loss = val_loss

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        process.terminate()
        
    process.wait()

    if process.returncode == 0:
        print("\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed with return code: {process.returncode}")
        return process.returncode

    hpsearch_line = (
        f"HPSEARCH_RESULT|accuracy={last_accuracy:.4f}"
        f"|val_loss={last_val_loss:.4f}"
        f"|best_accuracy={best_accuracy:.4f}"
        f"|best_iteration={best_iteration}"
        f"|config={config_path}"
    )
    print(hpsearch_line)

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