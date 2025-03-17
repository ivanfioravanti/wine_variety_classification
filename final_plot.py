import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate training metrics plot from log file')
parser.add_argument('--log_file', type=str, default='./logs/train.log', 
                    help='Path to the training log file (default: ./logs/train.log)')
args = parser.parse_args()

# Read data from the log file
try:
    with open(args.log_file, 'r') as f:
        log_data = f.read()
except FileNotFoundError:
    print(f"Error: Log file '{args.log_file}' not found.")
    exit(1)

# Initialize lists to store the data
iterations = []
train_losses = []
learning_rates = []
tokens_per_sec = []
iter_per_sec = []
val_iterations = []
val_losses = []

# Parse the log data
for line in log_data.strip().split('\n'):
    if "Train loss" in line and "Learning Rate" in line:
        # Extract data using regex
        iter_match = re.search(r'Iter (\d+):', line)
        loss_match = re.search(r'Train loss ([\d.]+)', line)
        lr_match = re.search(r'Learning Rate ([\d.e-]+)', line)
        it_sec_match = re.search(r'It/sec ([\d.]+)', line)
        tok_sec_match = re.search(r'Tokens/sec ([\d.]+)', line)
        
        if all([iter_match, loss_match, lr_match, it_sec_match, tok_sec_match]):
            iterations.append(int(iter_match.group(1)))
            train_losses.append(float(loss_match.group(1)))
            learning_rates.append(float(lr_match.group(1)))
            iter_per_sec.append(float(it_sec_match.group(1)))
            tokens_per_sec.append(float(tok_sec_match.group(1)))
    
    # Extract validation loss data
    elif "Val loss" in line and "Val took" in line:
        iter_match = re.search(r'Iter (\d+):', line)
        val_loss_match = re.search(r'Val loss ([\d.]+)', line)
        
        if iter_match and val_loss_match:
            val_iterations.append(int(iter_match.group(1)))
            val_losses.append(float(val_loss_match.group(1)))

# Calculate averages
avg_tokens_per_sec = np.mean(tokens_per_sec)
avg_iter_per_sec = np.mean(iter_per_sec)

# Create a single plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Add title with averages
fig.suptitle(f'MLX Distributed M3 Ultra + M2 Ultra - Phi-4 Training Metrics\nAvg Tokens/sec: {avg_tokens_per_sec:.2f} | Avg It/sec: {avg_iter_per_sec:.4f}', fontsize=14)

# Plot Train Loss and Validation Loss on left y-axis
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.plot(iterations, train_losses, 'b-', marker='o', markersize=4, label='Train Loss')
if val_iterations and val_losses:
    ax1.plot(val_iterations, val_losses, 'r-', marker='s', markersize=6, label='Validation Loss')

# Create second y-axis for Learning Rate
ax2 = ax1.twinx()
ax2.set_ylabel('Learning Rate')
ax2.plot(iterations, learning_rates, 'g-', marker='^', markersize=4, label='Learning Rate')

# Add legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
plt.savefig('training_metrics.png', dpi=300)

print(f"Data points extracted: {len(iterations)} train, {len(val_iterations)} validation")
print(f"Average Tokens/sec: {avg_tokens_per_sec:.2f}")
print(f"Average It/sec: {avg_iter_per_sec:.4f}")
print(f"Starting Train Loss: {train_losses[0]:.4f}")
print(f"Final Train Loss: {train_losses[-1]:.4f}")
if val_losses:
    print(f"Final Validation Loss: {val_losses[-1]:.4f}") 