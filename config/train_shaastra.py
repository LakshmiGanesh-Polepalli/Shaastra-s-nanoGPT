compile = False  # This bypasses the Triton requirement
dtype = 'float16' # T1000 supports float16, but NOT bfloat16

# config/train_shaastra.py
out_dir = 'out-shaastra'
eval_interval = 25 
eval_iters = 10
log_interval = 1

# Data
dataset = 'shaastra'
batch_size = 8    # Reduced to fit smaller context
block_size = 128  # Reduced to better match your doc length

# THE FIX: Stop the massive token over-sampling
gradient_accumulation_steps = 1 

# Model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

# Optimizer
learning_rate = 5e-4 # Lowered slightly for stability
max_iters = 500      # With a small dataset, 500 steps is plenty
lr_decay_iters = 500
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 20
