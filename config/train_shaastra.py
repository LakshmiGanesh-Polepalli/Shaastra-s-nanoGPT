# config/train_shaastra.py
out_dir = 'out-shaastra'
eval_interval = 50 
eval_iters = 20
log_interval = 10

# Data
dataset = 'shaastra'
batch_size = 16
block_size = 256 # The model looks at 256 tokens at a time

# Model (Scaled down for small data)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2 # Higher dropout adds 'noise' to prevent memorization

# Optimizer
learning_rate = 1e-3
max_iters = 1000 # Stop early before it overfits
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 50
