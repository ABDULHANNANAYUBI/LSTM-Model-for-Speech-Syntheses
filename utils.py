num_hidden_layers = 3
hidden_size = 512

batch_size = 32
n_workers = 2
pin_memory = True
nepoch = 25
lr = 0.001
weight_decay = 1e-6
use_cuda = torch.cuda.is_available()
print(use_cuda)