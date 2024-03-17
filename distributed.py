import torch
import torch.distributed as dist
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default="kv", help='algorithm')
parser.add_argument('--T', type=int, default=10000, help='time')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--S', type=int, default=8192, help='sequence length')
parser.add_argument('--N', type=int, default=32, help='num heads')
parser.add_argument('--R', type=int, default=128, help='head dimension')
args = parser.parse_args()

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

hidden_size = args.S
T = args.T
tensor = torch.rand(hidden_size).cuda()
gather_tensor = [torch.rand(hidden_size).cuda() for _ in range(world_size)]
for _ in range(10):
    dist.all_reduce(tensor)

torch.cuda.synchronize()
t1 = time.time()

for _ in range(T):
    dist.all_reduce(tensor)

torch.cuda.synchronize()
t2 = time.time()

if local_rank == 0:
    print("All Reduce Time: {}".format((t2 - t1)/ T))

for _ in range(10):
    dist.reduce(tensor, 0)

torch.cuda.synchronize()
t1 = time.time()

for _ in range(T):
    dist.reduce(tensor, 0)

torch.cuda.synchronize()
t2 = time.time()

if local_rank == 0:
    print("Reduce Time: {}".format((t2 - t1)/ T))

for _ in range(10):
    if local_rank == 0:
        dist.gather(tensor, gather_tensor)
    else:
        dist.gather(tensor)

torch.cuda.synchronize()
t1 = time.time()

for _ in range(T):
    if local_rank == 0:
        dist.gather(tensor, gather_tensor)
    else:
        dist.gather(tensor)

torch.cuda.synchronize()
t2 = time.time()

if local_rank == 0:
    print("Gather: {}".format((t2 - t1)/ T))




