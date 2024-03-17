import torch
import torch.nn.functional as F
import math
import argparse
import gc
import time
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default="kv", help='algorithm')
parser.add_argument('--T', type=int, default=10000, help='time')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--S', type=int, default=200000, help='sequence length')
parser.add_argument('--N', type=int, default=32, help='num heads')
parser.add_argument('--R', type=int, default=128, help='head dimension')
args = parser.parse_args()

def KVCacheAttention(
    q :torch.FloatTensor,
    k :torch.FloatTensor,
    v :torch.FloatTensor,
    batch_size :int,
    seq_len :int,
    num_head :int,
    head_dim :int,
):  
    weight = torch.matmul(q, k.transpose(2,3))
    weight = F.softmax(weight, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(weight, v)

    return attn_output

def ReAttention(
    x :torch.FloatTensor,
    y :torch.FloatTensor,
    batch_size :int,
    seq_len :int,
    num_head :int,
    head_dim :int,
):  
    weight = torch.matmul(y, x.transpose(2,3))
    weight = F.softmax(weight, dim=-1, dtype=torch.float32).to(y.dtype)
    attn_output = torch.matmul(weight, x)
   
    return attn_output


def capture_cuda_graph_for_kvcacheattention(
    batch_size :int,
    seq_len :int,
    num_head :int,
    head_dim :int,
    dtype=torch.float16,
    device="cuda:0",
    n_warmups=3, 
    mempool=None
):  
    static_q = torch.zeros((batch_size, num_head, 1, head_dim), dtype=dtype, device=device)
    static_k = torch.zeros((batch_size, num_head, seq_len, head_dim), dtype=dtype, device=device)
    static_v = torch.zeros((batch_size, num_head, seq_len, head_dim), dtype=dtype, device=device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_attn_output = KVCacheAttention(
                static_q,
                static_k,
                static_v,
                batch_size,
                seq_len,
                num_head,
                head_dim
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
         static_attn_output = KVCacheAttention(
                static_q,
                static_k,
                static_v,
                batch_size,
                seq_len,
                num_head,
                head_dim
            )
    def run(q, k, v):
        static_q.copy_(q)
        static_k.copy_(k)
        static_v.copy_(v)
        graph.replay()
        return static_attn_output.clone()
    
    return run

def capture_cuda_graph_for_reattention(
    batch_size :int,
    seq_len :int,
    num_head :int,
    head_dim :int,
    dtype=torch.float16,
    device="cuda:0",
    n_warmups=3, 
    mempool=None
):  
    static_y = torch.zeros((batch_size, num_head, 1, head_dim * num_head), dtype=dtype, device=device)
    static_x = torch.zeros((batch_size, 1, seq_len, head_dim  * num_head), dtype=dtype, device=device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_attn_output = ReAttention(
                static_x,
                static_y,
                batch_size,
                seq_len,
                num_head,
                head_dim
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
         static_attn_output = ReAttention(
                static_x,
                static_y,
                batch_size,
                seq_len,
                num_head,
                head_dim
            )
    def run(x, y):
        static_x.copy_(x)
        static_y.copy_(y)
        graph.replay()
        return static_attn_output.clone()
    
    return run

B = args.B
N = args.N 
S = args.S 
R = args.R
T = args.T
dtype = torch.float16
device = "cuda:0"
if args.alg == 'kv':
    q = torch.rand((B, N, 1, R), dtype=dtype, device=device)
    k = torch.rand((B, N, S, R), dtype=dtype, device=device)
    v = torch.rand((B, N, S, R), dtype=dtype, device=device)

    #graph = capture_cuda_graph_for_kvcacheattention(B, S, N, R, dtype=dtype, device=device)

    for _ in range(10):
        h = KVCacheAttention(q,k,v, B, S, N, R)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
       h = KVCacheAttention(q,k,v, B, S, N, R)
    torch.cuda.synchronize()
    t2 = time.time()
    print("BatchSize :{}, Seq Len:{}, Num Head:{}, Head Dim:{}, Time: {}".format(B, S, N, R, (t2 - t1)/ T))



if args.alg == 're':
    x = torch.rand((B, 1, S, N * R), dtype=dtype, device=device)
    y = torch.rand((B, 1, N, N * R), dtype=dtype, device=device)


    for _ in range(10):
        h = ReAttention(x,y,B, S, N, R)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        h = ReAttention(x,y,B, S, N, R)
    torch.cuda.synchronize()
    t2 = time.time()
    print("BatchSize :{}, Seq Len:{}, Num Head:{}, Head Dim:{}, Time: {}".format(B, S, N, R, (t2 - t1)/ T))


