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
    x :torch.FloatTensor,
    wq:torch.FloatTensor,
    wk:torch.FloatTensor,
    wv:torch.FloatTensor,
    k :torch.FloatTensor,
    v :torch.FloatTensor,
    batch_size :int,
    seq_len :int,
    num_head :int,
    head_dim :int,
):  
    
    q = torch.matmul(x, wq)
    dk = torch.matmul(x, wk)
    dv = torch.matmul(x, wv)

    q = q.view(batch_size, 1, num_head, head_dim).transpose(1,2)
    dk = dk.view(batch_size, 1, num_head, head_dim).transpose(1,2)
    dv = dv.view(batch_size, 1, num_head, head_dim).transpose(1,2)

    
    k[...,-1:, :].copy_(dk)
    v[...,-1:, :].copy_(dv)
    
    weight = torch.matmul(q, k.transpose(2,3)) / math.sqrt(head_dim)
    
    
    weight = F.softmax(weight, dim=-1, dtype=torch.float32).to(q.dtype)

    
    attn_output = torch.matmul(weight, v).transpose(1,2).view(batch_size, 1, num_head * head_dim)

    
    return attn_output

def ReAttention(
    x :torch.FloatTensor,
    y :torch.FloatTensor,
    wq:torch.FloatTensor,
    wk:torch.FloatTensor,
    wv:torch.FloatTensor,
    batch_size :int,
    seq_len :int,
    num_head :int,
    head_dim :int,
):  
    
    x[...,-1,:].copy_(y)
    y = torch.matmul(y, wq)
    y = y.view(batch_size, num_head, 1, head_dim)

    
    wk = wk.T.view(num_head, head_dim, num_head * head_dim)

    y = torch.matmul(y, wk).transpose(1,2)

    
    weight = torch.matmul(y, x.transpose(2,3)) / math.sqrt(head_dim)

    
    weight = F.softmax(weight, dim=-1, dtype=torch.float32).to(y.dtype)

    
    
    
    
    attn_output = torch.matmul(weight, x).transpose(1,2)
    
    
    wv = wv.view(num_head * head_dim, num_head, head_dim).transpose(0,1)

    attn_output = torch.matmul(attn_output, wv).transpose(1,2).view(batch_size, 1, num_head * head_dim)

    
    return attn_output

def KAttention(
    k :torch.FloatTensor,
    y :torch.FloatTensor,
    wq:torch.FloatTensor,
    wk:torch.FloatTensor,
    wv:torch.FloatTensor,
    batch_size :int,
    seq_len :int,
    num_head :int,
    head_dim :int,
):  
    q = torch.matmul(y, wq)
    dk = torch.matmul(y, wk)

    q = q.view(batch_size, 1, num_head, head_dim).transpose(1,2)
    dk = dk.view(batch_size, 1, num_head, head_dim).transpose(1,2)

    k[...,-1:, :].copy_(dk)

    
    weight = torch.matmul(q, k.transpose(2,3)) / math.sqrt(head_dim)

    
    
    weight = F.softmax(weight, dim=-1, dtype=torch.float32).to(y.dtype).transpose(1,2)

    
    
    
    
    attn_output = torch.matmul(weight, k.transpose(1,2).view(batch_size, seq_len, num_head * head_dim).unsqueeze(1)).transpose(1,2)
    
    
    
    wv = wv.view(num_head * head_dim, num_head, head_dim).transpose(0,1)
    
    attn_output = torch.matmul(attn_output, wv).transpose(1,2).view(batch_size, 1, num_head * head_dim)

    
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
    x = torch.rand((B, 1, N * R), dtype=dtype, device=device)
    k = torch.rand((B, N, S, R), dtype=dtype, device=device)
    v = torch.rand((B, N, S, R), dtype=dtype, device=device)

    wq = torch.rand((N * R, N * R), dtype=dtype, device=device)
    wk = torch.rand((N * R, N * R), dtype=dtype, device=device)
    wv = torch.rand((N * R, N * R), dtype=dtype, device=device)
    #graph = capture_cuda_graph_for_kvcacheattention(B, S, N, R, dtype=dtype, device=device)

    for _ in range(10):
        h = KVCacheAttention(x,wq, wk, wv, k,v, B, S, N, R)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
       h = KVCacheAttention(x,wq, wk, wv, k,v, B, S, N, R)
    torch.cuda.synchronize()
    t2 = time.time()
    print("BatchSize :{}, Seq Len:{}, Num Head:{}, Head Dim:{}, Time: {}".format(B, S, N, R, (t2 - t1)/ T))

elif args.alg == 'k':
    #x = torch.rand((B, 1, S, N * R), dtype=dtype, device=device) -0.5
    y = torch.rand((B, 1, N * R), dtype=dtype, device=device)-0.5
    
    

    wq = torch.rand((N * R, N * R), dtype=dtype, device=device)
    wk = torch.rand((N * R, N * R), dtype=dtype, device=device)
    wv = torch.rand((N * R, N * R), dtype=dtype, device=device)

    #k = torch.matmul(x, wk).squeeze(1).view(B, S, N, R).transpose(1,2)
    k = torch.rand((B, S, N, R), dtype=dtype, device=device).transpose(1,2)
    #graph = capture_cuda_graph_for_kvcacheattention(B, S, N, R, dtype=dtype, device=device)

    for _ in range(10):
        h = KAttention(k, y,wq, wk, wv, B, S, N, R)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
       h = KAttention(k, y,wq, wk, wv, B, S, N, R)
    torch.cuda.synchronize()
    t2 = time.time()
    print("BatchSize :{}, Seq Len:{}, Num Head:{}, Head Dim:{}, Time: {}".format(B, S, N, R, (t2 - t1)/ T))


elif args.alg == 're':
    x = torch.rand((B, 1, S, N * R), dtype=dtype, device=device)
    y = torch.rand((B, 1, N * R), dtype=dtype, device=device)


    wq = torch.rand((N * R, N * R), dtype=dtype, device=device)
    wk = torch.rand((N * R, N * R), dtype=dtype, device=device)
    wv = torch.rand((N * R, N * R), dtype=dtype, device=device)
    for _ in range(10):
        h = ReAttention(x,y,wq, wk, wv,B, S, N, R)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        h = ReAttention(x,y,wq, wk, wv,B, S, N, R)
    torch.cuda.synchronize()
    t2 = time.time()
    print("BatchSize :{}, Seq Len:{}, Num Head:{}, Head Dim:{}, Time: {}".format(B, S, N, R, (t2 - t1)/ T))

elif args.alg == 'verify':

    B = 1
    S = 256
    dtype = torch.float32
    x = torch.rand((B, 1, S, N * R), dtype=dtype, device=device) -0.5
    y = torch.rand((B, 1, N * R), dtype=dtype, device=device)-0.5


    wq = torch.rand((N * R, N * R), dtype=dtype, device=device)-0.5
    wk = torch.rand((N * R, N * R), dtype=dtype, device=device)-0.5
    wv = torch.rand((N * R, N * R), dtype=dtype, device=device)-0.5

    # torch.save(y, "y.pt")
    # torch.save(x, "x.pt")
    # torch.save(wq, "wq.pt")
    # torch.save(wk, "wk.pt")
    # torch.save(wv, "wv.pt")

    #exit(0)
    # x :torch.Tensor = torch.load("x.pt")
    # y :torch.Tensor= torch.load("y.pt")


    # wq :torch.Tensor= torch.load("wq.pt")
    # wk :torch.Tensor= torch.load("wk.pt")
    # wv :torch.Tensor= torch.load("wv.pt")

    wk_i = torch.inverse(wk)

    wv_new = torch.matmul(wk_i, wv)

    k = torch.matmul(x, wk).squeeze(1).view(B, S, N, R).transpose(1,2)
    v = torch.matmul(x, wv).squeeze(1).view(B, S, N, R).transpose(1,2)

    
    result0 = KAttention(k, y, wq, wk, wv_new, B, S, N, R)
    
    result1 = ReAttention(x,y, wq, wk, wv, B, S, N, R)
    result2 = KVCacheAttention(y, wq, wk, wv, k, v, B, S, N, R)

    print(result0)
    print(result1)
    print(result2)
    avg_delta0 = torch.abs(result0 - result2).sum() / (N * R)
    avg_delta1 = torch.abs(result1 - result2).sum() / (N * R)
    print(avg_delta0, avg_delta1)
    





