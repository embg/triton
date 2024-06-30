import torch
import triton

from triton import language as tl

import numpy as np
import sys


# Copied from PR #4179
@triton.jit
def flush_TMA_cache(desc_ptr):
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_ptr], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def tma_add_kernel(A, B, C, 
            a_desc_ptr, b_desc_ptr, c_desc_ptr,
            M, N,
            flush_tma_cache: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    
    if flush_tma_cache:
        tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                                [a_desc_ptr], dtype=tl.int32, is_pure=False, pack=1)
        tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                                [b_desc_ptr], dtype=tl.int32, is_pure=False, pack=1)
        tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                                [c_desc_ptr], dtype=tl.int32, is_pure=False, pack=1)
    
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a = tl._experimental_descriptor_load(a_desc_ptr, [offs_m, offs_n], [BLOCK_M, BLOCK_N], A.dtype.element_ty)
    b = tl._experimental_descriptor_load(b_desc_ptr, [offs_m, offs_n], [BLOCK_M, BLOCK_N], B.dtype.element_ty)
    c = a + b
    tl._experimental_descriptor_store(c_desc_ptr, c, [offs_m, offs_n])


def fill_tma_descriptor(gpu_desc, M, N, BLOCK_M, BLOCK_N, tensor):
    cpu_desc = torch.empty_like(gpu_desc, device="cpu")
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        tensor.data_ptr(),
        M, N, BLOCK_M, BLOCK_N,
        tensor.element_size(), cpu_desc.data_ptr()
    )
    gpu_desc.copy_(cpu_desc)

    # Note: we always flush TMA cache using a separate kernel!
    # But this is not enough to run all cases correctly.
    torch.cuda.synchronize()
    flush_TMA_cache[(1,)](gpu_desc,  num_warps=1)
    torch.cuda.synchronize()


def parse_argv():
    usage = """This script will always flush TMA cache using a separate kernel when creating TMA descriptors.
That's the approach from PR #4179. The point of the script is to demonstrate that flushing
in a separate kernel does *not* guarantee correctness. Thus, PR #4179 is insufficient.

Run "python tma_repro.py flush-in-main" to demonstrate that the code works correctly
when we add an additional TMA cache flush inside the main kernel.

Run "python tma_repro.py flush-separate-only" to demonstrate that the code crashes
when we remove the additional flush from the main kernel (so the only flush is in
the separate kernel). This reliably crashes or hangs on my machine."""
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)
    arg = sys.argv[1]
    if arg in ["flush-in-main", "flush-separate-only"]:
        return arg
    print(usage)
    sys.exit(1)


if __name__ == "__main__":
    command = parse_argv()

    # Use the same device memory for descriptors across kernel launches.
    TMA_SIZE = 128
    desc_a = torch.empty(TMA_SIZE, device="cuda", dtype=torch.int8)
    desc_b = torch.empty(TMA_SIZE, device="cuda", dtype=torch.int8)
    desc_c = torch.empty(TMA_SIZE, device="cuda", dtype=torch.int8)

    cases = [
        (256, 256),
        (512, 512),
    ]
    
    tile_sizes = [
        [128, 128],
        [128, 64],
        [64, 64]
    ]

    for M, N in cases:
        a = torch.randn((M, N), device="cuda", dtype=torch.float32)
        b = torch.randn((M, N), device="cuda", dtype=torch.float32)
        c = torch.randn((M, N), device="cuda", dtype=torch.float32)
        
        for BLOCK_M, BLOCK_N in tile_sizes:
            print(f"Case: M={M}, N={N}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
            torch.randn((M, N), device="cuda", dtype=torch.float32, out=c)

            # Note: fill_tma_descriptor() always flushes TMA cache using a separate kernel,
            # regardless of the script command.
            fill_tma_descriptor(desc_a, M, N, BLOCK_M, BLOCK_N, a)
            fill_tma_descriptor(desc_b, M, N, BLOCK_M, BLOCK_N, b)
            fill_tma_descriptor(desc_c, M, N, BLOCK_M, BLOCK_N, c)

            # This controls whether we flush TMA cache inside the tma_add_kernel
            flush_tma_cache_inside_main_kernel = (command == "flush-in-main")

            tma_add_grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)
            tma_add_kernel[tma_add_grid](a, b, c, desc_a, desc_b, desc_c, M, N,
                                        flush_tma_cache = flush_tma_cache_inside_main_kernel,
                                        BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N)
            torch.cuda.synchronize()
            assert torch.allclose(a + b, c)

    print("Successfully ran all cases!")
