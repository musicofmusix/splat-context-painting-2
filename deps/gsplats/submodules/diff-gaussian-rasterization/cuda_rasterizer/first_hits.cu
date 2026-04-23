/*
 * Based on: diff-gaussian-rasterization/cuda_rasterizer/forward.cu
 * Original license:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "first_hits.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicAbsMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(fabs(val) > abs(__int_as_float(assumed)) ? val : __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        const float2* __restrict__ points_xy_image,
        const float4* __restrict__ conic_opacity,
        const float*  __restrict__ mask,
        float* __restrict__ out_mask,
        const float min_alpha,
        const float min_T,
        const int max_intersections,
	const bool only_inside)  {
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !(inside && mask[pix_id]);

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    // Initialize helper variables
    float T = 1.0f;
    uint32_t contributor = 0;

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();
        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // Keep track of current position in range

            // Resample using conic matrix (cf. "Surface 
            // Splatting" by Zwicker et al., 2001)
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)
                continue;
            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha > min_alpha) {
		if (!only_inside || mask[int(xy.y + 0.5) * W + int(xy.x + 0.5)]) {
                    out_mask[collected_id[j]] = 1.0;
                    contributor++;
		}
            }
            if (alpha < 1.0f / 255.0f)
                continue;
            float test_T = T * (1 - alpha);
            if (test_T < min_T || contributor >= max_intersections)
            {
                done = true;
                break;
            }
            T = test_T;

        }
    }
}

void FIRST_HITS::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float4* conic_opacity,
    const float*  mask,
    float* output_mask,
    const float min_alpha,
    const float min_T,
    const int max_intersections,
    const bool only_inside)
{
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(
        ranges,
        point_list,
        W, H,
        means2D,
        conic_opacity,
        mask,
        output_mask,
        min_alpha,
        min_T,
        max_intersections,
	only_inside
    );
}

