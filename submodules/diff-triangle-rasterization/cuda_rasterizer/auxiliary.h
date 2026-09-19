/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#ifndef TRI_DISABLE_CENTER_DEPTH_CULL
#define TRI_DISABLE_CENTER_DEPTH_CULL 0
#endif

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

#define RENDER_AXUTILITY 1
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6

__device__ const float near_n = 0.2;
__device__ const float far_n = 100.0;
__device__ const float FilterSize = 0.707106; // sqrt(2) / 2
__device__ const float FilterInvSquare = 2.0f;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float sumf3(float3 a){return a.x + a.y + a.z;}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float3 transformPoint4x3Transpose(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2]  * p.z,
        matrix[4] * p.x + matrix[5] * p.y + matrix[6]  * p.z,
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z
    };
    return transformed;
}

__forceinline__ __device__ void substractionMat3x3(const float* M, const float* W, float* MW) {
    MW[0] = M[0] - W[0] ;
    MW[1] = M[1] - W[1] ;
    MW[2] = M[2] - W[2] ;

    MW[3] = M[3] - W[3] ;
    MW[4] = M[4] - W[4] ;
    MW[5] = M[5] - W[5] ;

    MW[6] = M[6] - W[6] ;
    MW[7] = M[7] - W[7] ;
    MW[8] = M[8] - W[8] ;
}

__forceinline__ __device__ void transformMat3x3(const float* M, const float* W, float* MW) {
    MW[0] = M[0] * W[0] + M[1] * W[3] + M[2] * W[6];
    MW[1] = M[0] * W[1] + M[1] * W[4] + M[2] * W[7];
    MW[2] = M[0] * W[2] + M[1] * W[5] + M[2] * W[8];

    MW[3] = M[3] * W[0] + M[4] * W[3] + M[5] * W[6];
    MW[4] = M[3] * W[1] + M[4] * W[4] + M[5] * W[7];
    MW[5] = M[3] * W[2] + M[4] * W[5] + M[5] * W[8];

    MW[6] = M[6] * W[0] + M[7] * W[3] + M[8] * W[6];
    MW[7] = M[6] * W[1] + M[7] * W[4] + M[8] * W[7];
    MW[8] = M[6] * W[2] + M[7] * W[5] + M[8] * W[8];
}


__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

// Convert a projected coordinate to a half-open tile bound.  Do the clamp
// before converting to uint: casting a negative coordinate to uint first
// wraps it to a very large value and can turn a valid edge triangle into an
// empty rectangle.
__forceinline__ __device__ uint clampTileBound(float coordinate, float block_size, uint limit, bool upper)
{
	float tile_coordinate = upper
		? ceilf(coordinate / block_size)
		: floorf(coordinate / block_size);
	if (!(tile_coordinate > 0.0f))
		return 0;
	if (tile_coordinate >= static_cast<float>(limit))
		return limit;
	return static_cast<uint>(tile_coordinate);
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

// Sample one triangle texture using projected barycentric coordinates.  The
// texture head stores an RGB tile per triangle in [triangle, channel, v, u]
// layout.  ``texture_color_sigma`` matches LGTM's convention: sigma=1 spans
// the complete tile, while larger/smaller values expand/shrink around its
// center.  Border clamping mirrors grid_sample(padding_mode="border").
__forceinline__ __device__ float3 sampleTriangleTexture(
	const float* textures,
	int triangle_id,
	const float2 pixel,
	const float2* projected_points,
	int cumsum,
	int texture_size,
	float texture_color_sigma)
{
	float2 a = projected_points[cumsum + 0];
	float2 b = projected_points[cumsum + 1];
	float2 c = projected_points[cumsum + 2];
	float2 pa = make_float2(pixel.x - a.x, pixel.y - a.y);
	float2 ba = make_float2(b.x - a.x, b.y - a.y);
	float2 ca = make_float2(c.x - a.x, c.y - a.y);
	float denom = ba.x * ca.y - ba.y * ca.x;
	float u = 0.3333333333f;
	float v = 0.3333333333f;
	if (fabsf(denom) > 1e-8f) {
		u = (pa.x * ca.y - pa.y * ca.x) / denom;
		v = (ba.x * pa.y - ba.y * pa.x) / denom;
	}
	float tx = fminf(1.0f, fmaxf(0.0f, 0.5f + (u - 0.5f) * texture_color_sigma));
	float ty = fminf(1.0f, fmaxf(0.0f, 0.5f + (v - 0.5f) * texture_color_sigma));
	if (texture_size <= 1) {
		int base = triangle_id * 3;
		return make_float3(textures[base + 0], textures[base + 1], textures[base + 2]);
	}
	float x = tx * (texture_size - 1);
	float y = ty * (texture_size - 1);
	int x0 = max(0, min(texture_size - 1, (int)floorf(x)));
	int y0 = max(0, min(texture_size - 1, (int)floorf(y)));
	int x1 = min(texture_size - 1, x0 + 1);
	int y1 = min(texture_size - 1, y0 + 1);
	float wx = x - (float)x0;
	float wy = y - (float)y0;
	float3 result = make_float3(0.0f, 0.0f, 0.0f);
	for (int ch = 0; ch < 3; ++ch) {
		int base = (triangle_id * 3 + ch) * texture_size * texture_size;
		float p00 = textures[base + y0 * texture_size + x0];
		float p10 = textures[base + y0 * texture_size + x1];
		float p01 = textures[base + y1 * texture_size + x0];
		float p11 = textures[base + y1 * texture_size + x1];
		float value = (1.0f - wy) * ((1.0f - wx) * p00 + wx * p10)
			+ wy * ((1.0f - wx) * p01 + wx * p11);
		if (ch == 0) result.x = value;
		else if (ch == 1) result.y = value;
		else result.z = value;
	}
	return result;
}

__forceinline__ __device__ void accumulateTriangleTextureGrad(
	float* grad_textures,
	int triangle_id,
	const float2 pixel,
	const float2* projected_points,
	int cumsum,
	int texture_size,
	float texture_color_sigma,
	const float* grad_color)
{
	if (grad_textures == nullptr || texture_size <= 0)
		return;
	float2 a = projected_points[cumsum + 0];
	float2 b = projected_points[cumsum + 1];
	float2 c = projected_points[cumsum + 2];
	float2 pa = make_float2(pixel.x - a.x, pixel.y - a.y);
	float2 ba = make_float2(b.x - a.x, b.y - a.y);
	float2 ca = make_float2(c.x - a.x, c.y - a.y);
	float denom = ba.x * ca.y - ba.y * ca.x;
	float u = 0.3333333333f;
	float v = 0.3333333333f;
	if (fabsf(denom) > 1e-8f) {
		u = (pa.x * ca.y - pa.y * ca.x) / denom;
		v = (ba.x * pa.y - ba.y * pa.x) / denom;
	}
	float tx = fminf(1.0f, fmaxf(0.0f, 0.5f + (u - 0.5f) * texture_color_sigma));
	float ty = fminf(1.0f, fmaxf(0.0f, 0.5f + (v - 0.5f) * texture_color_sigma));
	float x = tx * (texture_size - 1);
	float y = ty * (texture_size - 1);
	int x0 = max(0, min(texture_size - 1, (int)floorf(x)));
	int y0 = max(0, min(texture_size - 1, (int)floorf(y)));
	int x1 = min(texture_size - 1, x0 + 1);
	int y1 = min(texture_size - 1, y0 + 1);
	float wx = x - (float)x0;
	float wy = y - (float)y0;
	for (int ch = 0; ch < 3; ++ch) {
		int base = (triangle_id * 3 + ch) * texture_size * texture_size;
		float g = grad_color[ch];
		atomicAdd(grad_textures + base + y0 * texture_size + x0, g * (1.0f - wx) * (1.0f - wy));
		atomicAdd(grad_textures + base + y0 * texture_size + x1, g * wx * (1.0f - wy));
		atomicAdd(grad_textures + base + y1 * texture_size + x0, g * (1.0f - wx) * wy);
		atomicAdd(grad_textures + base + y1 * texture_size + x1, g * wx * wy);
	}
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + __expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	#if !TRI_DISABLE_CENTER_DEPTH_CULL
	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	#endif
	return true;
}


// Helper function to compute cross product of two vectors OA and OB
// A positive cross product indicates a counterclockwise turn,
// a negative cross product indicates a clockwise turn,
// and a zero cross product indicates the points are collinear.
__forceinline__ __device__ float crossProduct(const float2& O, const float2& A, const float2& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

__forceinline__ __device__ bool in_frustum_triangle(int idx,
	const float3 p_orig,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	#if !TRI_DISABLE_CENTER_DEPTH_CULL
	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	#endif
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
