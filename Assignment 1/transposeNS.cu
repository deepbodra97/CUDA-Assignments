/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*This code has been adapted for CIS4930/CIS6930 Accelerated Computing with GPUs*/

#include <stdio.h>
#include <iostream>
#include "cudaCheck.cuh"


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

// Check errors and print GB/s
void postprocess(const float* ref, const float* res, int n, float ms) {
	bool passed = true;
	for (int i = 0; i < n; i++)
		if (res[i] != ref[i]) {
			printf("%d %f %f\n", i, res[i], ref[i]);
			printf("%25s\n", "*** FAILED ***");
			passed = false;
			break;
		}
	if (passed)
		printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
__global__ void copy(float* odata, const float* idata, int nx, int ny) {
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	if (x<nx && y<ny){
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
			if (y + j < ny && x < nx){
				odata[(y + j) * nx + x] = idata[(y + j) * nx + x];
			}
		}
	}
}

// optimized copy kernel
__global__ void copyOptimized(float* odata, const float* idata, int nx, int ny) {
	__shared__ float cache[TILE_DIM * TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = nx;

	if (x<nx && y<ny){
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
			if (y + j < ny && x < nx) {
				cache[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];
			}
		}
		__syncthreads();

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
			if (y + j < ny && x < nx) {
				odata[(y + j) * width + x] = cache[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
			}
		}
	}
}

// Simplest transpose
__global__ void transposeNaive(float* odata, const float* idata, int nx, int ny) {
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	if (x<nx && y<ny){
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
			if (y + j < ny && x < nx) {
				odata[x * ny + (y + j)] = idata[(y + j) * nx + x];
			}
		}
	}
}

// Optimized transpose
__global__ void transposeOptimized(float* odata, const float* idata, int nx, int ny) {
	__shared__ float cache[TILE_DIM * (TILE_DIM + 1)];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	if (x < nx && y < ny) {
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
			if (y + j < ny && x < nx) {
				cache[(threadIdx.y + j) * (TILE_DIM + 1) + threadIdx.x] = idata[(y + j) * nx + x];
			}
		}
	}

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
		if (y + j < nx && x < ny) {
			odata[(y + j) * ny + x] = cache[threadIdx.x * (TILE_DIM + 1) + threadIdx.y + j];
		}
	}
}

int main(int argc, char* argv[]) {

	const int nx = 1027;
	const int ny = 1019;
	const int mem_size = nx * ny * sizeof(float);

	dim3 dimGrid((nx - 1) / TILE_DIM + 1, (ny - 1) / TILE_DIM + 1, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	int devId;
	cudaCheck(cudaGetDevice(&devId));
	printf("\nDevice number: %d", devId);

	cudaDeviceProp prop;
	cudaCheck(cudaGetDeviceProperties(&prop, devId));
	printf("\nDevice : %s\n", prop.name);
	printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
		nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
	printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
		dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

	cudaCheck(cudaSetDevice(devId));

	float* h_idata = (float*)malloc(mem_size);
	float* h_cdata = (float*)malloc(mem_size);
	float* h_tdata = (float*)malloc(mem_size);
	float* gold = (float*)malloc(mem_size);

	float* d_idata, * d_cdata, * d_tdata;
	cudaCheck(cudaMalloc(&d_idata, mem_size));
	cudaCheck(cudaMalloc(&d_cdata, mem_size));
	cudaCheck(cudaMalloc(&d_tdata, mem_size));

	// check parameters and calculate execution configuration
	/*if (nx % TILE_DIM || ny % TILE_DIM) {
		printf("nx and ny must be a multiple of TILE_DIM\n");
		goto error_exit;
	}*/

	/*if (TILE_DIM % BLOCK_ROWS) {
		printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
		goto error_exit;
	}*/

	// host
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			h_idata[j * nx + i] = j * nx + i;

	// correct result for error checking
	for (int j = 0; j < nx; j++)
		for (int i = 0; i < ny; i++)
			gold[j * ny + i] = h_idata[i * nx + j];

	// device
	cudaCheck(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

	// events for timing
	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));
	float ms;

	// ------------
	// time kernels
	// ------------
	printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

	// ----
	// copy 
	// ----
	printf("%25s", "copy");
	cudaCheck(cudaMemset(d_cdata, 0, mem_size));
	// warm up
	copy << <dimGrid, dimBlock >> > (d_cdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		copy << <dimGrid, dimBlock >> > (d_cdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	cudaCheck(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(h_idata, h_cdata, nx * ny, ms);

	// ----
	// copy optimized
	// ----
	//dim3 dimBlock2(TILE_DIM, TILE_DIM, 1);
	printf("%25s", "copyOptimized");
	cudaCheck(cudaMemset(d_cdata, 0, mem_size));
	// warm up
	copyOptimized << <dimGrid, dimBlock >> > (d_cdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		copyOptimized << <dimGrid, dimBlock >> > (d_cdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	cudaCheck(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(h_idata, h_cdata, nx * ny, ms);

	// --------------
	// transposeNaive 
	// --------------
	printf("%25s", "naive transpose");
	cudaCheck(cudaMemset(d_tdata, 0, mem_size));
	// warmup
	transposeNaive << <dimGrid, dimBlock >> > (d_tdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		transposeNaive << <dimGrid, dimBlock >> > (d_tdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	cudaCheck(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(gold, h_tdata, nx * ny, ms);

	// --------------
	// transposeOptimized 
	// --------------
	printf("%25s", "optimized transpose");
	cudaCheck(cudaMemset(d_tdata, 0, mem_size));
	// warmup
	transposeOptimized << <dimGrid, dimBlock >> > (d_tdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		transposeOptimized << <dimGrid, dimBlock >> > (d_tdata, d_idata, nx, ny);
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	cudaCheck(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(gold, h_tdata, nx * ny, ms);


error_exit:
	// cleanup
	cudaCheck(cudaEventDestroy(startEvent));
	cudaCheck(cudaEventDestroy(stopEvent));
	cudaCheck(cudaFree(d_tdata));
	cudaCheck(cudaFree(d_cdata));
	cudaCheck(cudaFree(d_idata));
	free(h_idata);
	free(h_tdata);
	free(h_cdata);
	free(gold);
}
