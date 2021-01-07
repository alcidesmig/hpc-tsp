#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#define ll long long
#define MAX_THREAD_PER_BLOCK 1024
#define MAX_NB_CITIES 50
#define DEBUG

__device__ int calc_perm_cost(ll idx, int n, ll nb_perm, int * dist, ll * fact, int tasks_per_thread, ll limit_blocks_1d) {
	// Test for valid idx
	int cont_tasks = 0, cost = 0;
	int min_cost = INT_MAX;
	// Resulting perm vector
	int perm[MAX_NB_CITIES + 1];

	if(idx >= nb_perm || cont_tasks == tasks_per_thread) return INT_MAX;

	// Get and calculate permutations for the current thread
	while (idx < nb_perm && cont_tasks != tasks_per_thread) {

		// compute factorial code
		for (int k = 0; k < n; ++k) {
			perm[k] = idx / fact[n - 1 - k];
			idx = idx % fact[n - 1 - k];
		}

		// readjust values to obtain the permutation
		for (int k = n - 1; k > 0; --k) {
			for (int j = k - 1; j >= 0; --j) {
				if (perm[j] <= perm[k]) {
					perm[k]++;
				}
			}
		}

		// Loop path
		perm[n] = perm[0];

		// Perm cost
		cost = 0;

		// Calc perm cost
		for (int i = 0; i < n; i++) {
			cost += dist[perm[i] * n + perm[i + 1]];
		}

		idx = idx + limit_blocks_1d;
		cont_tasks++;
		
		// maintain min cost
		cost = cost < min_cost ? cost : min_cost;
	}

	return cost;
}

__global__ void perm_cuda(int n, ll nb_perm, int * dist, ll * fact, int * mcost, int tasks_per_thread, ll limit_blocks_1d) {
	// Thread index
	int tidx = threadIdx.x;

	// Block index
	int bidx = blockIdx.x;

	// Global index
	ll idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Shared dist matrix
	__shared__ int s_dist[MAX_NB_CITIES*MAX_NB_CITIES];
	
	// Copy global dist matrix to shared dist matrix
	int step = (n*n + blockDim.x - 1)/(blockDim.x);
	int start = tidx*step;
	for (int i = 0; i < step; i++) {
		if ((start + i) < (n*n)) {
			s_dist[start + i] = dist[start + i];
		}
	}

	// Shared factorial
	__shared__ ll s_fact[MAX_NB_CITIES];

	// Copy global factorial to shared factorial
	if (tidx < n) {
		s_fact[tidx] = fact[tidx];
	}
	__syncthreads();

	// Shared cost
	__shared__ int s_mcost[MAX_THREAD_PER_BLOCK];

	// Minimal local cost
	s_mcost[tidx] = calc_perm_cost(idx, n, nb_perm, s_dist, s_fact, tasks_per_thread, limit_blocks_1d);
	__syncthreads();

	// Reduce local cost to find global
	ll step_size = 1;
	int nb_threads = blockDim.x/2;

	while (nb_threads > 0) {
		if (tidx < nb_threads) {
			int fst = tidx * step_size * 2;
			int snd = fst  + step_size;

			if (s_mcost[snd] < s_mcost[fst]) {
				s_mcost[fst] = s_mcost[snd];
			}
		}

		step_size <<= 1;
		nb_threads >>= 1;

		__syncthreads();
	}

	// Put mcost from block in global
	if (tidx == 0) {
		mcost[bidx] = s_mcost[0];
	}
}

void printDeviceProps() {
	cudaSetDevice(0);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	printf("prop.name                        %s\n",       prop.name);
	printf("prop.totalGlobalMem              %zdGB\n",    prop.totalGlobalMem >> 10 >> 10 >> 10);
	printf("prop.sharedMemPerBlock           %zdKB\n",    prop.sharedMemPerBlock >> 10);
	printf("prop.regsPerBlock                %d\n",       prop.regsPerBlock);
	printf("prop.warpSize                    %d\n",       prop.warpSize);
	printf("prop.memPitch                    %zdGB\n",    prop.memPitch >> 10 >> 10 >> 10);
	printf("prop.maxThreadsPerBlock          %d\n",       prop.maxThreadsPerBlock);
	printf("prop.maxThreadsDim               %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("prop.maxGridSize                 %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("prop.totalConstMem               %zdKB\n",    prop.totalConstMem >> 10);
	printf("prop.multiProcessorCount         %d\n",       prop.multiProcessorCount);
	printf("prop.l2CacheSize                 %dMB\n",     prop.l2CacheSize >> 10 >> 10);
	printf("prop.maxThreadsPerMultiProcessor %d\n",       prop.maxThreadsPerMultiProcessor);
	printf("prop.sharedMemPerMultiprocessor  %zdKB\n",    prop.sharedMemPerMultiprocessor >> 10);
	printf("prop.regsPerMultiprocessor       %d\n",       prop.regsPerMultiprocessor);
	printf("\n");
}

int * putDMatrixInDevice(int n) {
	// X coords
	int * x = (int *)malloc(n*sizeof(int));
	if (!x) {
		printf("malloc error\n");
		exit(1);
	}

	// Y coords
	int * y = (int *)malloc(n*sizeof(int));
	if (!y) {
		printf("malloc error\n");
		exit(1);
	}

	// Read
	for (int i = 0; i < n; i++) {
		scanf("%d %d", &x[i], &y[i]);
	}

	// Dist matrix on host
	int * h_dist = (int *)malloc(n*n*sizeof(int));
	if (!h_dist) {
		printf("malloc error\n");
		exit(1);
	}

	// Calc dist matrix
	int dx, dy;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			dx = x[i] - x[j];
			dy = y[i] - y[j];
			h_dist[i*n + j] = (int)sqrt(dx*dx + dy*dy);
		}
	}

	free(x);
	free(y);

	#ifdef DEBUG
		// Print dist
		printf("dist\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%3d ", h_dist[i*n + j]);
			}
			printf("\n");
		}
	#endif

	// Dist matrix on device
	int * d_dist;
	if (cudaMalloc(&d_dist, n*n*sizeof(int)) == cudaErrorMemoryAllocation) {
		printf("cudaMalloc error\n");
		exit(1);
	}

	// Copy host to device
	cudaMemcpy(d_dist, h_dist, n*n*sizeof(int), cudaMemcpyHostToDevice);

	free(h_dist);

	return d_dist;
}

ll * putFactInDevice(int n) {
	// Fact vector on host
	ll * h_fact = (ll *)malloc(n*sizeof(ll));
	if (!h_fact) {
		printf("malloc error\n");
		exit(1);
	}

	// Calc fact
	h_fact[0] = 1;
	for (int i = 1; i < n; i++) {
		h_fact[i] = i * h_fact[i - 1];
	}

	#ifdef DEBUG
		// Print fact
		printf("fact ");
		for (int i = 0; i < n; i++) {
			printf("%lld ", h_fact[i]);
		}
		printf("\n");
	#endif

	// Fact vector on device
	ll * d_fact;
	if (cudaMalloc(&d_fact, n*sizeof(ll)) == cudaErrorMemoryAllocation) {
		printf("cudaMalloc error\n");
		exit(1);
	}

	// Copy host to device
	cudaMemcpy(d_fact, h_fact, n*sizeof(ll), cudaMemcpyHostToDevice);

	return d_fact;
}

ll factorial(int n) {
	return (n > 1 ? n*factorial(n-1) : 1);
}

int run_tsp() {
	int nb_cities, tasks_per_thread = 1;
	scanf("%d", &nb_cities);

	// Dist matrix
	int * d_dist = putDMatrixInDevice(nb_cities);

	// Fact vector
	ll * d_fact = putFactInDevice(nb_cities);

	// Number of permutations
	ll nb_perm = factorial(nb_cities);

	#ifdef DEBUG
		printf("nb_cities\t\t%d\n", nb_cities);
		printf("nb_perm\t\t\t%lld\n", nb_perm);
	#endif

	cudaSetDevice(0);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	// Calculate number of tasks per thread
	ll limit_blocks_1d = prop.maxGridSize[0];
	if(nb_perm / MAX_THREAD_PER_BLOCK > limit_blocks_1d) {
		tasks_per_thread = ceil((nb_perm / MAX_THREAD_PER_BLOCK) / limit_blocks_1d);
	}

	// Set number of tasks/thread and # blocks
	ll nb_blocks, nb_threads;
	if(nb_perm > limit_blocks_1d) {
		nb_blocks = limit_blocks_1d;
		nb_threads = MAX_THREAD_PER_BLOCK;
	} else {
		nb_blocks  = (nb_perm + MAX_THREAD_PER_BLOCK - 1)/MAX_THREAD_PER_BLOCK;
		nb_threads = (nb_perm > MAX_THREAD_PER_BLOCK ? MAX_THREAD_PER_BLOCK : nb_perm);
	}

	#ifdef DEBUG
		printf("nb_blocks\t\t%lld\n", nb_blocks);
		printf("nb_threads\t\t%lld\n", nb_threads);
		printf("tasks/thread\t\t%lld\n", tasks_per_thread);
	#endif

	// Mim cost from blocks
	int * h_mcost = (int *) malloc(nb_blocks*sizeof(int));
	int * d_mcost;
	cudaMalloc(&d_mcost, nb_blocks*sizeof(int));

	// Call gpu
	clock_t start = clock();
	perm_cuda<<<nb_blocks, nb_threads>>>(nb_cities, nb_perm, d_dist, d_fact, d_mcost, tasks_per_thread, limit_blocks_1d);
	cudaDeviceSynchronize();
	clock_t end = clock();

	// Time
	int msec = (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("%ds %dms\n", msec/1000, msec%1000);

	// Copy device to host, ret
	cudaMemcpy(h_mcost, d_mcost, nb_blocks*sizeof(int), cudaMemcpyDeviceToHost);

	// Error check
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n",cudaGetErrorString(err));
	}

	// Host global min
	int global_mcost = INT_MAX;

	// Find global min
	for (int i = 0; i < nb_blocks; i++) {
		if (h_mcost[i] < global_mcost) {
			global_mcost = h_mcost[i];
		}
	}

	cudaFree(d_fact);
	cudaFree(d_dist);
	cudaFree(d_mcost);

	return global_mcost;
}

int main() {
	#ifdef DEBUG
		printDeviceProps();
	#endif

	int num_instances;
	
	scanf("%d", &num_instances);

	for (int i = 0; i < num_instances; i++) {
		printf("%d\n", run_tsp());
	}

	return 0;
}