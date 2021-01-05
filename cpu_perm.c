#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#define ll long long
#define MAX_NB_CITIES 50

int calc_perm_cost(ll idx, int n, ll nb_perm, int * dist, ll * fact) {
	// Test for valid idx
	if (idx < nb_perm) {
		// Resulting perm vector
		int perm[MAX_NB_CITIES + 1];

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
		int cost = 0;

		// Calc perm cost
		for (int i = 0; i < n; i++) {
			cost += dist[perm[i] * n + perm[i + 1]];
		}

		return cost;
	} else {
		return INT_MAX;
	}
}

// __global__ void perm_cuda(int n, ll nb_perm, int * dist, ll * fact, int * mcost) {
// 	// Thread index
// 	int tidx = threadIdx.x;

// 	// Block index
// 	int bidx = blockIdx.x;

// 	// Global index
// 	ll idx = blockIdx.x * blockDim.x + threadIdx.x;

// 	// Shared dist matrix
// 	__shared__ int s_dist[MAX_NB_CITIES*MAX_NB_CITIES];
	
// 	// Copy global dist matrix to shared dist matrix
// 	int step = (n*n + blockDim.x - 1)/(blockDim.x);
// 	int start = tidx*step;
// 	for (int i = 0; i < step; i++) {
// 		if ((start + i) < (n*n)) {
// 			s_dist[start + i] = dist[start + i];
// 		}
// 	}

// 	__syncthreads();

// 	// Shared factorial
// 	__shared__ ll s_fact[MAX_NB_CITIES];

// 	// Copy global factorial to shared factorial
// 	if (tidx < n) {
// 		s_fact[tidx] = fact[tidx];
// 	}
// 	__syncthreads();

// 	// Shared cost
// 	__shared__ int s_mcost[MAX_THREAD_PER_BLOCK];

// 	// Minimal local cost
// 	s_mcost[tidx] = calc_perm_cost(idx, n, nb_perm, s_dist, s_fact);
// 	__syncthreads();

// 	// Reduce local cost to find global
// 	ll step_size = 1;
// 	int nb_threads = blockDim.x/2;

// 	while (nb_threads > 0) {
// 		if (tidx < nb_threads) {
// 			int fst = tidx * step_size * 2;
// 			int snd = fst  + step_size;

// 			if (s_mcost[snd] < s_mcost[fst]) {
// 				s_mcost[fst] = s_mcost[snd];
// 			}
// 		}

// 		step_size <<= 1;
// 		nb_threads >>= 1;

// 		__syncthreads();
// 	}

// 	// Put mcost from block in global
// 	if (tidx == 0) {
// 		mcost[bidx] = s_mcost[0];
// 	}
// }

ll factorial(int n) {
	return (n > 1 ? n*factorial(n-1) : 1);
}

int run_tsp() {
	int nb_cities;
	scanf("%d", &nb_cities);

	int nb_perm = factorial(nb_cities);

	// X coords
	int * x = (int *)malloc(nb_cities*sizeof(int));
	if (!x) {
		printf("malloc error\n");
		exit(1);
	}

	// Y coords
	int * y = (int *)malloc(nb_cities*sizeof(int));
	if (!y) {
		printf("malloc error\n");
		exit(1);
	}

	// Read
	for (int i = 0; i < nb_cities; i++) {
		scanf("%d %d", &x[i], &y[i]);
	}

	// Dist matrix on host
	int * dist = (int *)malloc(nb_cities*nb_cities*sizeof(int));
	if (!dist) {
		printf("malloc error\n");
		exit(1);
	}

	// Calc dist matrix
	int dx, dy;
	for (int i = 0; i < nb_cities; i++) {
		for (int j = 0; j < nb_cities; j++) {
			dx = x[i] - x[j];
			dy = y[i] - y[j];
			dist[i*nb_cities + j] = (int)sqrt(dx*dx + dy*dy);
		}
	}

	free(x);
	free(y);

	// Fact vector on host
	ll * fact = (ll *)malloc(nb_cities*sizeof(ll));
	if (!fact) {
		printf("malloc error\n");
		exit(1);
	}

	// Calc fact
	fact[0] = 1;
	for (int i = 1; i < nb_cities; i++) {
		fact[i] = i * fact[i - 1];
	}

	for (int i = 0; i < nb_cities; i++) {
		for (int j = 0; j < nb_cities; j++) {
			printf("%d ", dist[i*nb_cities + j]);
		}
		printf("\n");
	}
	printf("\n");

	for (int i = 0; i < nb_cities; i++) {
		printf("%lld ", fact[i]);		
	}
	printf("\n");
	
	int min_mcost = INT_MAX;

	clock_t start = clock();
	
	for (int i = 0; i < nb_perm; i++) {
		int cost = calc_perm_cost(i, nb_cities, nb_perm, dist, fact);
		
		if (cost < min_mcost) {
			min_mcost = cost;
		}
	}

	clock_t end = clock();

	// Time
	int msec = (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("%ds %dms\n", msec/1000, msec%1000);

	return min_mcost;
}

int main() {
	int num_instances;
	
	scanf("%d", &num_instances);

	for (int i = 0; i < num_instances; i++) {
		printf("%d\n", run_tsp());
	}

	return 0;
}