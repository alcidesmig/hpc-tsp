#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>

#define ll long long
#define THREAD_NUM 16
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

void * calc_perm_cost_iter(void * args) {
	int  ** iargs = (int **)args;
	ll   ** largs = (ll **)args;
	void ** vargs = (void **)args;

	// ll idx, int n, ll nb_perm, int * dist, ll * fact
	int i         = *iargs[0];
	int nb_cities = *iargs[1];
	ll  nb_perm   = *largs[2];
	int * dist    = (int *)(vargs[3]);
	ll  * fact    =  (ll *)(vargs[4]);

	ll perm_per_thread = (nb_perm + THREAD_NUM - 1)/THREAD_NUM;

	ll start =     i * perm_per_thread;
	ll end   = start + perm_per_thread;

	int min_mcost = INT_MAX;
	int cost;

	for (ll j = start; j < end; j++) {
		if (j < nb_perm) {
			cost = calc_perm_cost(j, nb_cities, nb_perm, dist, fact);

			if (cost < min_mcost) {
				min_mcost = cost;
			}
		}
	}

	*iargs[0] = min_mcost;
}

ll factorial(int n) {
	return (n > 1 ? n*factorial(n-1) : 1);
}

int run_tsp() {
	int nb_cities;
	scanf("%d", &nb_cities);

	ll nb_perm = factorial(nb_cities);

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
	
	int min_mcost = INT_MAX;


	pthread_t * threads = (pthread_t *)malloc(sizeof(pthread_t)*THREAD_NUM);
	if (!threads) {
		exit(1);
	}

	void ***vargs = (void ***)malloc(sizeof(void **)*THREAD_NUM);
	if (!vargs) {
		exit(1);
	}

	// Time measurement variables
	struct timeval start, end;
	struct rusage r1, r2;

	// Start
	gettimeofday(&start, 0);
	getrusage(RUSAGE_SELF, &r1);

	// Call threads
	for (int i = 0; i < THREAD_NUM; i++) {
		// Alloc args
		void **args = (void **)malloc(sizeof(void *) * 5);
		if (!args) {
			exit(0);
		}
		vargs[i] = args;

		// Alloc first arg
		args[0] = (void *)malloc(sizeof(int));
		if (!args[0]) {
			exit(0);
		}

		// Build args
		*((int *)args[0]) = i;
		args[1] = (void *)(&nb_cities);
		args[2] = (void *)(&nb_perm);
		args[3] = (void *)(dist);
		args[4] = (void *)(fact);
		
		// ll idx, int n, ll nb_perm, int * dist, ll * fact
		pthread_create(&threads[i], NULL, calc_perm_cost_iter, args);
	}

	// Join thread
	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(threads[i], NULL);

		int ** iargs = (int **)vargs[i];
		int cost = *iargs[0];
		if (cost < min_mcost) {
			min_mcost = cost;
		}

		free(vargs[i][0]);
		free(vargs[i]);
		// printf("Thread %d terminou\n", i);
	}

	// End
	gettimeofday(&end, 0);
	getrusage(RUSAGE_SELF, &r2);

	free(dist);
	free(fact);
	free(vargs);

	// Time
	printf("\nElapsed time:%f sec\tUser time:%f sec\tSystem time:%f sec\n",
	 (end.tv_sec+end.tv_usec/1000000.) - (start.tv_sec+start.tv_usec/1000000.),
	 (r2.ru_utime.tv_sec+r2.ru_utime.tv_usec/1000000.) - (r1.ru_utime.tv_sec+r1.ru_utime.tv_usec/1000000.),
	 (r2.ru_stime.tv_sec+r2.ru_stime.tv_usec/1000000.) - (r1.ru_stime.tv_sec+r1.ru_stime.tv_usec/1000000.));

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