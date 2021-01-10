#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>

#define THREAD_NUM 4
#define MAX_NB_CITIES 50

int min(int x, int y) {
	return x > y ? y : x;
}

int calc_perm_cost(long long idx, int n, long long nb_perm, int * dist, long long * fact) {
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
	long long   ** largs = (long long **)args;
	void ** vargs = (void **)args;

	// long long idx, int n, long long nb_perm, int * dist, long long * fact
	int i         =  *iargs[0];
	int nb_cities =  *iargs[1];
	long long  nb_perm   =  *largs[2];
	int * dist    =  (int *)(vargs[3]);
	long long  * fact    =  (long long *)(vargs[4]);
	long long * offset	  =  (long long *)(vargs[5]);

	int perm_per_thread = (nb_perm + THREAD_NUM - 1) / THREAD_NUM;

	int start =     i * perm_per_thread + (int) * offset;
	int end   = start + perm_per_thread + (int) * offset;

	int min_mcost = INT_MAX;
	int cost;

	for (int j = start; j < end; j++) {
		if (j < nb_perm) {

			cost = calc_perm_cost(j, nb_cities, nb_perm, dist, fact);
			if (cost < min_mcost) {
				min_mcost = cost;
			}
		}

	}

	*iargs[0] = min_mcost;
}

long long factorial(int n) {
	return (n > 1 ? n * factorial(n - 1) : 1);
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int myrank, w_size, flag;

	MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
	MPI_Comm_size( MPI_COMM_WORLD, &w_size );

	MPI_Status status;
	MPI_Request * request_a = (MPI_Request *) malloc(sizeof(MPI_Request) * (w_size - 1));
	MPI_Request * request_b = (MPI_Request *) malloc(sizeof(MPI_Request) * (w_size - 1));
	MPI_Request * request_c = (MPI_Request *) malloc(sizeof(MPI_Request) * (w_size - 1));

	if (myrank == 0) {

		int nb_cities, cont, min_dist = INT_MAX;
		scanf("%d", &nb_cities);
		scanf("%d", &nb_cities);

		long long nb_perm = factorial(nb_cities);

		// X coords
		int * x = (int *)malloc(nb_cities * sizeof(int));
		if (!x) {
			printf("malloc error\n");
			exit(1);
		}

		// Y coords
		int * y = (int *)malloc(nb_cities * sizeof(int));
		if (!y) {
			printf("malloc error\n");
			exit(1);
		}

		// Read
		for (int i = 0; i < nb_cities; i++) {
			scanf("%d %d", &x[i], &y[i]);
		}

		// Dist matrix on host
		int * dist = (int *)malloc(nb_cities * nb_cities * sizeof(int));
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
				dist[i * nb_cities + j] = (int)sqrt(dx * dx + dy * dy);
			}
		}



		long long offset = 0;
		long long n_perms_per_node = nb_perm / (w_size - 1);
		for (int i = 1; i < w_size; i++) {
			long long data[4] = {nb_cities, nb_perm, n_perms_per_node, offset};
			MPI_Isend(data, 4, MPI_LONG, i, i, MPI_COMM_WORLD, &request_a[i - 1]);
			MPI_Isend(dist, nb_cities * nb_cities, MPI_INT, i, i * 1024, MPI_COMM_WORLD, &request_b[i - 1]);
			offset += n_perms_per_node;
		}

		for (int i = 1; i < w_size; i++) {
			MPI_Wait(&request_a[i - 1], &status);
			MPI_Test(&request_a[i - 1], &flag, &status);
			if (!flag) {
				printf("Error on sending data {nb_cities, nb_perm} to %d", i);
			}
			MPI_Wait(&request_b[i - 1], &status);
			MPI_Test(&request_b[i - 1], &flag, &status);
			if (!flag) {
				printf("Error on sending dist to %d", i);
			}
		}

		int result, number_amount;
		for (int i = 1; i < w_size; i++) {
			MPI_Recv(&result, 1, MPI_INT, i, i * 1024 * 1024, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_INT, &number_amount);
			if (number_amount != 1) {
				printf("Error in %d on receiving result", i);
			}
			min_dist = min(result, min_dist);
		}

		printf("Result: %d\n", min_dist);

	} else {

		int number_amount;
		long long data[4];

		MPI_Recv(data, 4, MPI_LONG, 0, myrank, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_LONG, &number_amount);
		if (number_amount != 4) {
			printf("Error in %d on receiving data {nb_cities, nb_perm}", myrank);
		}

		int nb_cities = (int) data[0];
		long long nb_perm = data[1];
		long long offset = data[3];
		long long n_perms_per_node = data[2];
		int * dist = (int *) malloc(sizeof(int) * nb_cities * nb_cities);

		MPI_Recv(dist, nb_cities * nb_cities, MPI_INT, 0, myrank * 1024, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &number_amount);
		if (number_amount != nb_cities * nb_cities) {
			printf("Error in %d on receiving dist", myrank);
		}

		// Fact vector on host
		long long * fact = (long long *)malloc(nb_cities * sizeof(long long));
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

		pthread_t * threads = (pthread_t *)malloc(sizeof(pthread_t) * THREAD_NUM);
		if (!threads) {
			exit(1);
		}

		void ***vargs = (void ***)malloc(sizeof(void **)*THREAD_NUM);
		if (!vargs) {
			exit(1);
		}

		// Calong long threads
		for (int i = 0; i < THREAD_NUM; i++) {

			// Along longoc args
			void **args = (void **)malloc(sizeof(void *) * 6);
			if (!args) {
				exit(0);
			}
			vargs[i] = args;

			// Along longoc first arg
			args[0] = (void *)malloc(sizeof(int));
			if (!args[0]) {
				exit(0);
			}

			// Build args
			*((int *)args[0]) = i;
			args[1] = (void *)(&nb_cities);
			args[2] = (void *)(&n_perms_per_node);
			args[3] = (void *)(dist);
			args[4] = (void *)(fact);
			args[5] = (void *)(&offset);

			// long long idx, int n, long long n_perms_per_node, int * dist, long long * fact
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

		MPI_Send(&min_mcost, 1, MPI_INT, 0, myrank * 1024 * 1024, MPI_COMM_WORLD);
	}
	MPI_Finalize();

}

