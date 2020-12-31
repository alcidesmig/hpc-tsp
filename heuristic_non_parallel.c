#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<math.h>
#include<time.h>

typedef struct {
	int to_town;
	int dist;
} d_info;

// Find 'town' in 'path[0:depth-1]'
int present(int town, int depth, int *path) { 
	for (int i = 0; i < depth; i++) {
		if (path[i] == town) {
			return 1;
		}
	}

	return 0;
}

void tsp(int nb_towns, int * min_distance, int depth, int current_length, int *path, d_info ** d_matrix, int * dist_to_origin) {
	if (current_length >= *min_distance) {
		return;
	}

	if (depth == nb_towns) {
		current_length += dist_to_origin[path[nb_towns - 1]];

		if (current_length < *min_distance) {
			*min_distance = current_length;
		}
	} else {
		int town, me, dist;
		me = path[depth - 1];

		for (int i = 0; i < nb_towns; i++) {
			town = d_matrix[me][i].to_town;
			if (!present (town, depth, path)) {
				path[depth] = town;
				dist = d_matrix[me][i].dist;
				tsp(nb_towns, min_distance, depth + 1, current_length + dist, path, d_matrix, dist_to_origin);
			}
		}
	}
}

void greedy_shortest_first_heuristic(int nb_towns, int *x, int *y, d_info ** d_matrix, int * dist_to_origin) {
	int dist;
	int *tempdist = (int*)malloc(nb_towns*sizeof(int));

	for (int i = 0; i < nb_towns; i++) {
		for (int j = 0; j < nb_towns; j++) {
			int dx = x[i] - x[j];
			int dy = y[i] - y[j];
			tempdist[j] = dx*dx + dy*dy;
		}

		for (int j = 0; j < nb_towns; j++) {
			int tmp = INT_MAX;
			int town = 0;

			for (int k = 0; k < nb_towns; k++) {
				if (tempdist[k] < tmp) {
					tmp = tempdist[k];
					town = k;
				}
			}

			tempdist[town] = INT_MAX;
			d_matrix[i][j].to_town = town;
			dist = (int)sqrt(tmp);

			d_matrix[i][j].dist = dist;
			if (i == 0) {
				dist_to_origin[town] = dist;
			}
		}
	}

	free (tempdist);
}

int run_tsp() {
	clock_t start, end;

	// Min dist
	int min_distance = INT_MAX;
	
	// Number of towns
	int nb_towns;
	scanf("%d", &nb_towns);

	// Dist matrix
	d_info **d_matrix = (d_info **)malloc(nb_towns*sizeof(d_info *));
	for (int i = 0; i < nb_towns; i++) {
		d_matrix[i] = (d_info *)malloc(nb_towns*sizeof(d_info));
	}

	// Dist to origin
	int *dist_to_origin = (int *)malloc(sizeof(int)*nb_towns);
	
	// Coords
	int * x = (int *)malloc(nb_towns*sizeof(int));
	int * y = (int *)malloc(nb_towns*sizeof(int));
	for (int i = 0; i < nb_towns; i++) {
		scanf("%d %d", &x[i], &y[i]);
	}

	// path taken
	int *path = (int*) malloc(sizeof(int) * nb_towns);
	path[0] = 0;

	// Time it
	start = clock();

	// Heuristic
	greedy_shortest_first_heuristic(nb_towns, x, y, d_matrix, dist_to_origin);

	// Solve
	tsp(nb_towns, &min_distance, 1, 0, path, d_matrix, dist_to_origin);
	
	// Time it
	end = clock();

	// Free all
	free(x);
	free(y);
	free(path);
	free(dist_to_origin);
	for (int i = 0; i < nb_towns; i++) {
		free(d_matrix[i]);
	}
	free(d_matrix);
	
	// Show time
	int msec = (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("%ds %dms\n", msec/1000, msec%1000);

	// Return best cost
	return min_distance;
}

int main (int argc, char **argv) {
	int num_instances;
	
	scanf("%d", &num_instances);

	for (int i = 0; i < num_instances; i++) {
		printf("%d\n", run_tsp());
	}

	return 0;
}