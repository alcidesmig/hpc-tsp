mpi:
		mpicc tsp_mpi.c -o out_mpi -lm -lpthread
run_mpi:
		mpicc tsp_mpi.c -o out_mpi -lm -lpthread
		mpirun -n 8 ./out_mpi < ../cases/case5.in
