/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s), memory allocation, data movement, etc. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

/* To index element (i,j) of a 2D square array of dimension NxN stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

#define BLOCK_NUM 8
#define BLOCK_SIZE 500

#define TILE_SIZE 50



/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);
void  gpu_optimized_heat_dist(float *, unsigned int, unsigned int);

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4) {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU version, 2: GPU opitmized version\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground ) {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 100;
  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 150;
  

  switch(type_of_device) {
	case 0: printf("CPU sequential version:\n");
			start = clock();
			seq_heat_dist(playground, N, iterations);
			end = clock();
			break;
		
	case 1: printf("GPU version:\n");
			start = clock();
			gpu_heat_dist(playground, N, iterations); 
			end = clock();  
			break;
			
	case 2: printf("GPU optimized version:\n");
			start = clock();
			gpu_optimized_heat_dist(playground, N, iterations); 
			end = clock();  
			break;
			
	default: printf("Invalid device type\n");
			 exit(1);
  }
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken = %lf\n", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }

  
}

__global__ void gpu_kernel(float* playground, float* temp, unsigned int N, int stride) {
  int pos = blockIdx.x * blockDim.x * threadIdx.x;

  for (int j = pos * stride; j < N*N && j < (pos+1)*stride; j++) {
    int row = j / N, col = j % N;
    if (row == 0 || col == 0 || row == N-1 || col == N-1) continue;
     
    temp[index(row, col, N)] = (playground[index(row-1,col,N)] 
                                + playground[index(row+1, col, N)] 
                                + playground[index(row, col-1, N)] 
                                + playground[index(row, col+1, N)]) / 4.0;

    playground[index(row, col, N)] = temp[index(row, col, N)];
    // printf("a+j\n");
  }
}


/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations) {
  float *pgd;
  float *temp;
  
  if (cudaErrorMemoryAllocation == cudaMalloc( (void **)&pgd, N * N * sizeof(float))) {
    printf("Error occurs when allocating memory for playground.\n");
    exit(0);
  }
  if (cudaErrorMemoryAllocation == cudaMalloc( (void **)&temp, N * N * sizeof(float))) {
    printf("Error occurs when allocating memory for temp.\n");
    exit(0);
  }  
  
  cudaMemcpy(pgd, playground, N * N * sizeof(float), cudaMemcpyHostToDevice);
  
  int stride = ceil(N*N / (float)(BLOCK_NUM * BLOCK_SIZE));
  for (int i = 0; i < iterations; i++) {
    gpu_kernel<<<BLOCK_NUM, BLOCK_SIZE>>>(pgd, temp, N, stride);
    // cudaDeviceSynchronize();
    // cudaMemcpy(playgroundd, tempd, num_bytes, cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(playground, pgd, N * N * sizeof(float), cudaMemcpyDeviceToHost);
}


/***************** The GPU optimized version: Write your code here *********************/
__global__ void gpu_optimized_kernel(float *playground, unsigned int N) {
  __shared__ float shared[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.x * TILE_SIZE + threadIdx.x;
  int col = blockIdx.y * TILE_SIZE + threadIdx.y;
  for(int i = 0; i < TILE_SIZE; i++) {
    for (int j = 0; j < TILE_SIZE; j++) {
      if (row+i == 0 ||row+i == N-1 || col+j == 0 || col+j == N-1) continue;
      shared[i][j] = (playground[index(row+i-1, col+j, N)]
                    + playground[index(row+i+1, col+j, N)]
                    + playground[index(row+i, col+j-1, N)]
                    + playground[index(row+i, col+j+1, N)]) / 4.0;
    }
  }
  __syncthreads();

  for (int i = 0; i < TILE_SIZE; i++) {
    for (int j = 0; j < TILE_SIZE; j++) {
      if (row+i == 0 ||row+i == N-1 || col+j == 0 || col+j == N-1) continue;
      
      playground[index(row+i, col+j, N)] = shared[i][j];
    }
  }

}


/*
nvcc -o heatdist heatdist.cu
nvcc -o xm2074 xm2074.cu
[xm2074@cuda5 lab2]$ ./xm2074 1000 100000 2
GPU optimized version:
Time taken = 0.180000
[xm2074@cuda5 lab2]$ ./xm2074 2000 100000 2
GPU optimized version:
Time taken = 0.170000
[xm2074@cuda5 lab2]$ ./xm2074 4000 100000 2
GPU optimized version:
Time taken = 0.190000
[xm2074@cuda5 lab2]$ ./xm2074 8000 100000 2
GPU optimized version:
Time taken = 0.260000
[xm2074@cuda5 lab2]$ ./xm2074 16000 100000 2
GPU optimized version:
Time taken = 0.430000
*/
/* This function can call one or more kernels if you want ********************/
void  gpu_optimized_heat_dist(float * playground, unsigned int N, unsigned int iterations) {
  float *pgd;

  cudaMalloc((void **)&pgd, N * N * sizeof(float));
  cudaMemcpy(pgd, playground, N * N * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 grid(N / TILE_SIZE, N / TILE_SIZE, 1);
  dim3 block(TILE_SIZE, TILE_SIZE, 1);

  for (int i = 0; i < iterations; i++) {
    gpu_optimized_kernel<<<grid, block>>>(pgd, N);
  }
  
  cudaMemcpy(playground, pgd, N * N * sizeof(float), cudaMemcpyHostToDevice);

}


/*
V7?HF@My
ssh cuda5
module load cuda-10.2
nvcc -o heatdist heatdist.cu
nvcc -o xm2074 xm2074.cu
*/