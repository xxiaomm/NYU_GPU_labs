## lab2

NYU GPU Lab 2: Write a program to determine the heat distribution in a space using synchronous iteration on a GPU.


### Usage

```bash
nvcc -o heatdist heatdist.cu
./heatdist -> usage
```

### Change Number of Blocks & Threads

Open `.cu` file and change the `BLOCK_NUM` and `THREADS_NUM` macros.


```c
#define BLOCK_NUM 4       // one grid contains 4 blocks.
#define THREADS_NUM 500   // one block contains 500 threads.   
```