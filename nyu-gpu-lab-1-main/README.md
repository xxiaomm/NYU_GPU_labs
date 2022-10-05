# nyu-gpu-lab-1

NYU GPU Lab 1: Implement a vector processor in CUDA.

## Build

Using make to build the project:

```bash
make
```

An executable named `vectorprog` will be generated under the project root directory.

## Usage

```bash
./vectorprog <vector_size>
```

## Change Number of Blocks & Threads

Open `vectorprog.cu` and change the `BLOCK_NUM` and `BLOCK_SIZE` macros.

For example, to use 4 blocks and 256 threads in a block, change the macros to:

```c++
#define BLOCK_NUM 4
#define BLOCK_SIZE 256
```
