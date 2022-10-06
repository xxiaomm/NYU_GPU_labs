# lab1

NYU GPU Lab 1: Implement a vector processor in CUDA.

## Getting Started

### Switch Modules on CIMS Machines
```bash
- login cims server:
  - ssh xm2074@access.cims.nyu.edu (V7?HF@My)
  - upload file: scp -r /Users/xiao/Desktop/A_GPU_22Spring/labs/lab1/xm2074.cu 
    xm2074@access.cims.nyu.edu:~/gpu/lab1
- ssh to cuda1, cuda2, cuda3, cuda4, or cuda5(prefer)
- 
```

### Module load and Build

```bash
module load cuda-10.2  (from cuda-9.0 and higher(11.4) is OK)
nvcc -o vectorprog xm2074.cu -lm
./vectorprog n
```
