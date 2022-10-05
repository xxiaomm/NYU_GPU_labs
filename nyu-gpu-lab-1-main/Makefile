.PHONY: clean

all: vectorprog

vectorprog: vectors.cu
	nvcc -o vectorprog vectors.cu -lm

clean:
	rm -f vectorprog
