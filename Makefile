CC = nvcc
CFLAGS = -O2 -I include

SRC_cuda_bitonic_sort = src/cuda_bitonic_sort.cu
OBJ_cuda_bitonic_sort = $(SRC_cuda_bitonic_sort:src/%.c=build/%.o)
BIN_cuda_bitonic_sort = build/cuda_bitonic_sort

all: $(BIN_cuda_bitonic_sort)

build/%.o: src/%.c
	mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_cuda_bitonic_sort): $(OBJ_cuda_bitonic_sort)
	$(CC) $(OBJ_cuda_bitonic_sort) -o $@
	rm -rf build/*.o

clean:
	rm -rf build
