CC = nvcc
CFLAGS = -O2 -I include

# V0
SRC_V0 = src/v0.cu
OBJ_V0 = $(SRC_V0:src/%.c=build/%.o)
BIN_V0 = build/v0

# V1
SRC_cuda_bitonic_sort = src/cuda_bitonic_sort.cu
OBJ_cuda_bitonic_sort = $(SRC_cuda_bitonic_sort:src/%.c=build/%.o)
BIN_cuda_bitonic_sort = build/cuda_bitonic_sort

all: $(BIN_V0) $(BIN_cuda_bitonic_sort)

build/%.o: src/%.c
	mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for V0
$(BIN_V0): $(OBJ_V0)
	$(CC) $(OBJ_V0) -o $@ $(LDFLAGS)

#Rule for V1
$(BIN_cuda_bitonic_sort): $(OBJ_cuda_bitonic_sort)
	$(CC) $(OBJ_cuda_bitonic_sort) -o $@
	rm -rf build/*.o

clean:
	rm -rf build
