#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>

const int INF = 1000000000;
const int V = 20010;
const int num_thread = 256;

int n, m, Dist[V][V];
int *device_ptr;
size_t pitch;

inline int ceil(int a, int b) {
	return (a + b - 1) / b;
}

void input(char *inFileName) {
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	Dist[i][j] = 0;
			else		Dist[i][j] = INF;
		}
	}
	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		Dist[a][b] = v;
	}
    fclose(infile);
}

void output(char *outFileName) {
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF)
                Dist[i][j] = INF;
		}
		fwrite(Dist[i], sizeof(int), n, outfile);
	}
    fclose(outfile);
}

//done &= cal(B,	r,	r,	r,	1,	1);
/*
done &= cal(B, r,     r,     0,             r,             1);
done &= cal(B, r,     r,  r +1,  round - r -1,             1);
done &= cal(B, r,     0,     r,             1,             r);
done &= cal(B, r,  r +1,     r,             1,  round - r -1);
*/

__global__ void cal_kernel(int stat, int *device_ptr, int n, size_t pitch, int B, int Round, int block_start_x, int block_start_y, int block_width) {
    int x = threadIdx.x / B;
    int y = threadIdx.x % B;
    int i = (block_start_x + blockIdx.x / block_width) * B + x;
    int j = (block_start_y + blockIdx.x % block_width) * B + y;
    
    /*for (int k = Round * B; k < (Round + 1) * B && k < n; k++) {
        int *i_row = (int*)((char*)device_ptr + i * pitch);
        int *k_row = (int*)((char*)device_ptr + k * pitch);
        if (i_row[k] + k_row[j] < i_row[j]) {
            i_row[j] = i_row[k] + k_row[j];
        }
        __syncthreads();
    }*/
    
    __shared__ int target[16][33];
    __shared__ int a[16][33];
    __shared__ int b[16][33];
    target[x][y] = *((int*)((char*)device_ptr + i * pitch) + j);
    a[x][y] =      *((int*)((char*)device_ptr + i * pitch) + Round * B + y);
    b[x][y] =      *((int*)((char*)device_ptr + (Round * B + x) * pitch) + j);
    __syncthreads();
    if (i >= n || j >= n) return;
    
    int rb = Round * B;
    if (stat == 1) {
        for (int k = 0; k < 16 && rb + k < n; k++) {
            if (target[x][k] + target[k][y] < target[x][y])
                target[x][y] = target[x][k] + target[k][y];
            __syncthreads();
        }
    } else if (stat == 2) {
        for (int k = 0; k < 16 && rb + k < n; k++) {
            if (a[x][k] + b[k][y] < target[x][y])
                target[x][y] = a[x][k] + b[k][y];
            __syncthreads();
        }
    } else if (stat == 3) {
        for (int k = 0; k < 16 && rb + k < n; k++) {
            if (a[x][k] + target[k][y] < target[x][y])
                target[x][y] = a[x][k] + target[k][y];
            __syncthreads();
        }
    } else {
        for (int k = 0; k < 16 && rb + k < n; k++) {
            if (target[x][k] + b[k][y] < target[x][y])
                target[x][y] = target[x][k] + b[k][y];
            __syncthreads();
        }
    }
    
    *((int*)((char*)device_ptr + i * pitch) + j) = target[x][y];
}

bool cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    bool done = true;
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

	for (int b_i =  block_start_x; b_i < block_end_x; ++b_i) {
		for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				int block_internal_start_x 	= b_i * B;
				int block_internal_end_x 	= (b_i +1) * B;
				int block_internal_start_y  = b_j * B;
				int block_internal_end_y 	= (b_j +1) * B;

				if (block_internal_end_x > n)	block_internal_end_x = n;
				if (block_internal_end_y > n)	block_internal_end_y = n;

				for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
						if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
							Dist[i][j] = Dist[i][k] + Dist[k][j];
                            done = false;
                        }
					}
				}
			}
		}
	}
    return done;
}

void show(int32_t input[V][V]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", input[i][j]);
        }
        puts("");
    }
}

void block_FW(int B) {
    cudaMallocPitch(&device_ptr, &pitch, n * sizeof(int), n);
    cudaMemcpy2D(device_ptr, pitch, Dist, V * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
    //show(Dist);
    //memset(Dist, 0, sizeof(Dist));
    //cudaMemcpy2D(Dist, V * sizeof(int), device_ptr, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    //show(Dist);
    
	int round = ceil(n, B);
	for (int r = 0; r < round; ++r) {
        //printf("%d %d\n", r, round);
        /*
		cal(B,	r,	r,	r,	1,	1);

		cal(B, r,     r,     0,             r,             1);
		cal(B, r,     r,  r +1,  round - r -1,             1);
		cal(B, r,     0,     r,             1,             r);
		cal(B, r,  r +1,     r,             1,  round - r -1);

		cal(B, r,     0,     0,            r,             r);
		cal(B, r,     0,  r +1,  round -r -1,             r);
		cal(B, r,  r +1,     0,            r,  round - r -1);
		cal(B, r,  r +1,  r +1,  round -r -1,  round - r -1);*/
        
        int temp = (round - r - 1);
        
        cal_kernel<<<                  1, num_thread>>>(1, device_ptr, n, pitch, B, r,     r,     r,             1);
        
        cal_kernel<<<                  r, num_thread>>>(3, device_ptr, n, pitch, B, r,     r,     0,             r);
        cal_kernel<<<      round - r - 1, num_thread>>>(3, device_ptr, n, pitch, B, r,     r, r + 1, round - r - 1);
        cal_kernel<<<                  r, num_thread>>>(4, device_ptr, n, pitch, B, r,     0,     r,             1);
        cal_kernel<<<      round - r - 1, num_thread>>>(4, device_ptr, n, pitch, B, r, r + 1,     r,             1);
        
        cal_kernel<<<              r * r, num_thread>>>(2, device_ptr, n, pitch, B, r,     0,     0,             r);
        cal_kernel<<<r * (round - r - 1), num_thread>>>(2, device_ptr, n, pitch, B, r,     0, r + 1, round - r - 1);
        cal_kernel<<<r * (round - r - 1), num_thread>>>(2, device_ptr, n, pitch, B, r, r + 1,     0,             r);
        cal_kernel<<<        temp * temp, num_thread>>>(2, device_ptr, n, pitch, B, r, r + 1, r + 1, round - r - 1);
	}
    cudaMemcpy2D(Dist, V * sizeof(int), device_ptr, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
	input(argv[1]);
	int B = atoi(argv[3]);
	block_FW(B);
	output(argv[2]);
	return 0;
}
