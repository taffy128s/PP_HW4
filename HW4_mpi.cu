#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <omp.h>
#include <mpi.h>

const int INF = 1000000000;
const int V = 20010;
const int num_thread = 256;
clock_t begin, end;
double comm_time = 0;

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

__global__ void cal_kernel(int stat, int *device_ptr, int n, size_t pitch, int B, int Round, int block_start_x, int block_start_y, int block_width) {
    int x = threadIdx.x / B;
    int y = threadIdx.x % B;
    int i = (block_start_x + blockIdx.x / block_width) * B + x;
    int j = (block_start_y + blockIdx.x % block_width) * B + y;

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

void getHostToDeviceP(cudaMemcpy3DParms *p, int B, int bPosX, int bPosY, int width, int height) {
    memset(p, 0, sizeof(cudaMemcpy3DParms));
    p->dstPtr.pitch = pitch;
    p->dstPtr.ptr = device_ptr;
    p->dstPtr.xsize = n;
    p->dstPtr.ysize = n;

    p->srcPtr.pitch = V * sizeof(int);
    p->srcPtr.ptr = Dist;
    p->srcPtr.xsize = n;
    p->srcPtr.ysize = n;

    int tempXB = bPosX * B;
    int tempYB = bPosY * B;

    p->dstPos.x = tempXB * 4;
    p->dstPos.y = tempYB;
    p->srcPos.x = tempXB * 4;
    p->srcPos.y = tempYB;

    p->extent.width = width * 4;
    p->extent.height = height;
    p->extent.depth = 1;

    if (tempXB + width > n) {
        p->extent.width = (n - tempXB) * 4;
    }
    if (tempYB + height > n) {
        p->extent.height = n - tempYB;
    }

    p->kind = cudaMemcpyHostToDevice;

}

void getDeviceToHostP(cudaMemcpy3DParms *p, int B, int bPosX, int bPosY, int width, int height) {
    memset(p, 0, sizeof(cudaMemcpy3DParms));
    p->srcPtr.pitch = pitch;
    p->srcPtr.ptr = device_ptr;
    p->srcPtr.xsize = n;
    p->srcPtr.ysize = n;

    p->dstPtr.pitch = V * sizeof(int);
    p->dstPtr.ptr = Dist;
    p->dstPtr.xsize = n;
    p->dstPtr.ysize = n;

    int tempXB = bPosX * B;
    int tempYB = bPosY * B;

    p->dstPos.x = tempXB * 4;
    p->dstPos.y = tempYB;
    p->srcPos.x = tempXB * 4;
    p->srcPos.y = tempYB;

    p->extent.width = width * 4;
    p->extent.height = height;
    p->extent.depth = 1;

    if (tempXB + width > n) {
        p->extent.width = (n - tempXB) * 4;
    }
    if (tempYB + height > n) {
        p->extent.height = n - tempYB;
    }

    p->kind = cudaMemcpyDeviceToHost;

}

void mySend(int dst, int B, int bPosX, int bPosY, int width, int height) {
    int x_start = bPosX * B, x_end = x_start + width;
    int y_start = bPosY * B, y_end = y_start + height;
    if (x_start >= n || y_start >= n)
        return;

    if (x_end > n) x_end = n;
    if (y_end > n) y_end = n;

    begin = clock();
    for (int i = y_start; i < y_end; i++) {
        MPI_Send(&Dist[i][x_start], x_end - x_start, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }
    end = clock();
    comm_time += (double) (end - begin) / CLOCKS_PER_SEC;
}

void myRecv(int src, int B, int bPosX, int bPosY, int width, int height) {
    int x_start = bPosX * B, x_end = x_start + width;
    int y_start = bPosY * B, y_end = y_start + height;
    if (x_start >= n || y_start >= n)
        return;

    if (x_end > n) x_end = n;
    if (y_end > n) y_end = n;

    begin = clock();
    for (int i = y_start; i < y_end; i++) {
        MPI_Recv(&Dist[i][x_start], x_end - x_start, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    end = clock();
    comm_time += (double) (end - begin) / CLOCKS_PER_SEC;
}

void block_FW(int B) {
    cudaMemcpy3DParms p;
    int tid;
    MPI_Comm_rank(MPI_COMM_WORLD, &tid);

    cudaSetDevice(tid);
    cudaMallocPitch(&device_ptr, &pitch, n * sizeof(int), n);
    cudaMemcpy2D(device_ptr, pitch, Dist, V * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);

    int round = ceil(n, B);
    for (int r = 0; r < round; r++) {
        int temp = (round - r - 1);

        if (tid == 0) {
            cal_kernel<<<1, num_thread>>>(1, device_ptr, n, pitch, B, r,     r,     r,             1);
            getDeviceToHostP(&p, B, r, r, B, B);
            cudaMemcpy3D(&p);
            mySend(1, B, r, r, B, B);
        } else {
            myRecv(0, B, r, r, B, B);
            getHostToDeviceP(&p, B, r, r, B, B);
            cudaMemcpy3D(&p);
        }

        //if (r < 1) {

        if (tid == 0) {
            cal_kernel<<<temp, num_thread>>>(3, device_ptr, n, pitch, B, r,     r, r + 1,          temp);
            cal_kernel<<<temp, num_thread>>>(4, device_ptr, n, pitch, B, r, r + 1,     r,             1);
            getDeviceToHostP(&p, B, r + 1, r, temp * B, B);
            cudaMemcpy3D(&p);
            getDeviceToHostP(&p, B, r, r + 1, B, temp * B);
            cudaMemcpy3D(&p);
            mySend(1, B, r + 1, r, temp * B, B);
            mySend(1, B, r, r + 1, B, temp * B);
            myRecv(1, B, 0, r, r * B, B);
            myRecv(1, B, r, 0, B, r * B);
            getHostToDeviceP(&p, B, 0, r, r * B, B);
            cudaMemcpy3D(&p);
            getHostToDeviceP(&p, B, r, 0, B, r * B);
            cudaMemcpy3D(&p);
        } else {
            cal_kernel<<<   r, num_thread>>>(3, device_ptr, n, pitch, B, r,     r,     0,             r);
            cal_kernel<<<   r, num_thread>>>(4, device_ptr, n, pitch, B, r,     0,     r,             1);
            getDeviceToHostP(&p, B, 0, r, r * B, B);
            cudaMemcpy3D(&p);
            getDeviceToHostP(&p, B, r, 0, B, r * B);
            cudaMemcpy3D(&p);
            myRecv(0, B, r + 1, r, temp * B, B);
            myRecv(0, B, r, r + 1, B, temp * B);
            mySend(0, B, 0, r, r * B, B);
            mySend(0, B, r, 0, B, r * B);
            getHostToDeviceP(&p, B, r + 1, r, temp * B, B);
            cudaMemcpy3D(&p);
            getHostToDeviceP(&p, B, r, r + 1, B, temp * B);
            cudaMemcpy3D(&p);
        }

        if (tid == 0) {
            cal_kernel<<<      r * r, num_thread>>>(2, device_ptr, n, pitch, B, r,     0,     0,             r);
            cal_kernel<<<temp * temp, num_thread>>>(2, device_ptr, n, pitch, B, r, r + 1, r + 1,          temp);
        } else {
            cal_kernel<<<   r * temp, num_thread>>>(2, device_ptr, n, pitch, B, r, r + 1,     0,             r);
            cal_kernel<<<   r * temp, num_thread>>>(2, device_ptr, n, pitch, B, r,     0, r + 1,          temp);
        }

        //}

    }

    if (tid == 0) {
        cudaMemcpy2D(Dist, V * sizeof(int), device_ptr, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
	input(argv[1]);
	int B = atoi(argv[3]);
	block_FW(B);
	output(argv[2]);
    printf("comm time: %f\n", comm_time);
	return 0;
}
