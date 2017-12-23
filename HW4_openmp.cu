#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <omp.h>

const int INF = 1000000000;
const int V = 20010;
const int num_thread = 256;

int n, m, Dist[V][V];
int *device_ptr[2];
size_t pitch[2];

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

void getP(cudaMemcpy3DPeerParms *p, int B, int src, int dst, int bPosX, int bPosY, int width, int height) {
    memset(p, 0, sizeof(cudaMemcpy3DPeerParms));
    p->dstDevice = dst;
    p->dstPtr.pitch = pitch[dst];
    p->dstPtr.ptr = device_ptr[dst];
    p->dstPtr.xsize = n;
    p->dstPtr.ysize = n;

    p->srcDevice = src;
    p->srcPtr.pitch = pitch[src];
    p->srcPtr.ptr = device_ptr[src];
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

    if (tempXB + width > n) {
        p->extent.width = (n - tempXB) * 4;
    }
    if (tempYB + height > n) {
        p->extent.height = n - tempYB;
    }

    p->extent.depth = 1;
}

void block_FW(int B) {
    cudaMemcpy3DPeerParms p, tp[4];
    //cudaStream_t stream[2][2];
    //cudaEvent_t event[2];

    #pragma omp parallel num_threads(2)
    {
        int id = omp_get_thread_num();
        cudaSetDevice(id);
        /*if (id == 0)
            cudaDeviceEnablePeerAccess(1, 0);
        else
            cudaDeviceEnablePeerAccess(0, 0);*/
        //cudaStreamCreate(&stream[id][0]);
        //cudaStreamCreate(&stream[id][1]);
        //cudaEventCreate(&event[id]);
        cudaMallocPitch(&device_ptr[id], &pitch[id], n * sizeof(int), n);
        cudaMemcpy2D(device_ptr[id], pitch[id], Dist, V * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
    }

    cudaThreadSynchronize();

    int round = ceil(n, B);
	for (int r = 0; r < round; ++r) {
        int temp = (round - r - 1);

        cudaThreadSynchronize();

        cudaSetDevice(0);
        cal_kernel<<<1, num_thread>>>(1, device_ptr[0], n, pitch[0], B, r,     r,     r,             1);
        cudaThreadSynchronize();
        getP(&p, B, 0, 1, r, r, B, B);
        cudaMemcpy3DPeer(&p);

        cudaThreadSynchronize();

        #pragma omp parallel num_threads(2)
        {
            int tid = omp_get_thread_num();
            cudaSetDevice(tid);
            if (tid == 0) {
                cal_kernel<<<temp, num_thread>>>(3, device_ptr[tid], n, pitch[tid], B, r,     r, r + 1,          temp);
                cal_kernel<<<temp, num_thread>>>(4, device_ptr[tid], n, pitch[tid], B, r, r + 1,     r,             1);
                getP(&tp[0], B, 0, 1, r + 1, r, temp * B, B);
                cudaMemcpy3DPeer(&tp[0]);
                getP(&tp[1], B, 0, 1, r, r + 1, B, temp * B);
                cudaMemcpy3DPeer(&tp[1]);
            } else {
                cal_kernel<<<   r, num_thread>>>(3, device_ptr[tid], n, pitch[tid], B, r,     r,     0,             r);
                cal_kernel<<<   r, num_thread>>>(4, device_ptr[tid], n, pitch[tid], B, r,     0,     r,             1);
                getP(&tp[2], B, 1, 0, 0, r, r * B, B);
                cudaMemcpy3DPeer(&tp[2]);
                getP(&tp[3], B, 1, 0, r, 0, B, r * B);
                cudaMemcpy3DPeer(&tp[3]);
            }
        }

        cudaThreadSynchronize();

        #pragma omp parallel num_threads(2)
        {
            int tid = omp_get_thread_num();
            cudaSetDevice(tid);
            if (tid == 0) {
                cal_kernel<<<      r * r, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r,     0,     0,             r);
                cal_kernel<<<temp * temp, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r, r + 1, r + 1,          temp);

            } else {
                cal_kernel<<<   r * temp, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r, r + 1,     0,             r);
                cal_kernel<<<   r * temp, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r,     0, r + 1,          temp);
            }
        }

        cudaThreadSynchronize();

        /*cudaSetDevice(0);
        getP(&p, B, 0, 1,     0,     0,    r * B, r * B);
        cudaMemcpy3DPeer(&p);
        getP(&p, B, 0, 1, r + 1, r + 1, temp * B, temp * B);
        cudaMemcpy3DPeer(&p);

        cudaSetDevice(1);
        getP(&p, B, 1, 0,     0, r + 1,    r * B, temp * B);
        cudaMemcpy3DPeer(&p);
        getP(&p, B, 1, 0, r + 1,     0, temp * B, r * B);
        cudaMemcpy3DPeer(&p);*/

        //cudaThreadSynchronize();


        //cudaThreadSynchronize();

        /*#pragma omp parallel num_threads(2)
        {
            int tid = omp_get_thread_num();
            cudaSetDevice(tid);
            cal_kernel<<<1, num_thread, 6336, stream[tid][0]>>>(1, device_ptr[tid], n, pitch[tid], B, r,     r,     r,             1);
        }

        cudaThreadSynchronize();

        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < r; i += chunk) {
            int tid = omp_get_thread_num(), blks;
            cudaSetDevice(tid);

            if (i + chunk > r)
                blks = r - i;
            else
                blks = chunk;

            cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(3, device_ptr[tid], n, pitch[tid], B, r,     r,     i,             blks);
            cudaEventRecord(event[tid], stream[tid][0]);

            int dst;
            if (tid == 0)
                dst = 1;
            else
                dst = 0;

            getP(&tp[tid], B, tid, dst, i, r, blks * B, B);
            cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
            cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
        }
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < temp; i += chunk) {
            int tid = omp_get_thread_num(), blks;
            cudaSetDevice(tid);

            if (i + chunk > temp)
                blks = temp - i;
            else
                blks = chunk;

            cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(3, device_ptr[tid], n, pitch[tid], B, r,     r, r + 1 + i,          blks);
            cudaEventRecord(event[tid], stream[tid][0]);

            int dst;
            if (tid == 0)
                dst = 1;
            else
                dst = 0;

            getP(&tp[tid], B, tid, dst, r + 1 + i, r, blks * B, B);
            cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
            cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
        }
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < r; i += chunk) {
            int tid = omp_get_thread_num(), blks;
            cudaSetDevice(tid);

            if (i + chunk > r)
                blks = r - i;
            else
                blks = chunk;

            cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(4, device_ptr[tid], n, pitch[tid], B, r,     i,     r,             1);
            cudaEventRecord(event[tid], stream[tid][0]);

            int dst;
            if (tid == 0)
                dst = 1;
            else
                dst = 0;

            getP(&tp[tid], B, tid, dst, r, i, B, B * blks);
            cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
            cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
        }
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < temp; i += chunk) {
            int tid = omp_get_thread_num(), blks;
            cudaSetDevice(tid);

            if (i + chunk > temp)
                blks = temp - i;
            else
                blks = chunk;

            cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(4, device_ptr[tid], n, pitch[tid], B, r, r + 1 + i,     r,             1);
            cudaEventRecord(event[tid], stream[tid][0]);

            int dst;
            if (tid == 0)
                dst = 1;
            else
                dst = 0;

            getP(&tp[tid], B, tid, dst, r, r + 1 + i, B, B * blks);
            cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
            cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
        }
        cudaThreadSynchronize();
        for (int i = 0; i < r; i++) {
            #pragma omp parallel for num_threads(2)
            for (int j = 0; j < r; j += chunk) {
                int tid = omp_get_thread_num(), blks;
                cudaSetDevice(tid);

                if (j + chunk > r)
                    blks = r - j;
                else
                    blks = chunk;

                cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(2, device_ptr[tid], n, pitch[tid], B, r,     i,     j,             blks);
                cudaEventRecord(event[tid], stream[tid][0]);

                int dst;
                if (tid == 0)
                    dst = 1;
                else
                    dst = 0;

                getP(&tp[tid], B, tid, dst, j, i, B * blks, B);
                cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
                cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
            }
        }
        for (int i = 0; i < r; i++) {
            #pragma omp parallel for num_threads(2)
            for (int j = 0; j < temp; j++) {
                int tid = omp_get_thread_num(), blks;
                cudaSetDevice(tid);

                if (j + chunk > temp)
                    blks = temp - j;
                else
                    blks = chunk;

                cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(2, device_ptr[tid], n, pitch[tid], B, r,     i, r + 1 + j,          blks);
                cudaEventRecord(event[tid], stream[tid][0]);

                int dst;
                if (tid == 0)
                    dst = 1;
                else
                    dst = 0;

                getP(&tp[tid], B, tid, dst, r + 1 + j, i, B * blks, B);
                cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
                cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
            }
        }
        for (int i = 0; i < temp; i++) {
            #pragma omp parallel for num_threads(2)
            for (int j = 0; j < r; j++) {
                int tid = omp_get_thread_num(), blks;
                cudaSetDevice(tid);

                if (j + chunk > r)
                    blks = r - j;
                else
                    blks = chunk;

                cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(2, device_ptr[tid], n, pitch[tid], B, r, r + 1 + i,     j,             blks);
                cudaEventRecord(event[tid], stream[tid][0]);

                int dst;
                if (tid == 0)
                    dst = 1;
                else
                    dst = 0;

                getP(&tp[tid], B, tid, dst, j, r + 1 + i, B * blks, B);
                cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
                cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
            }
        }
        for (int i = 0; i < temp; i++) {
            #pragma omp parallel for num_threads(2)
            for (int j = 0; j < temp; j++) {
                int tid = omp_get_thread_num(), blks;
                cudaSetDevice(tid);

                if (j + chunk > temp)
                    blks = temp - j;
                else
                    blks = chunk;

                cal_kernel<<<   blks, num_thread, 6336, stream[tid][0]>>>(2, device_ptr[tid], n, pitch[tid], B, r, r + 1 + i, r + 1 + j,          blks);
                cudaEventRecord(event[tid], stream[tid][0]);

                int dst;
                if (tid == 0)
                    dst = 1;
                else
                    dst = 0;

                getP(&tp[tid], B, tid, dst, r + 1 + j, r + 1 + i, B * blks, B);
                cudaStreamWaitEvent(stream[tid][1], event[tid], 0);
                cudaMemcpy3DPeerAsync(&tp[tid], stream[tid][1]);
            }
        }
        cudaThreadSynchronize();*/
        /*cudaSetDevice(0);
        cudaThreadSynchronize();
        cal_kernel<<<1, num_thread>>>(1, device_ptr[0], n, pitch[0], B, r,     r,     r,             1);

        cudaThreadSynchronize();
        cudaSetDevice(1);
        getP(&p, B, 0, 1, r, r, B, B);
        cudaMemcpy3DPeer(&p);*/

        /*#pragma omp parallel num_threads(2)
        {
            int tid = omp_get_thread_num();
            cudaSetDevice(tid);
            if (tid == 0) {
                cal_kernel<<<   r, num_thread>>>(3, device_ptr[tid], n, pitch[tid], B, r,     r,     0,             r);
                cal_kernel<<<temp, num_thread>>>(3, device_ptr[tid], n, pitch[tid], B, r,     r, r + 1,          temp);
            } else {
                //if (r < 1) {
                cal_kernel<<<   r, num_thread>>>(4, device_ptr[tid], n, pitch[tid], B, r,     0,     r,             1);
                cal_kernel<<<temp, num_thread>>>(4, device_ptr[tid], n, pitch[tid], B, r, r + 1,     r,             1);
                //}
            }
        }

        cudaThreadSynchronize();
        cudaSetDevice(1);
        getP(&p, B, 0, 1, 0, r, n, B);
        cudaMemcpy3DPeer(&p);

        //if (r < 1) {
        cudaThreadSynchronize();

        cudaSetDevice(0);
        getP(&p, B, 1, 0, r, 0, B, n);
        cudaMemcpy3DPeer(&p);
        cudaThreadSynchronize();
        //cudaThreadSynchronize();

        #pragma omp parallel num_threads(2)
        {
            int tid = omp_get_thread_num();
            cudaSetDevice(tid);
            if (tid == 0) {
                cal_kernel<<<   r * r, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r,     0,     0,             r);
                cal_kernel<<<r * temp, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r,     0, r + 1,          temp);
            } else {
                cal_kernel<<<   r * temp, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r, r + 1,     0,             r);
                cal_kernel<<<temp * temp, num_thread>>>(2, device_ptr[tid], n, pitch[tid], B, r, r + 1, r + 1,          temp);
            }
        }

        cudaThreadSynchronize();
        cudaSetDevice(1);
        getP(&p, B, 0, 1,     0,     0,    r * B, r * B);
        cudaMemcpy3DPeer(&p);
        getP(&p, B, 0, 1, r + 1,     0, temp * B, r * B);
        cudaMemcpy3DPeer(&p);

        cudaThreadSynchronize();

        cudaSetDevice(0);
        getP(&p, B, 1, 0,     0, r + 1,    r * B, temp * B);
        cudaMemcpy3DPeer(&p);
        getP(&p, B, 1, 0, r + 1, r + 1, temp * B, temp * B);
        cudaMemcpy3DPeer(&p);

        //}
        cudaThreadSynchronize();*/
	}
    cudaThreadSynchronize();
    cudaSetDevice(0);
    cudaMemcpy2D(Dist, V * sizeof(int), device_ptr[0], pitch[0], n * sizeof(int), n, cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
	input(argv[1]);
	int B = atoi(argv[3]);
	block_FW(B);
	output(argv[2]);
	return 0;
}
