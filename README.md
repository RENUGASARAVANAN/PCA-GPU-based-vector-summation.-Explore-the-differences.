# PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution confi guration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## Aim:

(i) To modify or set the execution configuration of block.xas1023 &1024 and compare the elapsed time obtained on Host and GPU.
(ii) To set the number of threads as 256 and obtain the elapsed time on Host and GPU.

## Procedure:

1.Initialize the device and set the device properties.

2.Allocate memory on the Host for input and output arrays.

3.Initialize input arrays with random values on the Host.
 
4.Allocate memory on the device for input and output arrays,and copy input data from Host to device.

5.Launch a CUDA kernal to perform vector addition on the device.

## Code:

Host-based array summation
 Case1: elements=1024:
```
 #include <stdlib.h>
#include <time.h>
/*
* This example demonstrates a simple vector sum on the host. sumArraysOnHost
* sequentially iterates through vector elements on the host.
*/
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
 for (int idx = 0; idx < N; idx++)
 {
 C[idx] = A[idx] + B[idx];
 }
}
void initialData(float *ip, int size)
{
 // generate different seed for random number
 time_t t;
 srand((unsigned) time(&t));
 for (int i = 0; i < size; i++)
 {
 ip[i] = (float)(rand() & 0xFF) / 10.0f;
 }
 return;
}
int main(int argc, char **argv)
{
 int nElem = 1024;
 size_t nBytes = nElem * sizeof(float);
 float *h_A, *h_B, *h_C;
 h_A = (float *)malloc(nBytes);
 h_B = (float *)malloc(nBytes);
 h_C = (float *)malloc(nBytes);
 initialData(h_A, nElem);
 initialData(h_B, nElem);
 sumArraysOnHost(h_A, h_B, h_C, nElem);
 free(h_A);
 free(h_B);
 free(h_C);
 return(0);
}
```


## Output:
![EXP 1 1 HOST](https://github.com/RENUGASARAVANAN/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/119292258/bbcfae1c-1cc7-4623-a1b5-a51c33c5db3d)


 Case1: elements=1023:
```
#include <stdlib.h>
#include <time.h>
/*
* This example demonstrates a simple vector sum on the host. sumArraysOnHost
* sequentially iterates through vector elements on the host.
*/
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
 for (int idx = 0; idx < N; idx++)
 {
 C[idx] = A[idx] + B[idx];
 }
}
void initialData(float *ip, int size)
{
 // generate different seed for random number
 time_t t;
 srand((unsigned) time(&t));
 for (int i = 0; i < size; i++)
 {
 ip[i] = (float)(rand() & 0xFF) / 10.0f;
 }
 return;
}
int main(int argc, char **argv)
{
 int nElem = 1023;
 size_t nBytes = nElem * sizeof(float);
 float *h_A, *h_B, *h_C;
 h_A = (float *)malloc(nBytes);
 h_B = (float *)malloc(nBytes);
 h_C = (float *)malloc(nBytes);
 initialData(h_A, nElem);
 initialData(h_B, nElem);
 sumArraysOnHost(h_A, h_B, h_C, nElem);
 free(h_A);
 free(h_B);
 free(h_C);
 return(0);
}
```

## Output:

![EXP 1 2 HOST](https://github.com/RENUGASARAVANAN/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/119292258/f77523fb-f4ba-4cd7-b6aa-6dd40c0c9fd4)

GPU BASED ARRAY SUMMATION:

 CASE 2: ELEMENTS =256:
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
/*
* This example demonstrates a simple vector sum on the GPU and on the host.
* sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
* GPU. Only a single thread block is used in this small case, for simplicity.
* sumArraysOnHost sequentially iterates through vector elements on the host.
* This version of sumArrays adds host timers to measure GPU and CPU
* performance.
*/
void checkResult(float *hostRef, float *gpuRef, const int N)
{
 double epsilon = 1.0E-8;
 bool match = 1;
 for (int i = 0; i < N; i++)
 {
 if (abs(hostRef[i] - gpuRef[i]) > epsilon)
 {
 match = 0;
 printf("Arrays do not match!\n");
 printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
 gpuRef[i], i);
 break;
 }
 }
 if (match) printf("Arrays match.\n\n");
 return;
}
void initialData(float *ip, int size)
{
 // generate different seed for random number
 time_t t;
 srand((unsigned) time(&t));
 for (int i = 0; i < size; i++)
 {
 ip[i] = (float)( rand() & 0xFF ) / 10.0f;
 }
 return;
}
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
 for (int idx = 0; idx < N; idx++)
 {
 C[idx] = A[idx] + B[idx];
 }
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) C[i] = A[i] + B[i];
}
int main(int argc, char **argv)
{
 printf("%s Starting...\n", argv[0]);
 // set up device
 int dev = 0;
 cudaDeviceProp deviceProp;
 CHECK(cudaGetDeviceProperties(&deviceProp, dev));
 printf("Using Device %d: %s\n", dev, deviceProp.name);
 CHECK(cudaSetDevice(dev));
 // set up data size of vectors
 int nElem = 1 << 24;
 printf("Vector size %d\n", nElem);
 // malloc host memory
 size_t nBytes = nElem * sizeof(float);
 float *h_A, *h_B, *hostRef, *gpuRef;
 h_A = (float *)malloc(nBytes);
 h_B = (float *)malloc(nBytes);
 hostRef = (float *)malloc(nBytes);
 gpuRef = (float *)malloc(nBytes);
 double iStart, iElaps;
 // initialize data at host side
 iStart = seconds();
 initialData(h_A, nElem);
 initialData(h_B, nElem);
 iElaps = seconds() - iStart;
 printf("initialData Time elapsed %f sec\n", iElaps);
 memset(hostRef, 0, nBytes);
 memset(gpuRef, 0, nBytes);
 // add vector at host side for result checks
 iStart = seconds();
 sumArraysOnHost(h_A, h_B, hostRef, nElem);
 iElaps = seconds() - iStart;
 printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
 // malloc device global memory
 float *d_A, *d_B, *d_C;
 CHECK(cudaMalloc((float**)&d_A, nBytes));
 CHECK(cudaMalloc((float**)&d_B, nBytes));
 CHECK(cudaMalloc((float**)&d_C, nBytes));
 // transfer data from host to device
 CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
 CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
 CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
 // invoke kernel at host side
 int iLen = 256;
 dim3 block (iLen);
 dim3 grid ((nElem + block.x - 1) / block.x);
 iStart = seconds();
 sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
 CHECK(cudaDeviceSynchronize());
 iElaps = seconds() - iStart;
 printf("sumArraysOnGPU <<< %d, %d >>> Time elapsed %f sec\n", grid.x,
 block.x, iElaps);
 // check kernel error
 CHECK(cudaGetLastError()) ;
 // copy kernel result back to host side
 CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
 // check device results
 checkResult(hostRef, gpuRef, nElem);
 // free device global memory
 CHECK(cudaFree(d_A));
 CHECK(cudaFree(d_B));
 CHECK(cudaFree(d_C));
 // free host memory
 free(h_A);
 free(h_B);
 free(hostRef);
 free(gpuRef);
 return(0);
}
```


## Output:

![EXP 1 3 GPU](https://github.com/RENUGASARAVANAN/PCA-GPU-based-vector-summation.-Explore-the-differences./assets/119292258/c34e95cd-87a9-44f8-a737-fff7d8e63f61)

## Result:

i) Thus the program to compare the execution time and result of summing two arrays using CUDA programming by setting block.x=1023 and block.x=1024 has been successfully implemented and verified.

ii)Thus the program to analyse the impact of changing execution configuration on the execution time and result of summing two arrays using CUDA programming by letting each thread handle two elements and setting block.x=256 has been successfully implemented and verified. 
