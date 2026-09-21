// Name: Brenna Fesmire
// Vector Dot product on many block and useing shared memory
// nvcc I_DotProductManyBlocksSharedMemory_Fixed_Brenna.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors smaller than the block size.

 Your tasks:
 - Extend the code to launch as many blocks as needed based on a fixed thread count and the vector length.
 - Use **shared memory** within each block to speed up the computation.
 - Pad the input with zeros to fill the last block, if necessary.
 - Perform the final reduction (summing partial results) on the **CPU**.
 - Set the thread count (block size) to 256.
 - Test your code by setting N to different values.
*/

/*
 Purpose:
 To understand that blocks do **not** synchronize with each other during a kernel call.
 In other words, you can't detect when **all blocks** are finished from inside the kernel.
 You can work around this by exiting the kernel, which ensures all blocks have completed.
 Also to learn how to use shared memory to speed up your code.
*/

/*
 Explain what you did to fix the code:

 1. Line 84: Changed the BlockSize.x to be 256, as instructed
 2. Line 88: Changed GridSize.x to calculate based on N
 3. Line 139: Created shared memory that is one BlockSize
 4. Line 141-2: Used indices for the thread id and the block id
 5. Line 148-51: Padded the block with zeroes
 6. Line 155-171: Changed the folding to affect one block at a time, using the thread id index
 7. Line 173-6: Copied each block value to memory after the dot product had been taken
 8. Line 248-254: Commented out an if block in main() that was ignoring additional blocks
 9. Line 269: Scaled the cudaMemCpy to work for more than just one block
 10. Line 277-281: Added up the values that were passed for each block.
 
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 15000 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256; // set to 256, per instructions
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This is the equation built in class
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{

	__shared__ float shared[256]; // confined to the BlockSize.x

	int id = threadIdx.x + blockDim.x*blockIdx.x; // global address
	int thread_id = threadIdx.x; // spot in the block

	if(id < n)
	{
		shared[thread_id] = a[id]*b[id];
	}
	else
	{
		shared[thread_id] = 0.0;
	}
	__syncthreads();
		
	int fold = blockDim.x;
	while(1 < fold)
	{
		if(fold%2 != 0)
		{
			if(thread_id == 0 && (fold - 1) < n)
			{
				shared[0] = shared[0] + shared[fold - 1];
			}
			fold = fold - 1;
		}
		fold = fold/2;
		if(thread_id < fold && (thread_id + fold) < n)
		{
			shared[thread_id] = shared[thread_id] + shared[thread_id + fold];
		}
		__syncthreads();
	}

	if(thread_id == 0)
	{
		c[blockIdx.x] = shared[0]; // copies to the memory
	}
}

// Checking to see if anything went wrong in the dot product.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void cleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	//if(BlockSize.x < N)
	//{
	//	printf("\n\n Your vector size is larger than the block size.");
	//	printf("\n Because we are only using one block this will not work.");
	//	printf("\n Good Bye.\n\n");
	//	exit(0);
	//}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x*sizeof(float), cudaMemcpyDeviceToHost); // scaled it to work for more blocks
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	
	DotGPU = 0.0;
	for(int i = 0; i < GridSize.x; i++)
	{
		DotGPU += C_CPU[i]; // add the values from all the blocks
	}


	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	cleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}