// Name: Brenna Fesmire
// Robust Vector Dot product 
// nvcc J_GeneralDotProductWithAtomics_Fixed_Brenna.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors of any length using shared memory to
 reduce the number of global memory accesses. However, since blocks can’t synchronize
 with each other, the final reduction must be handled on the CPU.

 To simplify the GPU-side logic, we’ll add some “pregame” setup and use atomic adds.

 1. Make sure the number of threads per block is a power of 2. This avoids messy edge
    cases during the reduction step. If it’s not a power of 2, print an error message
    and exit. (Without this, you'd have to check if the reduction is even or not,
    add the last element to the first, adjust the loop, etc.)

 2. Calculate the correct number of blocks needed to process the entire vector.
    Then check device properties to ensure the grid and block sizes are within hardware limits.
    Just because it works on your fancy GPU doesn’t mean it will work on your client’s older one.
    If the block or grid size exceeds the device’s capabilities, report the issue and exit gracefully.

 3. It’s inefficient to check inside your kernel if a thread is working past the end of the vector
    on every iteration. Instead, figure out how many extra elements are needed to fill out the grid,
    and pad the vector with zeros. Zero-padding doesn’t affect the dot product (0 * anything = 0).
    Use `cudaMemset` to explicitly zero out your device memory — don’t rely on "getting lucky"
    like you might have in previous assignments.

 4. In previous assignments, we had to do the final reduction on the CPU because we couldn't sync blocks.
    Now, use **atomic adds** to sum partial results directly on the GPU and avoid CPU post-processing.
    Then, copy the final result back to the CPU using `cudaMemcpy`.

    Note: Atomic operations on floats are only supported on GPUs with compute capability 3.0 or higher.
    Use device properties to check this before running the kernel.
    While you’re at it, if multiple GPUs are available, select the best one based on compute capability.

 5. Add any additional bells and whistles to make your code more robust and user-proof.
    Think of edge cases or bad input your client might provide and handle it cleanly.
*/

/*
 Purpose:
 To learn how to use atomic adds to avoid jumping out of the kernel for block synchronization.
 This is also your opportunity to make the code "foolproof" — handling edge cases gracefully.

 At this point, you should understand all the CUDA basics.
 From now on, we’ll focus on refining that knowledge and adding advanced features.
*/

/*
 Explain what you did to fix the code:

 1. defined a function checkBlockSize() to check if the block can hold 2^n threads
 2. GridSize.x = (N - 1)/BlockSize.x + 1; already existed to create the right number of blocks
 3. In setUpDevices(), added a check if the hardware can handle the block size and grid size
 4. In allocateMemory(), calculate the maximum possible vector size so we don't get in our neighbors yard
 5. In allocateMemory(), fill the allocated memory with zeroes ahead of time
 6. In setUpDevices(), added a check if the hardware can handle atomic adds
 7. In setUpDevices(), set the GPU to whichever has the best compute capability
 8. In dotProductGPU(), removed the even/odd check, since the vector that reaches the GPU will always be 2^n
 9. Changed the BLOCK_SIZE to be a multiple of 2 (after checking that it failed otherwise)
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 100000 // Length of the vector
#define BLOCK_SIZE 256 // Threads in a block

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
void checkBlockSize(int); // new function to check the BLOCK_SIZE

void checkBlockSize(int n) // new function to check the BLOCK_SIZE
{
	if((n & (n-1)) != 0) // not a power of 2
	{
		printf("\nWoah! Error, dude! Your block size isn't a power of 2!\n");
		exit(0);
	}
	if(n <= 0)
	{
		printf("\nWoah! Error, dude! Your block size isn't even positive! Have you seen a computer before?\n");
		exit(0);
	}
}

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
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;

	// Brenna added:
	cudaDeviceProp prop; // setting up to read properties of GPUs
	int count;
	cudaGetDeviceCount(&count);

	int bestGPU = -1;
	float bestCap = -1.0;
	int badGPUs = 0;

	if(count == 0)
	{
		printf("\nWoah! Error, dude! You don't have any GPUs to run CUDA!\n");
		exit(0);
	}

	for(int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);

		if(prop.maxThreadsPerBlock < BlockSize.x)
		{
			printf("\nWoah! Error, dude! This is way too many threads per block than your GPU can handle!\n");
			badGPUs++;
			continue; // don't exit on GPU 1, because GPU 2 might be okay
		}
		if(prop.maxGridSize[0] < GridSize.x)
		{
			printf("\nWoah! Error, dude! This is way too many blocks per grid than your GPU can handle!\n");
			badGPUs++;
			continue; // don't exit on GPU 1, because GPU 2 might be okay
		}
		float cap = prop.major + prop.minor * 0.1;
		if(cap < 3.0)
		{
			printf("\nWoah! Error, dude! Your GPU won't be able to do the atomic adds!\n");
			badGPUs++;
			continue; // don't exit on GPU 1, because GPU 2 might be okay
		}

		if(cap > bestCap)
		{
			bestCap = cap;
			bestGPU = i;
		}
	}
	
	if(badGPUs >= count)
	{
		printf("\nWoah! Error, dude! You don't have any GPUs that can do this!\n");
	}
	
	cudaSetDevice(bestGPU); // set the GPU we use to be the one with the most compute capability
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	int extraN = (BlockSize.x * GridSize.x); // maximum possible vector length

	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float)); 
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,extraN*sizeof(float)); // Now, we own all the land - no neighbor's yard to get into
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,extraN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,extraN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMemset(A_GPU, 0, extraN*sizeof(float)); // filling the GPU space with 0s
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(B_GPU, 0, extraN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(C_GPU, 0, extraN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will doting.
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
	int threadIndex = threadIdx.x;
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float c_sh[BLOCK_SIZE];
	
	c_sh[threadIndex] = (a[vectorIndex] * b[vectorIndex]);
	__syncthreads();
	
	int fold = blockDim.x;
	while(1 < fold)
	{
		//if(fold%2 != 0)
		//{
		//	if(threadIndex == 0 && (vectorIndex + fold - 1) < n)
		//	{
		//		c_sh[0] = c_sh[0] + c_sh[0 + fold - 1];
		//	}
		//	fold = fold - 1;
		//}
		fold = fold/2;
		if(threadIndex < fold)
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
			
		}
		__syncthreads();
	}
	
	//c[blockDim.x*blockIdx.x] = c_sh[0];
	if(threadIndex == 0)
	{
		atomicAdd(&c[0], c_sh[0]); // do the atomic add for each block
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < tolerence) 
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

	// Checking threads per block once on CPU
	checkBlockSize(BLOCK_SIZE);
	
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
	//cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&DotGPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost); // from atomic add, we can bring in just one value
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wait until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	//DotGPU = 0.0;
	//for(int i = 0; i < N; i += BlockSize.x)
	//{
	//	DotGPU += C_CPU[i]; // C_GPU was copied into C_CPU. 
	//}

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


