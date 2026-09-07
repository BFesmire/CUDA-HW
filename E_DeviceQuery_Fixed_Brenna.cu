// Name: Brenna Fesmire
// Device query
// nvcc E_DeviceQuery_Fixed_Brenna.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

/*
 Explain what you did to fix the code:
 
 Added everything below the "Brenna added" comment on line 84
 Changed the print statements for sharedMemPerBlock and regsPerBlock, as both said per mp before
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name); // What GPU?
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); // GPU version
		printf("Clock rate: %d\n", prop.clockRate); // GPU clock rate, in kiloHertz
		printf("Device copy overlap: "); // Can the GPU overlap memory with kernel execution?
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : "); // Can the os kill a kernel that's been running too long?
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem); // Total global memory on the GPU
		printf("Total constant Mem: %ld\n", prop.totalConstMem); // Memory available to CUDA programs
		printf("Max mem pitch: %ld\n", prop.memPitch); // memory between rows in 2D memory allocation
		printf("Texture Alignment: %ld\n", prop.textureAlignment); // Alignment helps CUDA be efficient with memory
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per block: %ld\n", prop.sharedMemPerBlock); // "Apartment storage"
		printf("Registers per block: %d\n", prop.regsPerBlock); // Registers per block
		printf("Threads in warp: %d\n", prop.warpSize); // Number of threads executeed at once
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");

		// Brenna added:
		printf("Brenna Added:\n\n");
		printf("L2 cache size: %d bytes\n", prop.l2CacheSize); // Memory cache
		printf("Memory clock rate: %d kHz\n", prop.memoryClockRate); // Speed of the memory
		printf("Memory bus width: %d bits\n", prop.memoryBusWidth); // How many bits can go between GPU and memory
		printf("ECC enabled: "); // ECC is "Error-Correcting Code"
		if (prop.ECCEnabled)
			printf("Yes\n");
		else
			printf("No\n");

		printf("Concurrent kernels: "); // Can the GPU launch more than one kernel?
		if (prop.concurrentKernels)
			printf("Yes\n");
		else
			printf("No\n");

		printf("Unified addressing: "); // Can the CPU and GPU use the same address space in memory?
		if (prop.unifiedAddressing)
			printf("Yes\n");
		else
			printf("No\n");

		printf("Can map host memory: "); // Can CUDA access CPU memory from the GPU?
		if (prop.canMapHostMemory)
			printf("Yes\n");
		else
			printf("No\n");

		printf("Compute mode: %d\n", prop.computeMode); // What are the rules for CUDA using the GPU?
	
		printf("Integrated GPU: "); // Does the GPU share resources with the CPU?
		if (prop.integrated)
			printf("Yes\n"); // Yes, resources are shared
		else
			printf("No\n"); // No, the resources are seperate

		printf("Maximum resident threads per multiprocessor: %d\n", // Maximum threads allowed on a multiprocessor
       		prop.maxThreadsPerMultiProcessor);

		printf("Maximum resident blocks per multiprocessor: %d\n", // Maximum blocks allowed on a multiprocessor
    	   prop.maxBlocksPerMultiProcessor);

		printf("\n");
		}	
	return(0);
}

