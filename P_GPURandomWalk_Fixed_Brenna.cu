// Name: Brenna Fesmire
// GPU random walk. 
// nvcc P_GPURandomWalk_Fixed_Brenna.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Explain what you did to fix the code:
 1. Created a kernel to run the GPU random walk in
 2. Assigned each thread a walk number (same equation as a global thread id that we used before)
 3. Gave each thread / walk a different seed, but marked them all as starting in the same location (0,0)
 4. For each walk, every step was randomized by choosing a random number
 	and using an over/under system to determine the direction of x and y.
 5. Save the final position of each walk
 6. In main, created two pointers (arrays) to hold the final positions of the random walks
 7. Created unified memory for the final positions
 8. Made a random seed to pass into the kernel to further randomize each individual walk
 9. Ran the kernel and synched devices to make sure we waited for all walks to finish
 10. Printed the final positions of all walks
 
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

#include <time.h> // added
#include <cuda_runtime.h> // added
#include <curand_kernel.h> // added

// Defines
#define NUMBER_OF_WALKS 20 // added

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
int main(int, char**);

// Random walk on GPU
__global__ void randomWalk(int *positionX, int *positionY, unsigned int seed, int steps)
{
	int walkNumber = blockIdx.x * blockDim.x + threadIdx.x; // each thread does a walk

	curandState state;
	curand_init(seed + walkNumber, 0, 0, &state); // each walk has a different seed

	int x = 0; // locations for each thread to start
	int y = 0;

	for(int i = 0; i < steps; i++) // each threads walk takes 10000 steps
	{
		float randomX = curand_uniform(&state);
		float randomY = curand_uniform(&state);

		if(randomX < 0.5f)
		{
			x--;
		}
		else
		{
			x++;
		}

		if(randomY < 0.5f)
		{
			y--;
		}
		else
		{
			y++;
		}
	}

	positionX[walkNumber] = x; // saving final positions
	positionY[walkNumber] = y;
}

// Random walk on CPU
int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

int main(int argc, char** argv)
{
	//srand(time(NULL));
	
	//printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);

	int *positionX; // holding final 20 positions
	int *positionY;

	// allocate unified memory
	cudaMallocManaged(&positionX, NUMBER_OF_WALKS * sizeof(int));
	cudaMallocManaged(&positionY, NUMBER_OF_WALKS * sizeof(int));

	unsigned int seed = (unsigned int)time(NULL);

	// Launch all 20 threads for their walks
	randomWalk<<<1, NUMBER_OF_WALKS>>>(positionX, positionY, seed, NumberOfRandomSteps);
	cudaDeviceSynchronize(); // wait for everyone to finish
	
	// int positionX = 0;
	// int positionY = 0;
	// for(int i = 0; i < NumberOfRandomSteps; i++)
	// {
	//	 positionX += getRandomDirection();
	// 	 positionY += getRandomDirection();
	// }
	
	// printf("\n Final position = (%d,%d) \n", positionX, positionY);

	// print all 20 positions
	for(int i = 0; i < NUMBER_OF_WALKS; i++)
	{
		printf("\nWalk #%d final position: (%d, %d)\n", i+1, positionX[i], positionY[i]);
	}

	// Free memory
	cudaFree(positionX);
	cudaFree(positionY);

	return 0;
}

