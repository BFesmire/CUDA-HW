// Name: Brenna Fesmire
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU_Fixed_Brenna.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
 Use __device__ for the escapeOrNotColor function
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

/*
 Explain what you did to fix the code:

 Added __device__ before escapeOrNotColor() function
 Added BlockSize, GridSize, *pixels_GPU, *pixels_CPU under Global variables section
 Added setUpDevices(), allocateMemory(), cleanUp() functions
 Added __global__ void createFractal(float *, unsigned int, unsigned int, float, float, float, float) function
 Removed the loops from display() function
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

dim3 BlockSize; // holds dimensions of the block
dim3 GridSize; // holds dimensions of the grid

float *pixels_GPU; // will be 1024*1024 (also, *3 since each has three colors/dimensions)
float *pixels_CPU;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor(float, float); // added __device__
void setUpDevices(); // added
__global__ void createFractal(float *, unsigned int, unsigned int, float, float, float, float); // added
void allocateMemory(); // added
void cleanUp(); // added

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

__device__ float escapeOrNotColor (float x, float y) // added __device__
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0); // did escape, color black
	}
	else
	{
		return(1.0); // did not escape, color red
	}
}

void setUpDevices() // added
{
	BlockSize.x = 1024; // threads
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1024; // blocks
	GridSize.y = 1;
	GridSize.z = 1;
}

void allocateMemory() // added
{
	//We need the 3 because each pixel has a red, green, and blue value.
	cudaMalloc(&pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float)); // allocating memory for threads
	pixels_CPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
}

__global__ void createFractal(float *pixels, unsigned int width, unsigned int height,
								float xMin, float xMax, float yMin, float yMax) // Finds the threads position / the pixel position
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	int xPixel = k % width; // splitting the pixels into x and y coordinates
	int yPixel = k / width;

	float stepSizeX = (xMax - xMin) / (float)width; // finding the steps for x and y
	float stepSizeY = (yMax - yMin) / (float)height;

	float x = xMin + xPixel * stepSizeX; // finding the current x and y coordinates for any pixel
	float y = yMin + yPixel * stepSizeY;

	int pixelIndex = k*3; // each pixel needs 3 dimensions for the colors

	pixels[pixelIndex] = escapeOrNotColor(x,y)*(153.0/255.0);	//Red on or off
	pixels[pixelIndex+1] = escapeOrNotColor(x,y)*(51.0/255.0);; 	//Green on or off
	pixels[pixelIndex+2] = escapeOrNotColor(x,y);	//Blue on or off
}

void cleanUp() // GPU clean up
{
	cudaFree(pixels_GPU);
	free(pixels_CPU);
}

void display(void)
{ 	
	createFractal<<<GridSize, BlockSize>>>(pixels_GPU, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax); // Launches the kernel

	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(pixels_CPU, pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels_CPU); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 
	setUpDevices(); // added
	allocateMemory(); // added

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();

	cleanUp(); // added
}