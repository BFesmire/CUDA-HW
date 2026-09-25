// Name: Brenna Fesmire
// Not simple Julia Set on the GPU
// nvcc Fractal_AMD.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size. 
   I will pick these on the fly unsigned int WindowWidth, WindowHeight; 
   float XMin, XMax, YMin, YMax; and your code should work. You will be graded on this.
   
 - But you can set these values to whatever you want for the art compitition.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
 - Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.
 - This is a competition with a prize!!!
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

/*
 Explain what you did to fix the code:

 1. passed WindowWidth and WindowHeight into several functions so the code works for any dimensions
 2. changed the A and B to make a different fractal
 3. changed the escapeOrNotColor() function to change the color based on when something escaped
 4. added mouse() function (when you left click it zooms in and when you right click it zooms out)

 I'm sure there's other things I changed, but those are the big ones.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <stdlib.h> // added
#include <math.h> // added

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume you have escaped
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape
#define A  -0.044 // Real part of C
#define B  0.8 // Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Added:
bool leftDown = false; // neither mouse button is pressed
bool rightDown = false;
int lastPixelX = 0;
int lastPixelY = 0;

float centerX = 0.0f;
float centerY = 0.0f;
float viewHalfHeight = 2.0f;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor(float, float, float*, float*);
__global__ void colorPixels(float*, float, float, float, float, unsigned int, unsigned int);
void display(void);
void mouse(int, int, int, int);

// Added:
void updateBounds();
void reshape(int, int);
void keyboard(unsigned char, int, int);
void timer(int);

// Error check
void cudaErrorCheck(const char *file, int line)
{
    cudaError_t error;

    error = cudaGetLastError();

    if(error != cudaSuccess)
    {
        printf("\nCUDA ERROR: message = %s, File = %s, Line = %d\n",
               cudaGetErrorString(error), file, line);
        exit(0);
    }
}

__device__ float escapeOrNotColor(float x, float y, float *green, float *blue)
{
    float mag, tempX;
    int count;
    int maxCount = MAXITERATIONS;
    float maxMag = MAXMAG;

	count = 0;
    mag = sqrt(x*x + y*y);
    while(mag < maxMag && count < maxCount)
    {
        tempX = x; // We will be changing the x but we need its old value to find y
        x = x*x - y*y + A;
        y = (2.0 * tempX * y) + B;
        mag = sqrt(x*x + y*y);
        count++;
    }

    // Point did not escape -- inside the Julia set.
    if(count >= maxCount)
    {
        *green = 0.0;
        *blue = 0.0;

        return 0.00; // red
    }

    // keeps the outside black
    if(count < 20)
    {
        *green = 0.0;
        *blue = 0.0;

        return 0.00; // red
    }

    float t = (float)count / (float)maxCount; // how many iterations

    // making the colors rainbow
    float red = 0.5 + 0.5 * sin(6.28318 * t);
    *green = 0.5 + 0.5 * sin(6.28318 * t + 2.09);
    *blue = 0.5 + 0.5 * sin(6.28318 * t + 4.19);

    return red;
}

// CUDA Kernel
__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, unsigned int width, unsigned int height)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalPixels = width * height;

    // Ignore threads outside the window
    if(k >= totalPixels)
    {
        return;
    }

    unsigned int pixelX = k % width;
    unsigned int pixelY = k / width;

    // Convert threads / pixels into fractal colors
    float x = xMin + dx * pixelX;
    float y = yMin + dy * pixelY;

    // Three values per pixel: R, G, B.
    unsigned int id = 3 * k;

    float green = 0.0;
    float blue = 0.0;
    float red = escapeOrNotColor(x, y, &green, &blue); // also changes green and blue

    pixels[id]     = red;
    pixels[id + 1] = green;
    pixels[id + 2] = blue;
}

void display(void)
{
    dim3 blockSize, gridSize;
    float *pixelsCPU, *pixelsGPU;
    float stepSizeX, stepSizeY;

    // We need the 3 becayse each pixel has a red, green, and blue value
    pixelsCPU = (float *)malloc(WindowWidth * WindowHeight * 3 * sizeof(float));
    cudaMalloc(&pixelsGPU,WindowWidth * WindowHeight * 3 * sizeof(float));

    cudaErrorCheck(__FILE__, __LINE__);

    // Distance between adjacent pixels in the fractal.
    stepSizeX =(XMax - XMin) / (float)WindowWidth;
    stepSizeY =(YMax - YMin) / (float)WindowHeight;

    blockSize.x = 256; // threads per block
    blockSize.y = 1;
    blockSize.z = 1;

    unsigned int totalPixels = WindowWidth * WindowHeight;

    gridSize.x = (totalPixels + blockSize.x - 1) / blockSize.x;
    gridSize.y = 1;
    gridSize.z = 1;

    // Run the CUDA kernel.
    colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
    cudaErrorCheck(__FILE__, __LINE__);

    // Copy pixels from GPU back to CPU.
    cudaMemcpy(pixelsCPU, pixelsGPU, WindowWidth * WindowHeight * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);


    // Draw pixels on the screen.
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU);
    glFlush();

    // Free memory.
    cudaFree(pixelsGPU);
    free(pixelsCPU);
}

// ------------------------------------------------------------------------------

void updateBounds()
{
    float aspect = (float)WindowWidth / (float)WindowHeight;
    float halfHeight = viewHalfHeight;
    float halfWidth  = viewHalfHeight * aspect;

    XMin = centerX - halfWidth;
    XMax = centerX + halfWidth;
    YMin = centerY - halfHeight;
    YMax = centerY + halfHeight;
}

void mouse(int button, int state, int pixelX, int pixelY) // added
{
    if(button == GLUT_LEFT_BUTTON)
    {
        leftDown = (state == GLUT_DOWN);
    }
    if(button == GLUT_RIGHT_BUTTON)
    {
        rightDown = (state == GLUT_DOWN);
    }

    lastPixelX = pixelX;
    lastPixelY = pixelY;
}

void timer(int val)
{
     if(leftDown || rightDown)
    {
        float stepSizeX = (XMax - XMin) / (float)WindowWidth;
        float stepSizeY = (YMax - YMin) / (float)WindowHeight;

        float clickedX = XMin + lastPixelX * stepSizeX;
        float clickedY = YMin + (WindowHeight - lastPixelY) * stepSizeY;

        float zoomFactor = leftDown ? 0.995f : (1.0f / 0.995f); // zoom in : zoom out // < 1 is in, > 1 is out

        // Move center toward/away from the clicked point, scaled by the zoom,
        // instead of snapping center directly to the clicked point.
        centerX = clickedX + zoomFactor * (centerX - clickedX);
        centerY = clickedY + zoomFactor * (centerY - clickedY);
        viewHalfHeight *= zoomFactor;

        updateBounds();
        glutPostRedisplay();
    }
    glutTimerFunc(16, timer, 0);
}


void keyboard(unsigned char key, int x, int y)
{
     if(key == 'r' || key == 'R')
    {
        centerX = 0.0f;
        centerY = 0.0f;
        viewHalfHeight = 2.0f;

        updateBounds();
        glutPostRedisplay();
    }
}

void reshape(int w, int h)
{
    WindowWidth = w;
    WindowHeight = h;
    updateBounds();
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowSize(WindowWidth, WindowHeight);

    glutCreateWindow("Fractals--Man--Fractals");

    glutDisplayFunc(display);

    glutMouseFunc(mouse); // added

    glutTimerFunc(16, timer, 0); // added

    glutKeyboardFunc(keyboard); // added

    glutReshapeFunc(reshape);

    updateBounds();

    glutMainLoop();
}
