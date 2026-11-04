// Name: Brenna Fesmire
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc S_NBodyCPUToGPU1Block_Fixed_Brenna.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To learn how to move an Nbody CPU simulation to an Nbody GPU simulation.
*/

/*
 Explain what you did to fix the code:
 1. Included the cudaErrorCheck() function from previous homeworks
 2. Adapted the allocateMemory() function from previous homeworks so I could set up the GPU memory
 3. Changed the nbody() function to call the kernel and pass the position and velocities back and forth
 4. In main(), added a check to make sure that N was between 0 and 1023 (within one block on the GPU)

 5. Created a kernel to do the nbody calculations on the GPU:
	5a. Set each thread to have its own object
	5b. Compare each object to each other so we can get forces
	5c. Use forces to get velocities and positions
	5d. Save the information for each thread / object

 6. Made the colors random
 
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float GlobeRadius, Diameter, Radius;
float Damp;

// GPU memory
float3 *GPU_P, *GPU_V, *GPU_F;
float *GPU_M;

// added for colors:
float *red, *green, *blue;

// Function prototypes
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
int main(int, char**);
__global__ void nbodyGPU(float3 *, float3 *, float3 *, float *, int, float, float, int); // added
void allocatememory(); // added
void cudaErrorCheck(const char *, int); // added

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glColor3d(red[i], green[i], blue[i]); // added
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
    float randomAngle1, randomAngle2, randomRadius;
    float d, dx, dy, dz;
    int test;
    	
    Damp = 0.5;
    	
    M = (float*)malloc(N*sizeof(float));
    P = (float3*)malloc(N*sizeof(float3));
    V = (float3*)malloc(N*sizeof(float3));
    F = (float3*)malloc(N*sizeof(float3));

	red = (float*)malloc(N*sizeof(float));
	green = (float*)malloc(N*sizeof(float));
	blue = (float*)malloc(N*sizeof(float));
    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, initial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;

		red[i] = 0.35f + 0.5f*(float)rand()/RAND_MAX;
		green[i] = 0.0f;
		blue[i] = 0.7f + 0.5f * (float)rand()/RAND_MAX;
	}

	allocatememory(); // set up the GPU

	printf("\n To start timing type s.\n");
}

void nBody()
{
	int drawCount = 0;
	float time = 0.0f;
	float dt = 0.0001f;

	int firstStep = 1;

	while(time < RUN_TIME)
	{
		// launch the kernel
		nbodyGPU<<<1, N>>>(GPU_P, GPU_V, GPU_F, GPU_M, N, dt, Damp, firstStep);
		cudaErrorCheck(__FILE__, __LINE__);

		// we finished the first step, so we can use it as a reference
		firstStep = 0;

		if(drawCount == DRAW_RATE)
		{
			if(DrawFlag)
			{
				// update positions
				cudaMemcpy(P, GPU_P, N*sizeof(float3), cudaMemcpyDeviceToHost);

				drawPicture();
			}
			drawCount = 0;
		}
		time += dt;
		drawCount++;
	}

	// wait for GPU to finish
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// final positions
	cudaMemcpy(P, GPU_P, N*sizeof(float3), cudaMemcpyDeviceToHost);

	// final velocities
	cudaMemcpy(V, GPU_V, N*sizeof(float3), cudaMemcpyDeviceToHost);
}

// added:
__global__ void nbodyGPU(float3 *P, float3 *V, float3 *F, float *M, int N, float dt, float damp, int firstStep)
{
	int i = threadIdx.x;
	float3 force;
	force.x = 0.0f;
	force.y = 0.0f;
	force.z = 0.0f;

	// get the force acting on each sphere
	for(int j = 0; j < N; j++)
	{
		// don't compare the object to itself
		if(i == j) continue;

		// find the difference in force
		float dx = P[j].x - P[i].x;
		float dy = P[j].y - P[i].y;
		float dz = P[j].z - P[i].z;

		float d2 = dx*dx + dy*dy + dz*dz;
		float d  = sqrt(d2);

		// Lennard-Jones function (it was one dude)
		float force_mag  = (G*M[i]*M[j])/(d2) - (H*M[i]*M[j])/(d2*d2);

		// update the forces
		force.x += force_mag*dx/d;
		force.y += force_mag*dy/d;
		force.z += force_mag*dz/d;
	}

	// save the forces
	F[i] = force;

	// make all the threads wait for each other
	__syncthreads();

	// velocity
	if(firstStep)
	{
		// the initial velocity
		V[i].x += (F[i].x/M[i])*0.5f*dt;
		V[i].y += (F[i].y/M[i])*0.5f*dt;
		V[i].z += (F[i].z/M[i])*0.5f*dt;
	}
	else
	{
		// velocity once the simulation is going
		V[i].x += ((F[i].x - damp*V[i].x)/M[i])*dt;
		V[i].y += ((F[i].y - damp*V[i].y)/M[i])*dt;
		V[i].z += ((F[i].z - damp*V[i].z)/M[i])*dt;
	}

	// position
	P[i].x += V[i].x*dt;
	P[i].y += V[i].y*dt;
	P[i].z += V[i].z*dt;
}

// added:
void allocatememory()
{
	// adapted from previous homeworks
	cudaMalloc(&GPU_P,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&GPU_V,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&GPU_F,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&GPU_M,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMemcpy(GPU_P, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(GPU_V, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(GPU_F, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(GPU_M, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
}

// added:
void cudaErrorCheck(const char *file, int line)
{
	// added from previous homeworks
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}

	if(N <= 0 || N >=1024) // make sure the threads are workable
	{
		printf("\n\nNumber of bodies must be between 1 and 1023!\n\n");
		exit(1);
	}
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}





