// Name: Brenna Fesmire
// Two body problem
// nvcc R_TwoBodyToNBodyCPU_Fixed_Brenna.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

/*
 Explain what you did to fix the code:
 1. Added NUMBER_OF_SPHERES so we can work with any number of spheres
 2. Created array fo px, py, pz, vx, vy, vz, fx, fy, fz, and mass so they can be defined for NUMBER_OF_SPHERES objects
 3. Added a nested loop to check the initial conditions of one object against all previous objects
 4. Added a loop to set the velocity and mass for each object
 5. Made a loop in keep_in_box() so it works for NUMBER_OF_SPHERES objects
 6. Added a nested loop in get_forces() so each object checks with other objects
 7. Reset the forces each time get_forces() runs so that forces are only accumulated each time step
 8. Added a loop in move_bodies() to work with NUMBER_OF_SPHERES objects
 9. Added a loop in draw_picture() to work with NUMBER_OF_SPHERES objects
 10. Added arrays for red, green, and blue so each sphere gets a random color

*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

#define NUMBER_OF_SPHERES 10 // added

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

float px[NUMBER_OF_SPHERES];
float py[NUMBER_OF_SPHERES];
float pz[NUMBER_OF_SPHERES];

float vx[NUMBER_OF_SPHERES];
float vy[NUMBER_OF_SPHERES];
float vz[NUMBER_OF_SPHERES];

float fx[NUMBER_OF_SPHERES];
float fy[NUMBER_OF_SPHERES];
float fz[NUMBER_OF_SPHERES];

float mass[NUMBER_OF_SPHERES];

float red[NUMBER_OF_SPHERES];
float green[NUMBER_OF_SPHERES];
float blue[NUMBER_OF_SPHERES];


// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy;
	float dx, dy, dz, seperation;
	
	// initialize the position of the first sphere so we can compare the others
	px[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	py[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	pz[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	yeahBuddy = 0;
	while(yeahBuddy == 0)
	{

		for(int i = 1; i < NUMBER_OF_SPHERES; i++)
		{
			px[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			py[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			pz[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;

			for(int j = 0; j < i; j++)
			{
				dx = px[i]-px[j];
				dy = py[i]-py[j];
				dz = pz[i]-pz[j];

				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				yeahBuddy = 1;
				if(seperation < DIAMETER) yeahBuddy = 0;
			}
		}
	}

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;

		mass[i] = 1.0;

		red[i] = (float)rand()/RAND_MAX;
		green[i] = (float)rand()/RAND_MAX;
		blue[i] = (float)rand()/RAND_MAX;
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		glColor3d(red[i], green[i], blue[i]);
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radius, 20, 20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(px[i] > halfBoxLength)
		{
			px[i] = 2.0*halfBoxLength - px[i];
			vx[i] = -vx[i];
		}
		else if(px[i] < -halfBoxLength)
		{
			px[i] = -2.0*halfBoxLength - px[i];
			vx[i] = -vx[i];
		}

		if(py[i] > halfBoxLength)
		{
			py[i] = 2.0*halfBoxLength - py[i];
			vy[i] = -vy[i];
		}
		else if(py[i] < -halfBoxLength)
		{
			py[i] = -2.0*halfBoxLength - py[i];
			vy[i] = -vy[i];
		}
		
		if(pz[i] > halfBoxLength)
		{
			pz[i] = 2.0*halfBoxLength - pz[i];
			vz[i] = -vz[i];
		}
		else if(pz[i] < -halfBoxLength)
		{
			pz[i] = -2.0*halfBoxLength - pz[i];
			vz[i] = -vz[i];
		}		
	}
}

void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;

	// reset forces so they are only accumulated each time step
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for(int j = i+1; j < NUMBER_OF_SPHERES; j++)
		{
			dx = px[j] - px[i];
			dy = py[j] - py[i];
			dz = pz[j] - pz[i];

			r2 = dx*dx + dy*dy + dz*dz;
			r = sqrt(r2);

			forceMag =  mass[i] * mass[j] * GRAVITY / r2;

			if(r < DIAMETER)
			{
				dvx = vx[j] - vx[i];
				dvy = vy[j] - vy[i];
				dvz = vz[j] - vz[i];

				inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}

			// using += so the sphere gets the total forces of all other spheres
			fx[i] += forceMag * dx/r;
			fy[i] += forceMag * dy/r;
			fz[i] += forceMag * dz/r;

			fx[j] += -forceMag * dx/r;
			fy[j] += -forceMag * dy/r;
			fz[j] += -forceMag * dz/r;


		}
	}
}

void move_bodies(float time)
{

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(time == 0.0)
		{
			vx[i] += 0.5*DT*(fx[i] - DAMP*vx[i])/mass[i];
			vy[i] += 0.5*DT*(fy[i] - DAMP*vy[i])/mass[i];
			vz[i] += 0.5*DT*(fz[i] - DAMP*vz[i])/mass[i];
		}
		else
		{
			vx[i] += DT*(fx[i] - DAMP*vx[i])/mass[i];
			vy[i] += DT*(fy[i] - DAMP*vy[i])/mass[i];
			vz[i] += DT*(fz[i] - DAMP*vz[i])/mass[i];
		}

		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	}
	
	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}


