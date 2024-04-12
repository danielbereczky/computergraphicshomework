//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bereczky Dániel
// Neptun : WKMTM2
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
int windowSize = 600;
int sideDistance = 40;
//classes

class Camera2D {
	vec2 wCenter = vec2(20.0f, 30.0f);
	vec2 wSize = vec2(150, 150);
public:
	mat4 V() { return TranslateMatrix(-wCenter); }

	mat4 P() //projection mx
	{
		return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y));
	}
	mat4 Vinv() //inverse view mx
	{
		return TranslateMatrix(wCenter);
	}
	mat4 pInv() //inverse projection mx
	{
		return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2));
	}

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};
Camera2D* camera;

class Object {
	unsigned int vbo, vao; // vertices on GPU
public:
	std::vector<vec2> vtx; // vertices on CPU
	Object() {
		//setting up vao
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		//setting up vbo
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		//setting vbo attribs
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void updateGPU() {
		//moving verts from cpu to gpu
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec2), &vtx[0], GL_DYNAMIC_DRAW);
	}
	void Draw(int type, vec3 color) {
		if (vtx.size() > 0) {
			glBindVertexArray(vao);
			gpuProgram.setUniform(color, "color");
			glDrawArrays(type, 0, vtx.size());
		}
	}
};

class Star {
	Object squareGPUpoints;
	std::vector<vec2> pointsCPU;
public:
	Star() {
		//center
		pointsCPU.push_back(vec2(50.0f, 30.0f));
		//top
		pointsCPU.push_back(vec2(50.0f, 30.0f + sideDistance));
		//top right
		pointsCPU.push_back(vec2(50.0f + 40.0f, 30.0f + 40.0f));
		//right
		pointsCPU.push_back(vec2(50.0f + sideDistance, 30.0f));
		//bottom right
		pointsCPU.push_back(vec2(50.0f + 40.0f, 30.0f - 40.0f));
		//bottom
		pointsCPU.push_back(vec2(50.0f, 30.0f - sideDistance));
		//bottom left
		pointsCPU.push_back(vec2(50.0f - 40.0f, 30.0f - 40.0f));
		//left
		pointsCPU.push_back(vec2(50.0f - sideDistance, 30.0f));
		//top left
		pointsCPU.push_back(vec2(50.0f - 40.0f, 30.0f + 40.0f));
		//adding top again to complete the fan
		pointsCPU.push_back(vec2(50.0f, 30.0f + sideDistance));
		

		//pushing to GPU
		for (vec2 v : pointsCPU) {
			squareGPUpoints.vtx.push_back(v);
		}
		squareGPUpoints.updateGPU();
	}
	void Draw() {
		mat4 MVPMat = camera->V() * camera->P();
		gpuProgram.setUniform(MVPMat, "MVP");
		squareGPUpoints.Draw(GL_TRIANGLE_FAN,vec3(1.0f,1.0f,1.0f));
	}
	void adjustS(float adjustment){
		//top
		pointsCPU.at(1).y += adjustment;
		pointsCPU.at(9).y += adjustment;
		//right
		pointsCPU.at(3).x += adjustment;
		//bottom
		pointsCPU.at(5).y -= adjustment;
		//left
		pointsCPU.at(7).x -= adjustment;
		//updating verts on gpu

		squareGPUpoints.vtx.clear();

		for (vec2 v : pointsCPU) {
			squareGPUpoints.vtx.push_back(v);
		}
		squareGPUpoints.updateGPU();
	}
};

Star* star;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowSize, windowSize);

	camera = new Camera2D();
	star = new Star();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	
	star->Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'h':
		star->adjustS(-10.0f);
		glutPostRedisplay();
		break;
	case 'H':
		star->adjustS(10.0f);
		glutPostRedisplay();
		break;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
