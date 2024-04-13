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
	layout(location = 1) in vec2 vUV;
	out vec2 txCoord;

	void main() {
		txCoord = vUV;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers
	out vec4 outColor;		// computed color of the current pixel
	uniform sampler2D textureUnit;
	
	in vec2 txCoord;

	void main() {
		outColor = texture(textureUnit, txCoord); 
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
int windowSize = 600;
float sideDistance = 40.0f;
int texSize = 300;
int texMode = GL_LINEAR;
//classes

class Camera2D {
	vec2 wCenter = vec2(20.0f, 30.0f);
	vec2 wSize = vec2(150.0f, 150.0f);
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

struct Circle {
	Circle(vec2 c, float r) : center(c), R(r) {};
	vec2 center;
	float R;
	bool In(vec2 r) { return(dot(r - center, r - center) - R * R < 0); }

};

vec2 projectFromHyperbolic(vec3 inP) {
	// z != -1
	return vec2(inP.x / (inP.z + 1), inP.y / (inP.z + 1));
}

std::vector<vec4> createCustomTexture(int size) {
	//output
	std::vector<vec4> image(size * size);

	std::vector<Circle> circles;

	//step 1: calculating the points

	for (int i = 0; i < 360; i += 40) {
		float phi = i;
		float phiRad = phi * 3.14159f / 180.0f;
		//moving along the line
		for (int j = 0; j < 6; j++) {
			vec3 tempP = vec3(cos(phiRad) * sinh(0.5f + j), sin(phiRad) * sinh(0.5f + j), cosh(0.5f + j));

			//step 2: projecting to euclidian space
			vec2 projectedP = projectFromHyperbolic(tempP);

			//step 3: calculating circles
			//calculating the centre of the inverse circle. Radius = 1 (unit circle)
			// |OP'| = r^2 / |OP|   (Point = |OP|)

			float distFromOrigin = 1.0f / length(projectedP);

			vec2 pInverse = normalize(projectedP) * distFromOrigin;

			vec2 newC = vec2((pInverse.x + projectedP.x) / 2.0f, (pInverse.y + projectedP.y) / 2.0f);

			circles.push_back(Circle(newC, length(newC - pInverse)));
		}
	}

	Circle unitC = Circle(vec2(0.0f, 0.0f), 1);
	//step 4: calculate pixel colors
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			//normalizing , OK
			float normalizedX = -1.0f + 2.0f * x / (size - 1);
			float normalizedY = 1.0f - 2.0f * y / (size - 1);

			int pixelInCircles = 0;
			for each (Circle c in circles) {
				if (c.In(vec2(normalizedX, normalizedY))) {
					pixelInCircles++;
				}
			}
			//coloring the pixel, OK
			if (!unitC.In(vec2(normalizedX, normalizedY))) {
				image[y * size + x] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			}
			else {
				if (pixelInCircles % 2 == 1) {
					image[y * size + x] = vec4(0.0f, 0.0f, 1.0f, 1);
				}
				else {
					image[y * size + x] = vec4(1.0f, 1.0f, 0.0f, 1);
				}
			}
		}
	}
	return image;
}



class Object {
	//vbo[0] : vertices
	//vbo[1] : uvs
	unsigned int vbo[2], vao; // vertices on GPU
public:
	std::vector<vec2> vtx; // vertices on CPU
	std::vector<vec2> uv; //uv coordinates on CPU
	Texture* tex;
	Object(int w, int h, const std::vector<vec4> image){
		tex = new Texture(w, h, image);
		//setting up vao
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		//generating vertex buffers
		glGenBuffers(2, vbo);

		//setting up vbo 0
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		//setting vbo attribs
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		//setting up vbo 1
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		//setting vbo attribs
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void updateGPU() {
		//moving verts from cpu to gpu
		glBindVertexArray(vao);
		//vertices
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec2), &vtx[0], GL_DYNAMIC_DRAW);
		//uvs
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(vec2), &uv[0], GL_STATIC_DRAW);
	}
	void Draw(int type, vec3 color) {
		if (vtx.size() > 0) {
			mat4 MVPMat = camera->V() * camera->P();
			gpuProgram.setUniform(MVPMat, "MVP");
			gpuProgram.setUniform(*tex, "textureUnit");
			glBindVertexArray(vao);
			glDrawArrays(type, 0, vtx.size());
		}
	}
	void createnewTexture(int size) {
		delete tex;
		tex = new Texture(size, size, createCustomTexture(size),texMode);
	}
};



class Star {
	Object squareGPUpoints;
	std::vector<vec2> pointsCPU;
	//std::vector<vec2> uvs;
	vec2 uvs[10];
	vec2 translation = vec2(50.0f, 30.0f);
	float sideLen = 40.0f;
	float phi;
public:
	Star(int w, int h, const std::vector<vec4> image) :squareGPUpoints(w, h, image) {
		phi = 0.0f;
		//center
		pointsCPU.push_back(translation);
		uvs[0] = (vec2(0.5f, 0.5f));
		//top
		pointsCPU.push_back(vec2(0, sideDistance) + translation);
		uvs[1] = (vec2(0.5f, 1.0f));
		//top right
		pointsCPU.push_back(vec2(sideLen, sideLen) + translation);
		uvs[2] = (vec2(1.0f, 1.0f));
		//right
		pointsCPU.push_back(vec2(sideDistance, 0) + translation);
		uvs[3] = (vec2(1.0f, 0.5f));
		//bottom right
		pointsCPU.push_back(vec2(sideLen, -sideLen) + translation);
		uvs[4] = (vec2(1.0f, 0.0f));
		//bottom
		pointsCPU.push_back(vec2(0, -sideDistance) + translation);
		uvs[5] = (vec2(0.5f, 0.0f));
		//bottom left
		pointsCPU.push_back(vec2(-sideLen, -sideLen) + translation);
		uvs[6] = (vec2(0.0f, 0.0f));
		//left
		pointsCPU.push_back(vec2(-sideDistance, 0) + translation);
		uvs[7] = (vec2(0.0f, 0.5f));
		//top left
		pointsCPU.push_back(vec2(-sideLen, sideLen) + translation);
		uvs[8] = (vec2(0.0f, 1.0f));
		//adding top again to complete the fan
		pointsCPU.push_back(vec2(0, sideDistance) + translation);
		uvs[9] = (vec2(0.5f, 1.0f));

		//pushing to GPU
		for (vec2 v : pointsCPU) {
			squareGPUpoints.vtx.push_back(v);
		}

		for (vec2 u : uvs) {
			squareGPUpoints.uv.push_back(u);
		}
		squareGPUpoints.updateGPU();
	}
	void Draw() {
		squareGPUpoints.Draw(GL_TRIANGLE_FAN, vec3(1.0f, 1.0f, 1.0f));
	}

	void adjustS(float adjustment) {
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
	void Animate(float r) {
		translation = vec2(0.0f, 0.0f);
		phi = r;
	}
	mat4 starM() {
		mat4 rotation(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		mat4 translation(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			translation.x, translation.y, 0, 1);

		return rotation * translation;
	}
	void adjustTexSize(int siz){
		squareGPUpoints.createnewTexture(siz);
	}
};

Star* star;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowSize, windowSize);

	camera = new Camera2D();

	//star = new Star(width,height,image);
	star = new Star(texSize,texSize,createCustomTexture(texSize));


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
	case 'r':
		texSize -= 100;
		if (texSize <= 0) { texSize = 100; }
		star->adjustTexSize(texSize);
		break;
	case 'R':
		texSize += 100;
		star->adjustTexSize(texSize);
		break;
	case 't':
		texMode = GL_NEAREST;
		star->adjustTexSize(texSize);
		break;
	case 'T':
		texMode = GL_LINEAR;
		star->adjustTexSize(texSize);
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
	float elapsedSec = time / 1000.0f;
	//star->Animate(elapsedSec);
	glutPostRedisplay();
}
