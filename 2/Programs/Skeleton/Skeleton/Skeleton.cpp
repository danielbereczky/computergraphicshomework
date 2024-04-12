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

/*DISCLAIMER:
A KOD IRASA SORAN par reszlet azaz MVP matrix kiszamitasa, szorzasa, Catmull-rom gorbeben a sebesseg vektorok szamitasa keszitese soran az alabbi videot hasznaltam segitsegkent:
https://www.youtube.com/watch?v=XPuOS39qadE
*/
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


//GLOBAL VARS

int winSize = 600;
GPUProgram gpuProgram; // vertex and fragment shaders
enum operation {lagrange,bezier,catmullRom};
operation programMode;
int grabbedPtIdx;
float catmullRomTension = 0.0f;

void calculateKnotValues(std::vector<float> knots, int cPoints){
	knots.clear();
	for (int i = 0; i <= cPoints;i++){
		//linearly mapping knot values from 0 to 1
		knots.push_back((1 / cPoints ) * i);
	}
}
float calculatePointDistance(vec2 p1, vec2 p2){
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

class Camera2D {
	vec2 wCenter = vec2(0.0f, 0.0f);
	vec2 wSize = vec2(30, 30);
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
//CLASSES

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
//abstract base class for curves
class Curve{
protected:
	Object controlPoints;
	Object interPoints;
	std::vector<vec2> interPointsCPU;
	std::vector<vec2> controlPointsCPU;
public:
	virtual ~Curve() = default;
	virtual void addControlPoint(vec2 np,float t = 0) = 0;
	void movePoint(int idx,vec2 moveToP){
		vec4 newVert = vec4(moveToP.x, moveToP.y, 0, 1) * camera->pInv() * camera->Vinv();
		vec2 newVert2D = vec2(newVert.x, newVert.y);
		//avoids out of bounds
		if (idx == -1) return;
		controlPointsCPU.at(idx).x = newVert2D.x;
		controlPointsCPU.at(idx).y = newVert2D.y;
		controlPoints.vtx.clear();
		for (size_t i = 0; i < controlPointsCPU.size();i++){
			controlPoints.vtx.push_back(vec2(controlPointsCPU.at(i).x, controlPointsCPU.at(i).y));
		}
		controlPoints.updateGPU();
		return;

	}
	void Draw(){
		mat4 MVPMat = camera->V() * camera->P();
		gpuProgram.setUniform(MVPMat,"MVP");
		if (controlPointsCPU.size() >= 2) {
			interPoints.Draw(GL_LINE_STRIP, vec3(1.0f, 1.0f, 0.0f));
		}
		if (controlPointsCPU.size() > 0) {
			controlPoints.Draw(GL_POINTS, vec3(1.0f, 0.0f, 0.0f));
		}
	}
	//if there is a control point near 'inp', return its index. otherwise return -1
	int getNearbyPoint(vec2 inp){
		vec4 inpT = vec4(inp.x, inp.y, 0, 1) * camera->pInv() * camera->Vinv();
		vec2 inpT2D = vec2(inpT.x, inpT.y);
		float threshold = 0.1;
		for (size_t i = 0; i < controlPointsCPU.size();i++) {
			if (calculatePointDistance(controlPointsCPU.at(i), inpT2D) < threshold) return i;
		}
		return -1;
	}
	virtual void calculateInterPoints() = 0;
};

class LagrangeCurve : public Curve {
	std::vector<float> knots;
	float L(int i, float t) {
		float Li = 1.0f;
		for (size_t j = 0; j < controlPointsCPU.size();j++) {
			if ((int)j != i) Li *= (t - knots[j]) / (knots[i] - knots[j]);
		}
		return Li;
	}
	vec2 r(float t) {
		vec2 rt(0.0f, 0.0f);
		for (size_t i = 0; i < controlPointsCPU.size();i++) rt = rt + (controlPointsCPU.at(i) * L(i, t));
		return rt;

	}
	void calculateKnotVals() {
		knots.clear();
		float totalDist = 0.0f;
		//calculating total distance between points
		for (size_t i = 0; i < controlPointsCPU.size() - 1; ++i) {
			totalDist += calculatePointDistance(controlPointsCPU.at(i + 1), controlPointsCPU.at(i));
		}
		for (size_t i = 0; i < controlPointsCPU.size();i++) {
			float distFromStart = 0.0f;
			// the first CP gets a knot value of 0
			if (i == 0) knots.push_back(0.0f);
			//else the knot value will be (distance to current CP/total length)
			else {
				for (size_t j = 0; j < i; j++) {
					distFromStart += calculatePointDistance(controlPointsCPU.at(j + 1), controlPointsCPU.at(j));
				}
				knots.push_back(distFromStart / totalDist);
			}
		}
	}
	void calculateInterPoints() {
		interPoints.vtx.clear();
		interPointsCPU.clear();
		for (int i = 0;i <= 100;i++) {
			float f = (float)i / 100;
			interPointsCPU.push_back(r(f));
			interPoints.vtx.push_back(r(f));
		}
		interPoints.updateGPU();
	}
	virtual void addControlPoint(vec2 np,float t = 0){
		vec4 newVert = vec4(np.x, np.y, 0, 1) * camera->pInv() * camera->Vinv();
		controlPointsCPU.push_back(vec2(newVert.x,newVert.y));
		controlPoints.vtx.push_back(vec2(newVert.x, newVert.y));
		controlPoints.updateGPU();
		//maybe???
		//knots.push_back((float)controlPointsCPU.size());
		this->calculateKnotVals();
		this->calculateInterPoints();
	}
};


class BezierCurve : public Curve {
	float B(int i, float t) {
		int n = controlPointsCPU.size() - 1;
		float coef = 1;
		for (size_t j = 1; (int)j <= i;j++) {
			coef *= (float) (n - j + 1) / j;
		}
		return coef * pow(t, i) * pow(1 - t, n - i);
	}
	vec2 r(float t) {
		vec2 rt(0.0f, 0.0f);
		for (size_t i = 0; i < controlPointsCPU.size(); i++) {
			rt = rt + controlPointsCPU.at(i) * B(i, t);
		}
		return rt;
	}
	void calculateInterPoints(){
		interPoints.vtx.clear();
		interPointsCPU.clear();
		for (int i = 0;i <= 100;i++){
			float f = (float)i / 100;
			interPointsCPU.push_back(r(f));
			interPoints.vtx.push_back(r(f));
		}
		interPoints.updateGPU();
	}
	virtual void addControlPoint(vec2 np,float t = 0) {
		vec4 newVert = vec4(np.x, np.y, 0, 1) * camera->pInv() * camera->Vinv();
		controlPointsCPU.push_back(vec2(newVert.x, newVert.y));
		controlPoints.vtx.push_back(vec2(newVert.x, newVert.y));
		controlPoints.updateGPU();
		controlPoints.updateGPU();
		this->calculateInterPoints();
	};
};

class CatmullRomSpline : public Curve {
	std::vector<float> knots;
public:
	vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		float dt = t1 - t0;
		t -= t0;
		float dttopow_2 = dt * dt;
		float dttopow_3  = dttopow_2* dt;

		vec2 a0 = p0, a1 = v0;
		vec2 a2 = (p1 - p0) * 3 / dttopow_2 - (v1 + v0 * 2) / dt;
		vec2 a3 = (p0 - p1) * 2 / dttopow_3 + (v1 + v0) / dttopow_2;

		return ((a3 * t + a2) * t + a1) * t + a0;
	}

	vec2 r(float t) {
		vec2 v0, v1;
		vec2 vP, vC, vN;
		for (size_t i = 0; i < controlPointsCPU.size() - 1; i++) {
			if (knots.at(i) <= t && t <= knots.at(i + 1)) {
				//velocity vector of the current node.
				vec2 vC = (controlPointsCPU[i + 1] - controlPointsCPU[i]) / (knots[i + 1] - knots[i]);
				//if the parameter would be found between the starting cp and the one adjecent to it, we set the velocity vector to 0,0.
				if (i > 0){
					vP = (controlPointsCPU[i] - controlPointsCPU[i - 1]) / (knots[i] - knots[i - 1]);
				}
				else {
					vP = vec2(0.0f, 0.0f);
				}
				//same thing. if the parameter is found between the last cp and the one before it, we set the velocity vector to 0,0
				if (i < controlPointsCPU.size() - 2){
					vN = (controlPointsCPU[i + 2] - controlPointsCPU[i + 1]) / (knots[i + 2] - knots[i + 1]);
				}
				else {
					vN = vec2(0.0f, 0.0f);
				}
				v0 = (vP + vC) * (1 - catmullRomTension) * 0.5f;
				v1 = (vC + vN) * (1 - catmullRomTension) * 0.5f;
				// Hermite interpolation
				return Hermite(controlPointsCPU.at(i), v0, knots.at(i), controlPointsCPU.at(i + 1), v1, knots.at(i + 1), t);
			}
		}
		return controlPointsCPU.at(0);
	}
	void calculateKnotVals(){
		knots.clear();
		float totalDist = 0.0f;
		//calculating total distance between points
		for (size_t i = 0; i < controlPointsCPU.size() - 1; ++i) {
			totalDist += calculatePointDistance(controlPointsCPU.at(i + 1), controlPointsCPU.at(i));
		}
		for (size_t i = 0; i < controlPointsCPU.size();i++) {
			float distFromStart = 0.0f;
			// the first CP gets a knot value of 0
			if (i ==  0) knots.push_back(0.0f);
			//else the knot value will be (distance to current CP/total length)
			else {
				for (size_t j = 0; j < i; j++){
					distFromStart += calculatePointDistance(controlPointsCPU.at(j + 1), controlPointsCPU.at(j));
				}
				knots.push_back(distFromStart / totalDist);
			}
		}
	}

	virtual void addControlPoint(vec2 np, float t) {
		// Add the control point
		vec4 newVert = vec4(np.x, np.y, 0, 1) * camera->pInv() * camera->Vinv();
		controlPointsCPU.push_back(vec2(newVert.x, newVert.y));
		controlPoints.vtx.push_back(vec2(newVert.x, newVert.y));
		controlPoints.updateGPU();
		this->calculateKnotVals();
		this->calculateInterPoints();
	}

	void calculateInterPoints() {
		interPoints.vtx.clear();
		interPointsCPU.clear();
		for (int i = 0;i <= 100;i++) {
			float f = (float)i / 100;
			float par = (knots.back() - knots.front()) * f + knots.front();
			interPoints.vtx.push_back(r(par));
		}
		interPoints.updateGPU();
	}
};


Curve* curCurve;

// Initialization, create an OpenGL context
void onInitialization() {
	glPointSize(10);	//point and line width
	glLineWidth(2);
	glViewport(0, 0, winSize, winSize);	//window 

	camera = new Camera2D();
	//By default, the program starts in Bezier curve mode
	curCurve = new BezierCurve();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	//clear BG with black
	glClearColor(0, 0, 0, 0);    
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	//draw curve
	curCurve->Draw();
	
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	//changing curve modes
	case('l'):
		//programMode = lagrange;
		delete curCurve;
		curCurve = new LagrangeCurve();
		break;
	case('b'):
		//programMode = bezier;
		delete curCurve;
		curCurve = new BezierCurve();
		break;
	case('c'):
		//programMode = catmullRom;
		delete curCurve;
		curCurve = new CatmullRomSpline();
		break;
	//camera manipulation
	case('z'):
		camera->Zoom(1/1.1f);
		break;
	case('Z'):
		camera->Zoom(1.1f);
		break;
	case('P'):
		camera->Pan(vec2(1.0f, 0.0f));
		break;
	case('p'):
		camera->Pan(vec2(-1.0f, 0.0f));
		break;
	case('t'):
		catmullRomTension -= 0.1f;
		curCurve->calculateInterPoints();
		break;
	case('T'):
		catmullRomTension += 0.1f;
		curCurve->calculateInterPoints();
		break;
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / winSize - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / winSize;
	
	curCurve->movePoint(grabbedPtIdx, vec2(cX, cY));
	curCurve->calculateInterPoints();
	glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / winSize - 1;	
	float cY = 1.0f - 2.0f * pY / winSize; // flip y axis

	switch (button) {
	case GLUT_LEFT_BUTTON:
		switch (state){
		case GLUT_DOWN:
			curCurve->addControlPoint(vec2(cX,cY));
			break;
		case GLUT_UP:
			break;
		}
		break;
	case GLUT_RIGHT_BUTTON:
		switch (state){
		case GLUT_DOWN:
			grabbedPtIdx = curCurve->getNearbyPoint(vec2(cX, cY));
			break;
		case GLUT_UP:
			grabbedPtIdx = -1;
			break;
		}
	}
	glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
