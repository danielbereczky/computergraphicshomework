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

int winSize = 600;

GPUProgram gpuProgram; // vertex and fragment shaders

//enum for storing current operation
enum operation { addPoint, addLine, moveLine, intersectLines };

operation programMode;

std::vector<vec3> lineCreationTemp;

std::string getProgramStatus(){
	switch(programMode){
	case addPoint:
		return "Add point";
	case addLine:
		return "Add line";
	case moveLine:
		return "Move";
	case intersectLines:
		return "Intersect";
	default:
		return "invalid";
	}
}

//classes
class Object {
	unsigned int vao, vbo; // GPU
	std::vector<vec3> vtx; // CPU
public:
	Object() {
		glGenVertexArrays(1, &vao); glBindVertexArray(vao);
		glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	std::vector<vec3>& Vtx() { return vtx; }
	void updateGPU() { // CPU -> GPU
		glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec3), &vtx[0], GL_DYNAMIC_DRAW);
	}
	void Draw(int type, vec3 color) {
		if (vtx.size() > 0) {
			glBindVertexArray(vao);
			gpuProgram.setUniform(color, "color");
			glDrawArrays(type, 0, vtx.size());
		}
	}
};

class PointCollection {
	Object points;
	
public:
	std::vector<vec3> getVtx(){
		return points.Vtx();
	}
	void addPoint(vec3 np) {
		points.Vtx().push_back(np);
		points.updateGPU();
	}
	boolean isNear(vec3 p1, vec3 p2) {
		return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)) < 0.01 ? true : false;
	}
	void Draw() {
		points.Draw(GL_POINTS, vec3(1.0f, 0.0f, 0.0f));
	}
};

class Line {
	vec3 p1, p2;
	vec3 dirVec;
public:
	vec3 getp1() { return p1; }
	vec3 getp2() { return p2; }
	Line(vec3 np1, vec3 np2) :p1(np1), p2(np2) {
		printf("Line added\n");
		// line going through point p with normal vector n
		// nx*px + ny*py + -(n)*p = 0 
		//
		// direction vector:
		dirVec = vec3(np2.x - np1.x, np2.y - np1.y, 1);
		//creating normal vector : (x,y) -> (-y,x)
		vec3 normalVec = vec3(-dirVec.y, dirVec.x, 1);
		//printf("    Implicit: %.1f x + %.1f y + %.1f = 0\n", normalVec.x, normalVec.y, -(vec3(np2.x - np1.x, np2.y - np1.y, 1)) * np1);
		printf("    Implicit: %.1f x + %.1f y + %.1f = 0\n", normalVec.x, normalVec.y,-normalVec.x * p1.x -normalVec.y*p1.y);
		//line going through points p and q
		// r(t) = p + (q-p)*t
		printf("    Parametric: r(t) = (%.1f, %.1f) + (%.1f, %.1f)t\n", np1.x, np1.y, np2.x - np1.x, np2.y - np1.y);
	};

	vec3 intersect(Line l2) {
		//function doesnt handle the case where the two lines are parallel, make sure no parallel line gets passed to it
		// Solve for intersection point using determinant

		float det = dirVec.x * l2.dirVec.y - dirVec.y * l2.dirVec.x; // 0 if the lines are equal -> division by zero!!!!
		float t = (dirVec.x * (p1.y - l2.p1.y) - dirVec.y * (p1.x - l2.p1.x)) / det;
		vec3 intersection = p1 + t * dirVec;
		return intersection;
	}

	boolean isPointNearLine(vec3 point) {
		float threshold = 0.01;
		//line: (x1,y1) , (x2,y2) point: (x0,y0)
		// |(y2-y1)*x0 - (x2-x1)*y0	+ x2*y1 - y2*x1|
		float nominator = (p2.y - p1.y) * point.x - (p2.x - p1.x) * point.y + p2.x * p1.y - p2.y * p1.x;
		//making sure to return the absolute value of the nominator
		if (nominator < 0) {
			nominator *= -1;
		}
		//sqrt((y2-y1)^2 + (x2*x1)^2)
		float denominator = sqrt(pow(p2.y - p1.y, 2) + pow(p2.x - p1.x, 2));

		return (nominator / denominator) < threshold;
	}

	void extend() {
		//this function extends a line segment to cross the boundaries of the (-1,-1), (1,1) square
		//the longest possible vector in the square would be one going from corner to corner, which has the length of sqrt(8) (sqrt(2^2 + 2^2))
		/*if we multiply the direction vector by a scalar to have a length of sqrt(8) and add it to p1 to get a new p1 and subtract it twice from new p1 to get the new p2,
		we can guarantee that both points are on the same line as before and outside the unit square*/

		float dirVecLen = sqrt(pow(dirVec.x, 2) + pow(dirVec.y, 2));
		float coef = sqrt(8) / dirVecLen;
		
		//altering the direction vector to have a correct length for correct behaviour
		dirVec.x = dirVec.x * coef;
		dirVec.y = dirVec.y * coef;

		//adding a direction vector with the length of sqrt(8)
		p1.x += dirVec.x;
		p1.y += dirVec.y;
		
		//to get p2, we subtract the same vector twice from p1
		p2.x = p1.x - 2 * (dirVec.x);
		p2.y = p1.y - 2 * (dirVec.y);
	}

	void passThroughPoint(vec3 point) {
		//p1
		p1.x  = point.x + dirVec.x;
		p1.y  = point.y + dirVec.y;
		//p2
		p2.x  = point.x - dirVec.x;
		p2.y  = point.y - dirVec.y;
	}
};
class LineCollection {
	Object lines;
	std::vector<Line> internalLineStorage;
public:
	void addLine(Line newLine) {
			lines.Vtx().push_back(newLine.getp1());
			lines.Vtx().push_back(newLine.getp2());
			internalLineStorage.push_back(newLine);
			lines.updateGPU();
		}
	void Draw() {
			lines.Draw(GL_LINES, vec3(0.0f, 1.0f, 1.0f));
		}
	Line findNearbyLine(vec3 point){
		//iterating over every line stored so far
		for (int i = 0; i < internalLineStorage.size(); i++){
			//if we found a line near the point we just return it
			if (internalLineStorage.at(i).isPointNearLine(point)){
				return internalLineStorage.at(i);
			}
		}
		//if not, we return a line with every point set to 0. a valid line will never have these coordinates, so this can be handled correctly.
		return Line(vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, 1.0f));
	}
};

	// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
	const char* const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, vp.z, 1);		// transform vp from modeling space to normalized device space
	}
)";

	// fragment shader in GLSL
	const char* const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";
	//declaring objects
	PointCollection* memPoints;
	LineCollection* memLines;
	// Initialization, create an OpenGL context
	void onInitialization() {
		glViewport(0, 0, winSize, winSize);
		//sizing
		glPointSize(10);
		glLineWidth(3);

		//initializing the program in point mode
		programMode = addPoint;
		//point and line collection
		memPoints = new PointCollection();
		memLines = new LineCollection();

		/*TESTING
		Line testLine = Line(vec3(-0.1f, -0.1f, 1.0f), vec3(0.1f, 0.1f, 1.0f));
		testLine.extend();
		memLines->addLine(testLine);*/
		
		// create program for the GPU
		gpuProgram.create(vertexSource, fragmentSource, "outColor");
	}

	// Window has become invalid: Redraw
	void onDisplay() {
		glClearColor(0.2, 0.2, 0.2, 0);     // background color
		glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

		// Set color to (0, 1, 0) = green
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 0.0f, 0.0f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		//glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		memPoints->Draw();
		memLines->Draw();
		glutSwapBuffers(); // exchange buffers for double buffering
	}

	// Key of ASCII code pressed
	void onKeyboard(unsigned char key, int pX, int pY) {

		//changing program modes according to keyboard input
		switch (key){
		case 'p':
			programMode = addPoint;
			break;
		case 'l':
			programMode = addLine;
			break;
		case 'i':
			programMode = intersectLines;
			break;
		case 'm':
			programMode = moveLine;
			break;
		}

		printf("%s\n", getProgramStatus().c_str());
	}

	// Key of ASCII code released
	void onKeyboardUp(unsigned char key, int pX, int pY) {
	}

	// Move mouse with key pressed
	void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
		// Convert to normalized device space
		float cX = 2.0f * pX / winSize - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / winSize;
		printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
	}

	// Mouse click event
	void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
		// Convert to normalized device space
		float cX = 2.0f * pX / winSize - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / winSize;

		
		char* buttonStat;

		switch (state) {
		case GLUT_DOWN: buttonStat = "pressed"; break;
		case GLUT_UP:   buttonStat = "released"; break;
		}

		switch (button) {
		case GLUT_LEFT_BUTTON:
			switch (state) {
			case GLUT_DOWN:
				switch (programMode) {
				case addPoint:
					//adding a point if LMB is pressed and we are in add point mode
					printf("Point %.1f, %.1f added\n", cX, cY);
					memPoints->addPoint(vec3(cX, cY, 1.0f));
					break;
				case addLine:
					for (int i = 0; i < memPoints->getVtx().size();i++) {

						if (memPoints->isNear(memPoints->getVtx().at(i), vec3(cX, cY, 1.0f))) {
							//if our click is on a point and it is the first valid point we selected:
							if (lineCreationTemp.size() == 0) {
								//we append it to our temporary vector
								lineCreationTemp.push_back(memPoints->getVtx().at(i));
							}
							//if the vector has a size of one and we are adding a point which is not too close:
							if (lineCreationTemp.size() == 1 && !memPoints->isNear(lineCreationTemp.at(0), memPoints->getVtx().at(i))) {
								lineCreationTemp.push_back(memPoints->getVtx().at(i));
								//we can create a line from the first and the new point.
								Line tempLine = Line(lineCreationTemp.at(0), lineCreationTemp.at(1));
								tempLine.extend();
								memLines->addLine(tempLine);
								//clearing the vector
								lineCreationTemp.clear();
							}
						}
					}
					break;
				}
				break;
			}
			break;
		}
	}

	// Idle event indicating that some time elapsed: do animation here
	void onIdle() {
		long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	}
