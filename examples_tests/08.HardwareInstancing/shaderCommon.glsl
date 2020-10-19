#include "common.glsl"

struct VisibleObject_t
{
	mat4	modelViewProjectionMatrix;
	vec3	normalMatrixRow0;
	uint	cameraUUID;
	vec3	normalMatrixRow1;
	uint	objectUUID;
	vec3	normalMatrixRow2;
	uint	meshUUID;
};


/**

We know what objects we want to draw with which mesh and for what camera.
Per-camera MDIs have been cleared

Now we can sort by camera OR start expanding meshes into meshbuffers... what to do?

*/