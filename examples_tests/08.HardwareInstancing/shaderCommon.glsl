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