#ifndef WAVES_COMMON
#define WAVES_COMMON

#include <nbl/builtin/glsl/math/complex.glsl>
#include <nbl/builtin/glsl/sampling/box_muller_transform.glsl>
#define PI 3.1415926538
#define G 9.8
#define UINT_MAX 4294967295u

struct displacement_spectrum
{
	vec2 d[3];
};

#endif