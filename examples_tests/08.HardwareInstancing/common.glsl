#define kOptimalWorkgroupSize 256u

#ifdef __cplusplus

#define mat4 core::matrix4SIMD

#else
#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>

#include <irr/builtin/glsl/utils/indirect_commands.glsl>

#include <irr/builtin/glsl/utils/culling.glsl>
#endif


struct TransformProperty_t
{
	mat4 world;
};