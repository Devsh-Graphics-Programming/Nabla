#ifndef _VIRTUAL_GEOMETRY_GLSL_INCLUDED_
#define _VIRTUAL_GEOMETRY_GLSL_INCLUDED_


#define _NBL_VG_USE_SSBO
#define _NBL_VG_USE_SSBO_UINT
#define _NBL_VG_SSBO_UINT_BINDING 0
#define _NBL_VG_USE_SSBO_UVEC3
#define _NBL_VG_SSBO_UVEC3_BINDING 1
#define _NBL_VG_USE_SSBO_INDEX
#define _NBL_VG_SSBO_INDEX_BINDING 2
// TODO: remove after all quantization optimizations in CSerializedLoader and the like
#define _NBL_VG_USE_SSBO_UVEC2
#define _NBL_VG_SSBO_UVEC2_BINDING 3
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl>


#endif
