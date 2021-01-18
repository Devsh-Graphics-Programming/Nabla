#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_BASIC_INCLUDED_


#include <nbl/builtin/glsl/math/typeless_arithmetic.glsl>
#include <nbl/builtin/glsl/subgroup/basic_portability.glsl>	


//! all functions must be called in uniform control flow (all workgroup invocations active)
bool nbl_glsl_workgroupElect()
{
	return gl_LocalInvocationIndex==0u;
}


#endif
