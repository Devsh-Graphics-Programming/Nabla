#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_BASIC_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_BASIC_INCLUDED_


#include <irr/builtin/glsl/math/typeless_arithmetic.glsl>
#include <irr/builtin/glsl/subgroup/basic_portability.glsl>	


//! all functions must be called in uniform control flow (all workgroup invocations active)
bool irr_glsl_workgroupElect()
{
	return gl_LocalInvocationIndex==0u;
}


#endif
