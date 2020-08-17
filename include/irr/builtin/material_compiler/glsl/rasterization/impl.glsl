#version 430 core

#include <irr/builtin/material_compiler/glsl/common.glsl>

void main()
{
	mat2 dUV = mat2(dFdx(UV),dFdy(UV));

	InstanceData instData = InstData.data[InstanceIndex];
	runTexPrefetchStream(instData.prefetch_instrStream);
	runNormalPrecompStream(instData.nprecomp_instrStream);


}