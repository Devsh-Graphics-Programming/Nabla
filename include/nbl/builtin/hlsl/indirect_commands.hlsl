// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_INDIRECT_COMMANDS_INCLUDED_
#define _NBL_BUILTIN_HLSL_INDIRECT_COMMANDS_INCLUDED_


#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl
{
namespace hlsl
{

struct DrawArraysIndirectCommand_t
{
	uint32_t  count;
	uint32_t  instanceCount;
	uint32_t  first;
	uint32_t  baseInstance;
};

struct DrawElementsIndirectCommand_t
{
	uint32_t count;
	uint32_t instanceCount;
	uint32_t firstIndex;
	uint32_t baseVertex;
	uint32_t baseInstance;
};

struct DispatchIndirectCommand_t
{
	uint32_t  num_groups_x;
	uint32_t  num_groups_y;
	uint32_t  num_groups_z;
};

}
}

#endif