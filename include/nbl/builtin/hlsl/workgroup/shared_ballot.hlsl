// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

#define getDWORD(IX) ((IX)>>5)
// bitfieldDWORDs essentially means 'how many DWORDs are needed to store ballots in bitfields, for each invocation of the workgroup'
#define bitfieldDWORDs getDWORD(_NBL_HLSL_WORKGROUP_SIZE_+31) // in case WGSZ is not a multiple of 32 we might miscalculate the DWORDs after the right-shift by 5 which is why we add 31


#endif