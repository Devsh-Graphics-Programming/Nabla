// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_STOCHASTIC_LIGHTCUT_TREE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_STOCHASTIC_LIGHTCUT_TREE_INCLUDED_

// Umbrella header for the stochastic light-cut tree. Split by concern; include a
// part directly if that is all you need.
//


#include <nbl/builtin/hlsl/sampling/lightcut_tree/node.hlsl> // decoded views the sampler consumes, layout-agnostic
#include <nbl/builtin/hlsl/sampling/lightcut_tree/packing.hlsl> // the 32 B CWBVH-4 byte layout, encode and decode
#include <nbl/builtin/hlsl/sampling/lightcut_tree/weight.hlsl> // per-child importance, the receiver-aware weight modes
#include <nbl/builtin/hlsl/sampling/lightcut_tree/sampler.hlsl> // the descent itself, generate / forwardPdf / backwardPdf

#endif
