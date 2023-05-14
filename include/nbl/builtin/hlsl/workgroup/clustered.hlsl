// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_CLUSTERED_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_CLUSTERED_INCLUDED_

#include <nbl/builtin/hlsl/workgroup/shared_clustered.hlsl>
#include <nbl/builtin/hlsl/workgroup/ballot.hlsl>

/** TODO: @Hazardu or @Przemog or lets have it as a recruitment task
// `clusterSize` needs to be Power of Two, but the workgroup size does not!
// use `nbl_glsl_subgroupOp` to implement optimally

float nbl_glsl_workgroupClusteredAnd(in float val, in uint clusterSize);
uint nbl_glsl_workgroupClusteredAnd(in uint val, in uint clusterSize);
int nbl_glsl_workgroupClusteredAnd(in int val, in uint clusterSize);

bool nbl_glsl_workgroupClusteredXor(in bool val, in uint clusterSize);
float nbl_glsl_workgroupClusteredXor(in float val, in uint clusterSize);
uint nbl_glsl_workgroupClusteredXor(in uint val, in uint clusterSize);
int nbl_glsl_workgroupClusteredXor(in int val, in uint clusterSize);

bool nbl_glsl_workgroupClusteredOr(in bool val, in uint clusterSize);
float nbl_glsl_workgroupClusteredOr(in float val, in uint clusterSize);
uint nbl_glsl_workgroupClusteredOr(in uint val, in uint clusterSize);
int nbl_glsl_workgroupClusteredOr(in int val, in uint clusterSize);

bool nbl_glsl_workgroupClusteredAdd(in bool val, in uint clusterSize);
float nbl_glsl_workgroupClusteredAdd(in float val, in uint clusterSize);
uint nbl_glsl_workgroupClusteredAdd(in uint val, in uint clusterSize);
int nbl_glsl_workgroupClusteredAdd(in int val, in uint clusterSize);

// mul and min/max dont need boolean variants, since they're achievable with And and Or
float nbl_glsl_workgroupClusteredMul(in float val, in uint clusterSize);
uint nbl_glsl_workgroupClusteredMul(in uint val, in uint clusterSize);
int nbl_glsl_workgroupClusteredMul(in int val, in uint clusterSize);

float nbl_glsl_workgroupClusteredMin(in float val, in uint clusterSize);
uint nbl_glsl_workgroupClusteredMin(in uint val, in uint clusterSize);
int nbl_glsl_workgroupClusteredMin(in int val, in uint clusterSize);

float nbl_glsl_workgroupClusteredMax(in float val, in uint clusterSize);
uint nbl_glsl_workgroupClusteredMax(in uint val, in uint clusterSize);
int nbl_glsl_workgroupClusteredMax(in int val, in uint clusterSize);
*/

#endif