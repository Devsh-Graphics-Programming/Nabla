// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/material_compiler3/CTrueIR.h"

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/portable/vector_t.hlsl"


namespace nbl::asset::material_compiler3
{

constexpr auto ELL_ERROR = nbl::system::ILogger::E_LOG_LEVEL::ELL_ERROR;
using namespace nbl::system;

// TODO: Move these notes, they're for the backend.
// 
// Any material (unless its emission only) requires a shading normal to be computed and kept, this should be the first register.
// Materials are primitive agnostic, don't assume triangles, so any attribute interpolation happens in Material System attribute getting callbacks.
// In practice most renderers will apply them on triangles, the advice is to keep triangle indices in own live state raw or compressed (uint32_t base, and then uint16_t diffs).
// To use the material outside of Raytracing or Visibility Buffer one could use HW interpolators, so all interpolated attributes will sit in global input variables.
// Whenever Material Compiler calls `getUV` or `getTBN` its the user's choice how to implement that.
// 
// However whenever HW interpolators are not used, to compute a UV it costs:
// - 3x index buffer fetch (from primitive ID and index buffer BDA)
// - 3x dependent UV attribute fetch
// - 2 FMA per channel to interpolate
// And to compute a TBN its:
// - 3x index buffer fetch (from primitive ID and index buffer BDA)
// - 3x dependent fetch of Tangent (and optional Bitangent), or Rotation Around normal + scales
// - 2 FMA per channel to interpolate
// - 1 reconstruction of a TBN matrix (can be expensive, i.e. from a quaternion or include renormalizations of interpolated tangents)
// For this reason, it makes sense to spill interpolated UVs and Tangent+BitangentScale instead of recomputing.
// The material compiler will keep a bitmask of UVs and TBNs (TBN depends on UV set) already requested and will spill if user marks sets as expensive to recompute.
// 
// If everyone is using the same TBN set, we could back up the interaction and sample because these are just V and L transformed into the TBN.
// In theory this requires us to use anisotropic interactions whenever the isotropic BRDF has a normal perturbation which seems like an overhead.
// However specular AA mitigations which derive roughness from per-pixel surface curvature and path regularizations would force this upon us anyway.
// Problem is that importance sampling requires us to transform a sampled tangent space L into worldspace and that requires knowing the TBN.
// I can see some optimizations for TBN calc where we only recompute the tangent space .xy of L and V for each TBN since N stays constant and NdotX2 can be known.
// Since we must have full TBN for anisotropic BxDF, tangent space V can be computed quicker than fetching it. 
// 
// Already computed shading normals and other BRDF parameters will be computed on demand but also cached (statically or dynamically) in registers.
// Register leftovers can be passed between command streams to not re-fetch or recompute BxDF parameters.
// Register allocation should spill whole quantities (all of RGB of a color or XYZ of a normal) not singular channels.
// Registers spilled should be the ones with least upcoming uses (should assume streams will run as AOV emission, AOV throughputs, Emission, NEE Eval, Generate, Quotient order)
// Relative cost to recompute or refetch should be included in the decision whether to keep around when register allocating.
// Perturbed Normals should be spilled to VRAM instead of recomputed as they require fetching at least a quaternion and a texture sample.
// Spill should be a big circular buffer allocated per-subgroup - hard to control in Pipeline Shaders https://github.com/KhronosGroup/Vulkan-Docs/issues/2717
// So need to allocate worst case spill for all rays in a dispatch (although can do persistent threads and reduce the dispatch size a little)

bool CTrueIR::addMaterials(const CFrontendIR* forest)
{
	auto& forestPool = forest->getObjectPool();
	//
	struct StackEntry
	{
		const IExprNode* node;
		// using post-order like stack but with a pre-order DFS
		uint8_t visited = false;
	};
	core::stack<StackEntry> exprStack;
	for (const auto& rootH : forest->getMaterials())
	{
		// AST is Sum Expression to the BRDF nodes
		// We need to keep the Ancestor prefix as an unrolled linked list
	}
	// 
	return false;
}


}