// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_


#include "nbl/asset/material_compiler3/CFrontendIR.h"


namespace nbl::asset::material_compiler3
{

// You make the Materials with a classical expression IR, one Root Node per material's interface layer, but here they're in "Accumulator Form"
// They appear "flipped upside down", its expected our backends will evaluate contributors first, and then bother with the attenuators. 
class CTrueIR : public CNodePool
{
		template<typename T>
		using _typed_pointer_type = CNodePool::obj_pool_type::mem_pool_type::typed_pointer_type<T>;

	public:
		// constructor
		using creation_params_type = typename obj_pool_type::creation_params_type;
		static inline core::smart_refctd_ptr<CTrueIR> create(creation_params_type&& params)
		{
			if (params.composed.blockSizeKBLog2<4)
				return nullptr;
			return core::smart_refctd_ptr<CTrueIR>(new CTrueIR(std::move(params)),core::dont_grab);
		}		

		// basic "built-in" nodes
		class INode : public CNodePool::INode
		{
			public:

			protected:
				// Each node is final and immutable, has a precomputed hash for the whole subtree beneath it.
				// Debug info does not form part of the hash, so can get wildly replaced.
				core::blake3_hash_t hash;
		};
		//
		template<typename T> requires std::is_base_of_v<INode,std::remove_const_t<T>>
		using typed_pointer_type = _typed_pointer_type<T>;
		// To use a bump map, the Material needs to be provided UVs (which can or can not have associated tangents and smooth normals), but that's the responsibility of backend.
		// The bump map just needs to be provided a TBN basis thats orthogonal so that normals or derivatives can be transformed.
		// Normal maps will use Olano so won't be assumed to be normalized (they wouldn't be anyway due to interpolation), and derivative maps are never normalized and have huge range.
		// Depending on the mode (deformation stretch enabled or not) the TB of the matrix will be normalized before the multiplication so that TBN is orthonormal and does not impart a UV stretch.
		// After multiplication the normal will be normalized again, which is when we should apply Schussler or Yining.
		//  
		// Schussler and Yining require we define a tangential microfacet from the perturbed normal `P` but differently:
		// - Schussler wants normalize({-P.x,-P.y,0}) which is totally tangential and orthogonal to the geometric normal
		// - Yining wants normalize({-P.xy*P.z,1-P.z*P.z}) which is orthogonal to the perturbed normal
		// Both give an NDF with average normal projected onto the geometric normal being colinear to the geometric normal.
		// Each microfacet has different pros and cons:
		// - Tangential microfacet can use 100% perfect mirror material because its impossible to get 1-order scattering from it (camera and light can't see it at the same time)
		// - Mirror Tangential microfacet makes 2-nd order scattering tractable, just evaluate the BRDF 3 times with original L and V, then once with reflected L and reflected V
		// - for Lambertian BRDF this is only 2 evaluations because reflected V doesn't change anything
		// - the dot products with reflected L or V can be computed quickly from the regular ones, but `H` always changes in this case
		// - Orthogonal microfacet makes sure zero masking is achieved when LdotP=1 and appearance is 100% preserved in this case
		// - Orthogonal microfacet can be seen directly and therefore it needs to have the same BxDF as the perturbed microfacet
		// - The above requirements makes 2-nd order scattering intractable for orthogonal microfacets without random walks in the surface profile
		// - We only need to evaluate the BxDF twice with Yinning and all products with perturbed normal can be obtained similarly to obtaining the reflected L and V for schussler (permutations and scaling)
		// Both approaches are imperfect and in the limit will cause "fresnel lens" like appearance with Cook Torrance low roughness materials, you'll see a blend between classical bump mapped appearance and:
		// - for 2nd-order Schussler a reflection of the objects by a flipped normal {-P.xy,P.z} due to mirror microfacet making a "virtual normal" behind the mirror 
		// - for 1st-order Yinning a reflection of the objects by an orthogonal normal which approaches the geometric normal as the perturbed normal gets more extreme
		// This means that a tangential microfacet will show no blend/retroreflection if both L and V are behind the tangential microfacet.
		// Whereas an orthogonal microfacet will show no blend/retroreflection if both L and V are below the orthogonal microfacet, so tangential ensures this property for half the hemisphere.
		// But with extreme normal perturbation, the appearance of Schussler will match that of a very deep V-groove losing a lot of energy, but Yining will look like a flat surface of the same material.
		// This is because Schussler has no bound on the ratio of the projected microsurface area on the geometric surface, whereas Yining bounds it to be <= 2.0 
		// 
		// There's also a Schussler variant where the tangential microfacet has the same BRDF as the perturbed one, with only 1-st order scattering.
		// Then with this you can actually see the tangential microfacet (smooth mirror makes it black with only 1st order), which works better for BSDF than 2nd order scattering with a mirror facet.
		// Note that transmissive or smooth glass tangential face could never work, because it presents same issues as a non-tangential mirror facet (V and L can interact directly in 1-st order).
		// Mirror facet would cause you to see refractions from a reflected V, which would be weird when the VdotP->0 and you'd expect 100% fresnel reflection. Worse yet you'd see them in the "wrong" direction.
		// Having a tangential microfacet with same BSDF does blend towards transmission as `VdotP->0`, but at least the refractive rays correctly have `L.xy = -V.xy` and are going "forward" although
		// they are bending upwards vs. the transmitted direction, not downwards like they would against the macro surface or the perturbed surface before VdotP==0.
		// 
		// Both this and Yining are faster because of the 2 BxDF evaluations instead of 3, and they only differ by the inter-facet shadowing and masking functions and the tangent facet normal is computed.
		// However an orthogonal microfacet will work better for BSDFs because the facet tangents and normals form a square and not a rhombus, which means that a V to the "left" of the perturbed normal
		// will always be to the "right" of the orthogonal microfacet, meaning we'll get a consistent refraction (both L=refract(V,facet) will curve away from -V in the same direction.


		// Each material comes down to this
		struct Material
		{
//			TypedHandle<CRootNode> root;
			CNodePool::typed_pointer_type<CNodePool::CDebugInfo> debugInfo;
			//
			constexpr static inline uint8_t MaxUVSlots = 32;
			std::bitset<MaxUVSlots> usedUVSlots;
			// the tangent frames are a subset of used UV slots, unless there's an anisotropic BRDF involved
			std::bitset<MaxUVSlots> usedTangentFrames;
		};
		inline std::span<const Material> getMaterials() const {return m_materials;}

		// We take the trees from the forest, and canonicalize them into our weird Domain Specific IR with Upside down expression trees.
		// Process:
		// 1. Schusslerization (for derivative map usage) and Decompression (duplicating nodes, etc.)
		// 2. Canonicalize Expressions (Transform into Sum-Product form, DCE, etc.)
		// 3. Split BTDFs (front vs. back part), reciprocate Etas
		// 4. Simplify and Hoist Layer terms (delta sampling property)
		// 5. Subexpression elimination
		// It is the backend's job to handle:
		// - constant encoding precision (scale factors, UV matrices, IoRs)
		// - multiscatter compensation
		// - compilation failure to unsupported complex layering
		// - compilation failure to unsupported complex layering
		NBL_API2 bool addMaterials(const CFrontendIR* forest);

	protected:
		using CNodePool::CNodePool;

		core::vector<Material> m_materials;
		core::unordered_map<core::blake3_hash_t,typed_pointer_type<INode>> m_uniqueNodes;
};

//! DAG (baked)

} // namespace nbl::asset::material_compiler3

#endif
