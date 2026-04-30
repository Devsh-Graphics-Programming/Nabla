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
		bool addMaterials(const CFrontendIR* forest);

	protected:
		using CNodePool::CNodePool;

		core::vector<Material> m_materials;
		core::unordered_map<core::blake3_hash_t,typed_pointer_type<INode>> m_uniqueNodes;
};

//! DAG (baked)

} // namespace nbl::asset::material_compiler3

#endif
