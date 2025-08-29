// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_


#include "nbl/asset/material_compiler3/CFrontendIR.h"


namespace nbl::asset::material_compiler3
{

// You make the Materials with a classical expression IR, one Root Node per material's interface layer, but here they're in "Accumulator Form"
// They appeared "flipped upside down" 
class CTrueIR : public CNodePool
{
	public:
		// constructor
		inline core::smart_refctd_ptr<CTrueIR> create(const uint8_t chunkSizeLog2=19, const uint8_t maxNodeAlignLog2=4, refctd_pmr_t&& _pmr={})
		{
			if (chunkSizeLog2<14 || maxNodeAlignLog2<4)
				return nullptr;
			if (!_pmr)
				_pmr = core::getDefaultMemoryResource();
			return core::smart_refctd_ptr<CTrueIR>(new CFrontendIR(chunkSizeLog2,maxNodeAlignLog2,std::move(_pmr)),core::dont_grab);
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
		template<typename T> requires std::is_base_of_v<INode,T>
		using TypedHandle = CNodePool::TypedHandle<T>;


		// Each material comes down to this
		struct Material
		{
//			TypedHandle<CRootNode> root;
			CNodePool::TypedHandle<CNodePool::CDebugInfo> debugInfo;
		};
		inline std::span<const Material> getMaterials() const {return m_materials;}

		// We take the trees from the forest, and canonicalize them into our weird Domain Specific IR with Upside down expression trees.
		bool addMaterials(const CFrontendIR* forest);

	protected:
		using CNodePool::CNodePool;

		core::vector<Material> m_materials;
		core::unordered_map<core::blake3_hash_t,Handle> m_uniqueNodes;
};

//! DAG (baked)

} // namespace nbl::asset::material_compiler3

#endif