// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_NODE_POOL_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_NODE_POOL_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/core/definitions.h"
#include "nbl/core/alloc/refctd_memory_resource.h"

#include <type_traits>


namespace nbl::asset::material_compiler3
{

// Class to manage all nodes' backing and hand them out as `uint32_t` handles
class CNodePool : public core::IReferenceCounted
{
		struct Config
		{
			using AddressAllocator = core::LinearAddressAllocator<uint32_t>;
			using HandleValue = uint32_t;
			constexpr static inline bool ThreadSafe = false;
		};

	public:
		//
		using obj_pool_type = nbl::core::CObjectPool<Config>;

		//
		inline obj_pool_type& getObjectPool() {return m_composed;}
		inline const obj_pool_type& getObjectPool() const {return m_composed;}
		
		//
		class INode
		{
			public:
				virtual const std::string_view getTypeName() const = 0;
		};
		//
		template<typename T> requires std::is_base_of_v<INode,std::remove_const_t<T>>
		using typed_pointer_type = obj_pool_type::template typed_pointer_type<T>;

		// Debug Info node
		class CDebugInfo : public obj_pool_type::IVariableSize, public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::asset::material_compiler3::CNodePool::CDebugInfo";}
				inline uint32_t getSize() const {return calc_size(nullptr,m_size);}
				
				static inline uint32_t calc_size(const void* data, const uint32_t size)
				{
					return core::alignUp(sizeof(CDebugInfo)+size,alignof(CDebugInfo));
				}
				static inline uint32_t calc_size(const std::string_view& view)
				{
					return calc_size(nullptr,view.length()+1);
				}
				inline CDebugInfo(const void* data, const uint32_t size) : m_size(size)
				{
					if (data)
						memcpy(std::launder(this+1),data,m_size);
				}
				inline CDebugInfo(const std::string_view& view) : CDebugInfo(nullptr,view.length()+1)
				{
					auto* out = std::launder(reinterpret_cast<char*>(this+1));
					if (m_size>1)
						memcpy(out,view.data(),m_size);
					out[m_size-1] = 0;
				}

				inline const std::span<const uint8_t> data() const
				{
					return {reinterpret_cast<const uint8_t*>(this+1),m_size};
				}

			protected:
				const uint32_t m_size;
		};

		template<typename T>
		inline const std::string_view getTypeName(const typed_pointer_type<T> h) const
		{
			const auto* node = getObjectPool().deref<const T>(h);
			return node ? node->getTypeName():"nullptr";
		}
		

	protected:
		inline CNodePool(typename obj_pool_type::creation_params_type&& params) : m_composed(std::move(params)) {}

		obj_pool_type m_composed;
};

} // namespace nbl::asset::material_compiler3
#endif