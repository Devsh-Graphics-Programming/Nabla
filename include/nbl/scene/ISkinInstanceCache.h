// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SCENE_I_SKIN_INSTANCE_CACHE_H_INCLUDED_
#define _NBL_SCENE_I_SKIN_INSTANCE_CACHE_H_INCLUDED_

#include "nbl/scene/ITransformTree.h"

namespace nbl::scene
{

class ISkinInstanceCache : public virtual core::IReferenceCounted
{
	public:
		// properties
		using skinning_matrix_t = core::matrix3x4SIMD;
		using joint_t = ITransformTree::node_t;
		using recomputed_stamp_t = ITransformTree::recomputed_stamp_t;
		// self
		using skin_instance_t = uint32_t;
		using AddressAllocator = core::GeneralpurposeAddressAllocator<skin_instance_t>;
		static inline constexpr auto invalid = AddressAllocator::invalid_address;

		//
        struct CreationParametersBase
        {
            video::ILogicalDevice* device;
            uint8_t minAllocSizeInJoints = 8u;
        };
        struct ImplicitBufferCreationParameters : CreationParametersBase
        {
			uint32_t jointCapacity;
        };
        struct ExplicitBufferCreationParameters : CreationParametersBase
        {
			asset::SBufferRange<video::IGPUBuffer> skinningMatrixBuffer;
			asset::SBufferRange<video::IGPUBuffer> jointNodeBuffer;
			asset::SBufferRange<video::IGPUBuffer> recomputedTimestampBuffer;
        };

		// useful for everyone
		static inline constexpr uint32_t BufferCount = 3u;
		static inline constexpr asset::ISpecializedShader::E_SHADER_STAGE DefaultDescriptorAccessFlags[BufferCount] = {
			asset::ISpecializedShader::ESS_COMPUTE,asset::ISpecializedShader::ESS_COMPUTE,asset::ISpecializedShader::ESS_COMPUTE
		};
		template<typename BindingType>
		static inline void fillDescriptorLayoutBindings(BindingType* bindings, const asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=DefaultDescriptorAccessFlags)
		{
			for (auto i=0u; i<BufferCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = stageAccessFlags ? stageAccessFlags[i]:asset::ISpecializedShader::ESS_ALL;
				bindings[i].samplers = nullptr;
			}
		}
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createDescriptorSetLayout(const asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=DefaultDescriptorAccessFlags)
		{
			asset::ICPUDescriptorSetLayout::SBinding bindings[BufferCount];
			fillDescriptorLayoutBindings(bindings,stageAccessFlags);
			return core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+BufferCount);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDescriptorSetLayout(video::ILogicalDevice* device, const asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=DefaultDescriptorAccessFlags)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[BufferCount];
			fillDescriptorLayoutBindings(bindings,stageAccessFlags);
			return device->createGPUDescriptorSetLayout(bindings,bindings+BufferCount);
		}

		//
		inline const video::IGPUDescriptorSet* getDescriptorSet() const {return m_descriptorSet.get();}

		//
		inline const auto& getSkinningMatrixMemoryBlock() const {return m_skinningMatrixBlock;}
		inline const auto& getJointNodeMemoryBlock() const {return m_jointNodeBlock;}
		inline const auto& getRecomputedTimestampMemoryBlock() const {return m_recomputedTimestampBlock;}

		//
		inline uint32_t getAllocatedSkinningMatrices() const
		{
			return m_skinAllocator.get_allocated_size();
		}
		inline uint32_t getFreeSkinningMatrices() const
		{
			return m_skinAllocator.get_free_size();
		}
		inline uint32_t getCapacitySkinningMatrices() const
		{
			// special case allows us to use `get_total_size`, because the allocator has no added offsets
			return m_skinAllocator.get_total_size();
		}

		//
		struct Allocation
		{
			uint32_t count = 0u;
			// must point to an array initialized with `invalid`
			skin_instance_t* skinInstances;
			// self explanatory
			const uint32_t* jointCountPerSkin;
		};
		inline bool allocate(Allocation& params)
		{
            for (auto i=0u; i<params.count; i++)
            {
                auto& skinInstance = params.skinInstances[i];
                if (skinInstance!=invalid)
                    continue;

				skinInstance = m_skinAllocator.alloc_addr(params.jointCountPerSkin[i],1u);
                if (skinInstance==invalid)
                    return false;
            }
			return true;
		}
		//
		inline void free(const Allocation& params)
		{
            for (auto i=0u; i<params.count; i++)
            {
                auto& skinInstance = params.skinInstances[i];
                if (skinInstance==invalid)
                    continue;

				m_skinAllocator.free_addr(skinInstance,params.jointCountPerSkin[i]);
            }
		}

		// This removes all cache entries
		inline void clear()
		{
			m_skinAllocator.reset();
		}

	protected:
		ISkinInstanceCache(
			uint8_t minAllocSizeInJoints, const uint32_t capacity, void* _skinAllocatorReserved,
			asset::SBufferRange<video::IGPUBuffer>&& _skinningMatrixBuffer,
			asset::SBufferRange<video::IGPUBuffer>&& _jointNodeBuffer,
			asset::SBufferRange<video::IGPUBuffer>&& _recomputedTimestampBuffer,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _descriptorSet
		) : m_skinAllocator(_skinAllocatorReserved,0u,0u,1u,capacity,minAllocSizeInJoints),
			m_skinAllocatorReserved(_skinAllocatorReserved),
			m_skinningMatrixBlock({ _skinningMatrixBuffer.offset,sizeof(skinning_matrix_t)*capacity,std::move(_skinningMatrixBuffer.buffer)}),
			m_jointNodeBlock({_jointNodeBuffer.offset,sizeof(joint_t)*capacity,std::move(_jointNodeBuffer.buffer)}),
			m_recomputedTimestampBlock({_recomputedTimestampBuffer.offset,sizeof(recomputed_stamp_t)*capacity,std::move(_recomputedTimestampBuffer.buffer)}),
			m_descriptorSet(std::move(_descriptorSet))
		{
			m_skinningMatrixBlock.buffer->setObjectDebugName("ISkinInstanceCache::skinning_matrix_t");
			m_jointNodeBlock.buffer->setObjectDebugName("ISkinInstanceCache::joint_t");
			m_recomputedTimestampBlock.buffer->setObjectDebugName("ISkinInstanceCache::recomputed_stamp_t");
		}
		virtual ~ISkinInstanceCache()
		{
			// m_skinAllocatorReserved must be freed outside
		}

		friend class ISkinInstanceCacheManager;
		
        AddressAllocator m_skinAllocator;
		void* m_skinAllocatorReserved;
        asset::SBufferRange<video::IGPUBuffer> m_skinningMatrixBlock;
        asset::SBufferRange<video::IGPUBuffer> m_jointNodeBlock;
        asset::SBufferRange<video::IGPUBuffer> m_recomputedTimestampBlock;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_descriptorSet;
};

} // end namespace nbl::scene

#endif

