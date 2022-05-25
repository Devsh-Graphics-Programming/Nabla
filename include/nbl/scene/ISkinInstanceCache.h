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
		using skin_instance_t = uint32_t;
		using AddressAllocator = core::GeneralpurposeAddressAllocator<skin_instance_t>;
		static inline constexpr auto invalid_instance = AddressAllocator::invalid_address;

		// main pseudo-pool properties
		using joint_t = ITransformTree::node_t;
		using skinning_matrix_t = core::matrix3x4SIMD;
		using recomputed_stamp_t = ITransformTree::recomputed_stamp_t;
		using inverse_bind_pose_offset_t = uint32_t;

		// for correct intialization
		static inline constexpr recomputed_stamp_t initial_recomputed_timestamp = 0xffffffffu;
		static_assert(initial_recomputed_timestamp<ITransformTree::min_timestamp || initial_recomputed_timestamp>ITransformTree::max_timestamp);
		static_assert(initial_recomputed_timestamp!=ITransformTree::initial_modified_timestamp);
		static_assert(initial_recomputed_timestamp!=ITransformTree::initial_recomputed_timestamp);

		static inline constexpr uint32_t joint_prop_ix = 0u;
		static inline constexpr uint32_t skinning_matrix_prop_ix = 1u;
		static inline constexpr uint32_t recomputed_stamp_prop_ix = 2u;
		static inline constexpr uint32_t inverse_bind_pose_offset_prop_ix = 3u;

		// for the inverse bind pose pool
		using inverse_bind_pose_t = core::matrix3x4SIMD;
		static inline constexpr uint32_t inverse_bind_pose_prop_ix = 0u;
		

		//
		static inline constexpr uint32_t CacheDescriptorSetBindingCount = 7u;
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createCacheDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			asset::ICPUDescriptorSetLayout::SBinding bindings[CacheDescriptorSetBindingCount];
            asset::ICPUDescriptorSetLayout::fillBindingsSameType(bindings,CacheDescriptorSetBindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+CacheDescriptorSetBindingCount);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createCacheDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[CacheDescriptorSetBindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,CacheDescriptorSetBindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createDescriptorSetLayout(bindings,bindings+CacheDescriptorSetBindingCount);
		}
		//
		template<class TransformTree>
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createRenderDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			constexpr auto BindingCount = TransformTree::RenderDescriptorSetBindingCount+1u;
			asset::ICPUDescriptorSetLayout::SBinding bindings[BindingCount];
            asset::ICPUDescriptorSetLayout::fillBindingsSameType(bindings,BindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+BindingCount);
		}
		template<class TransformTree>
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createRenderDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			constexpr auto BindingCount = TransformTree::RenderDescriptorSetBindingCount+1u;
			video::IGPUDescriptorSetLayout::SBinding bindings[BindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,BindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createDescriptorSetLayout(bindings,bindings+BindingCount);
		}

		//
        struct CreationParametersBase
        {
            video::ILogicalDevice* device;
			core::smart_refctd_ptr<ITransformTree> associatedTransformTree;
            uint8_t minAllocSizeInJoints = 8u;

		protected:
			inline bool isValid() const
			{
				return device && associatedTransformTree &&	minAllocSizeInJoints!=0u;
			}
        };
        struct ImplicitCreationParameters : CreationParametersBase
        {
			uint32_t jointCapacity;
			uint32_t inverseBindPoseCapacity;

			inline bool isValid() const
			{
				return CreationParametersBase::isValid() && jointCapacity!=0u;
			}
        };
        struct ExplicitCreationParameters : CreationParametersBase
        {
			asset::SBufferRange<video::IGPUBuffer> jointNodeBuffer;
			asset::SBufferRange<video::IGPUBuffer> skinningMatrixBuffer;
			asset::SBufferRange<video::IGPUBuffer> recomputedTimestampBuffer;
			asset::SBufferRange<video::IGPUBuffer> inverseBindPoseOffsetBuffer;
			core::smart_refctd_ptr<video::IPropertyPool> inverseBindPosePool;

			inline bool isValid() const
			{
				return CreationParametersBase::isValid() &&
					jointNodeBuffer.isValid() &&
					skinningMatrixBuffer.isValid() &&
					recomputedTimestampBuffer.isValid() &&
					inverseBindPoseOffsetBuffer.isValid() &&
					inverseBindPosePool &&
					inverseBindPosePool->getPropertyCount() >= 1u &&
					inverseBindPosePool->getPropertySize(inverse_bind_pose_prop_ix) == sizeof(inverse_bind_pose_t);
			}
        };

		//
		inline const video::IGPUDescriptorSet* getCacheDescriptorSet() const {return m_cacheDS.get();}
		inline const video::IGPUDescriptorSet* getRenderDescriptorSet() const {return m_renderDS.get();}

		//
		inline ITransformTree* getAssociatedTransformTree() { return m_tt.get(); }
		inline const ITransformTree* getAssociatedTransformTree() const { return m_tt.get(); }
		
		//
		inline auto* getInverseBindPosePool() { return m_inverseBindPosePool.get(); }
		inline const auto* getInverseBindPosePool() const { return m_inverseBindPosePool.get(); }
		
		//
		inline const auto& getJointNodeMemoryBlock() const { return m_jointNodeBlock; }
		inline const auto& getSkinningMatrixMemoryBlock() const {return m_skinningMatrixBlock;}
		inline const auto& getRecomputedTimestampMemoryBlock() const {return m_recomputedTimestampBlock;}
		inline const auto& getInverseBindPoseOffsetMemoryBlock() const {return m_inverseBindPoseOffsetBlock;}

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
			Allocation() : skinInstances(nullptr,nullptr) {}

			inline uint32_t computeSkinInstanceTotalCount() const
			{
				const auto skinCount = skinInstances.size();
				if (instanceCounts)
					return std::accumulate(instanceCounts,instanceCounts+skinCount,0u);
				else
					return skinCount;
			}

			inline bool isValid() const
			{
				return skinInstances.begin() && skinInstances.begin()<=skinInstances.end() && jointCountPerSkin;
			}

			// must point to arrays initialized with `invalid` large enough to hold 1 value per instance (as dictated by `instanceCounts`)
			core::SRange<ISkinInstanceCache::skin_instance_t* const> skinInstances;
			// self explanatory, needs to point at memory with `outSkinInstances.size()` uint32_t
			const uint32_t* jointCountPerSkin;
			// if nullptr then treated like a buffer of {1,1,...,1,1}, else needs to be same length as the skeleton range
			const uint32_t* instanceCounts = nullptr;
		};
		//
		[[nodiscard]] inline bool allocate(const Allocation& params)
		{
            for (auto i=0u; i<params.skinInstances.size(); i++)
            {
				const auto jointCount = params.jointCountPerSkin[i];
				const auto instanceCount = params.instanceCounts ? params.instanceCounts[i]:1u;
				for (auto j=0u; j<instanceCount; j++)
				{
					auto& skinInstance = params.skinInstances.begin()[i][j];
					if (skinInstance!=invalid_instance)
						continue;

					skinInstance = m_skinAllocator.alloc_addr(jointCount,1u);
					if (skinInstance==invalid_instance)
						return false;
				}
            }
			return true;
		}
		//
		inline void free(const Allocation& params)
		{
            for (auto i=0u; i<params.skinInstances.size(); i++)
            {
				const auto jointCount = params.jointCountPerSkin[i];
				const auto instanceCount = params.instanceCounts ? params.instanceCounts[i]:1u;
				for (auto j=0u; j<instanceCount; j++)
				{
					auto& skinInstance = params.skinInstances.begin()[i][j];
					if (skinInstance==invalid_instance)
						continue;

					m_skinAllocator.free_addr(skinInstance,jointCount);
				}
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
			asset::SBufferRange<video::IGPUBuffer>&& _jointNodeBuffer,
			asset::SBufferRange<video::IGPUBuffer>&& _skinningMatrixBuffer,
			asset::SBufferRange<video::IGPUBuffer>&& _recomputedTimestampBuffer,
			core::smart_refctd_ptr<ITransformTree>&& _tt,
			asset::SBufferRange<video::IGPUBuffer>&& _inverseBindPoseOffsetBuffer,
			core::smart_refctd_ptr<video::IPropertyPool>&& _inverseBindPosePool,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _cacheDS,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _renderDS
		) : m_skinAllocator(_skinAllocatorReserved,0u,0u,1u,capacity,minAllocSizeInJoints),
			m_skinAllocatorReserved(_skinAllocatorReserved),
			m_jointNodeBlock({_jointNodeBuffer.offset,sizeof(joint_t)*capacity,std::move(_jointNodeBuffer.buffer)}),
			m_skinningMatrixBlock({ _skinningMatrixBuffer.offset,sizeof(skinning_matrix_t)*capacity,std::move(_skinningMatrixBuffer.buffer)}),
			m_recomputedTimestampBlock({_recomputedTimestampBuffer.offset,sizeof(recomputed_stamp_t)*capacity,std::move(_recomputedTimestampBuffer.buffer)}),
			m_tt(std::move(_tt)),
			m_inverseBindPoseOffsetBlock({_inverseBindPoseOffsetBuffer.offset,sizeof(inverse_bind_pose_offset_t)*capacity,std::move(_inverseBindPoseOffsetBuffer.buffer)}),
			m_inverseBindPosePool(std::move(_inverseBindPosePool)), m_cacheDS(std::move(_cacheDS)), m_renderDS(std::move(_renderDS))
		{
			m_jointNodeBlock.buffer->setObjectDebugName("ISkinInstanceCache::joint_t");
			m_skinningMatrixBlock.buffer->setObjectDebugName("ISkinInstanceCache::skinning_matrix_t");
			m_recomputedTimestampBlock.buffer->setObjectDebugName("ISkinInstanceCache::recomputed_stamp_t");
			m_inverseBindPoseOffsetBlock.buffer->setObjectDebugName("ISkinInstanceCache::inverse_bind_pose_offset_t");
			m_inverseBindPosePool->getPropertyMemoryBlock(inverse_bind_pose_prop_ix).buffer->setObjectDebugName("ISkinInstanceCache::inverse_bind_pose_t");
		}
		virtual ~ISkinInstanceCache()
		{
			// m_skinAllocatorReserved must be freed outside
		}

		friend class ISkinInstanceCacheManager;
		
        AddressAllocator m_skinAllocator;
		void* m_skinAllocatorReserved;
		asset::SBufferRange<video::IGPUBuffer> m_jointNodeBlock;
        asset::SBufferRange<video::IGPUBuffer> m_skinningMatrixBlock;
        asset::SBufferRange<video::IGPUBuffer> m_recomputedTimestampBlock;
		core::smart_refctd_ptr<ITransformTree> m_tt;
        asset::SBufferRange<video::IGPUBuffer> m_inverseBindPoseOffsetBlock;
		core::smart_refctd_ptr<video::IPropertyPool> m_inverseBindPosePool;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_cacheDS,m_renderDS;
};

} // end namespace nbl::scene

#endif

