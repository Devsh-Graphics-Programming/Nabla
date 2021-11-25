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
		static_assert(initial_recomputed_timestamp!=ITransformTree::initial_recomputed_timestamp);

		static inline constexpr uint32_t joint_prop_ix = 0u;
		static inline constexpr uint32_t skinning_matrix_prop_ix = 1u;
		static inline constexpr uint32_t recomputed_stamp_prop_ix = 2u;
		static inline constexpr uint32_t inverse_bind_pose_offset_prop_ix = 3u;

		static inline constexpr uint32_t inverse_bind_pose_prop_ix = 0u;
		

		//
		static inline constexpr uint32_t CacheDescriptorSetBindingCount = 7u;
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createCacheDescriptorSetLayout(asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			asset::ICPUDescriptorSetLayout::SBinding bindings[CacheDescriptorSetBindingCount];
            asset::ICPUDescriptorSetLayout::fillBindingsSameType(bindings,CacheDescriptorSetBindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+CacheDescriptorSetBindingCount);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createCacheDescriptorSetLayout(video::ILogicalDevice* device, asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[CacheDescriptorSetBindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,CacheDescriptorSetBindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createGPUDescriptorSetLayout(bindings,bindings+CacheDescriptorSetBindingCount);
		}
		//
		template<class TransformTree>
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createRenderDescriptorSetLayout(asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			constexpr auto BindingCount = TransformTree::RenderDescriptorSetBindingCount+1u;
			asset::ICPUDescriptorSetLayout::SBinding bindings[BindingCount];
            asset::ICPUDescriptorSetLayout::fillBindingsSameType(bindings,BindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+BindingCount);
		}
		template<class TransformTree>
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createRenderDescriptorSetLayout(video::ILogicalDevice* device, asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			constexpr auto BindingCount = TransformTree::RenderDescriptorSetBindingCount+1u;
			video::IGPUDescriptorSetLayout::SBinding bindings[BindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,BindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createGPUDescriptorSetLayout(bindings,bindings+BindingCount);
		}










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
			uint32_t count = 0u;
			// must point to an array initialized with `invalid`
			skin_instance_t* skinInstances;
			// self explanatory
			const uint32_t* jointCountPerSkin;
		};
		[[nodiscard]] inline bool allocate(const Allocation& params)
		{
            for (auto i=0u; i<params.count; i++)
            {
                auto& skinInstance = params.skinInstances[i];
                if (skinInstance!=invalid_instance)
                    continue;

				skinInstance = m_skinAllocator.alloc_addr(params.jointCountPerSkin[i],1u);
                if (skinInstance==invalid_instance)
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
                if (skinInstance==invalid_instance)
                    continue;

				m_skinAllocator.free_addr(skinInstance,params.jointCountPerSkin[i]);
            }
		}
		// TODO: setup transfers, etc.

		// This removes all cache entries
		inline void clear()
		{
			m_skinAllocator.reset();
		}
#if 0
		template<class TransformTree, typename... Args>
		static inline bool create(
			core::smart_refctd_ptr<typename TransformTree::property_pool_t>& outPool,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>& outPoolDS,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>& outRenderDS,
			video::ILogicalDevice* device, Args... args
		)
		{
			using property_pool_t = typename TransformTree::property_pool_t;
			outPool = property_pool_t::create(device,std::forward<Args>(args)...);
			if (!outPool)
				return false;

			video::IDescriptorPool::SDescriptorPoolSize size = {asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,property_pool_t::PropertyCount+TransformTree::RenderDescriptorSetBindingCount};
			auto dsp = device->createDescriptorPool(video::IDescriptorPool::ECF_NONE,1u,1u,&size);
			if (!dsp)
				return false;

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[property_pool_t::PropertyCount];
			static_assert(TransformTree::RenderDescriptorSetBindingCount<=property_pool_t::PropertyCount);
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				writes[i].binding = i;
				writes[i].descriptorType = asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER;
				writes[i].count = 1u;
			}
			auto poolLayout = createPoolDescriptorSetLayout<TransformTree>(device);
			auto renderLayout = createRenderDescriptorSetLayout<TransformTree>(device);
			if (!poolLayout || !renderLayout)
				return false;

			outPoolDS = device->createGPUDescriptorSet(dsp.get(),std::move(poolLayout));
			outRenderDS = device->createGPUDescriptorSet(dsp.get(),std::move(renderLayout));
			if (!outPoolDS || !outRenderDS)
				return false;

			using DescriptorInfo = video::IGPUDescriptorSet::SDescriptorInfo;
			DescriptorInfo infos[property_pool_t::PropertyCount];
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				writes[i].dstSet = outPoolDS.get();
				writes[i].arrayElement = 0u;
				writes[i].info = infos+i;

				infos[i] = DescriptorInfo(outPool->getPropertyMemoryBlock(i));
			}
			device->updateDescriptorSets(property_pool_t::PropertyCount,writes,0u,nullptr);
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
				writes[i].dstSet = outRenderDS.get();
			infos[0] = DescriptorInfo(outPool->getPropertyMemoryBlock(global_transform_prop_ix));
			if (TransformTree::HasNormalMatrices)
				infos[1] = DescriptorInfo(outPool->getPropertyMemoryBlock(TransformTree::normal_matrix_prop_ix));
			device->updateDescriptorSets(TransformTree::RenderDescriptorSetBindingCount,writes,0u,nullptr);
			return true;
		}
#endif
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

