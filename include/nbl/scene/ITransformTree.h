// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_TRANSFORM_TREE_H_INCLUDED__
#define __NBL_SCENE_I_TRANSFORM_TREE_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/video/declarations.h"
#include "nbl/video/definitions.h"

namespace nbl::scene
{

class ITransformTree : public virtual core::IReferenceCounted
{
	public:
		using node_t = uint32_t;
		static inline constexpr node_t invalid_node = video::IPropertyPool::invalid;

		using timestamp_t = video::IGPUAnimationLibrary::timestamp_t;
		// two timestamp values are reserved for initialization
		static inline constexpr timestamp_t min_timestamp = 0u;
		static inline constexpr timestamp_t max_timestamp = 0xfffffffdu;
		static inline constexpr timestamp_t initial_modified_timestamp = 0xffffffffu;
		static inline constexpr timestamp_t initial_recomputed_timestamp = 0xfffffffeu;
		
		using parent_t = node_t;
		using relative_transform_t = core::matrix3x4SIMD;
		using modified_stamp_t = timestamp_t;
		using global_transform_t = core::matrix3x4SIMD;
		using recomputed_stamp_t = timestamp_t;

		using property_pool_t = video::CPropertyPool<core::allocator,
			parent_t,
			relative_transform_t,modified_stamp_t,
			global_transform_t,recomputed_stamp_t
		>;
		static inline constexpr uint32_t parent_prop_ix = 0u;
		static inline constexpr uint32_t relative_transform_prop_ix = 1u;
		static inline constexpr uint32_t modified_stamp_prop_ix = 2u;
		static inline constexpr uint32_t global_transform_prop_ix = 3u;
		static inline constexpr uint32_t recomputed_stamp_prop_ix = 4u;

		// useful for everyone
		template<typename BindingType>
		static inline void fillDescriptorLayoutBindings(BindingType* bindings, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = stageAccessFlags ? stageAccessFlags[i]:asset::IShader::ESS_ALL;
				bindings[i].samplers = nullptr;
			}
		}
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			asset::ICPUDescriptorSetLayout::SBinding bindings[property_pool_t::PropertyCount];
			fillDescriptorLayoutBindings(bindings,stageAccessFlags);
			return core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+property_pool_t::PropertyCount);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[property_pool_t::PropertyCount];
			fillDescriptorLayoutBindings(bindings,stageAccessFlags);
			return device->createGPUDescriptorSetLayout(bindings,bindings+property_pool_t::PropertyCount);
		}

		// the creation is the same as that of a `video::CPropertyPool`
		template<typename... Args>
		static inline core::smart_refctd_ptr<ITransformTree> create(video::ILogicalDevice* device, Args... args)
		{
			auto pool = property_pool_t::create(device,std::forward<Args>(args)...);
			if (!pool)
				return nullptr;

			video::IDescriptorPool::SDescriptorPoolSize size = {asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,property_pool_t::PropertyCount};
			auto dsp = device->createDescriptorPool(video::IDescriptorPool::ECF_NONE,1u,1u,&size);
			if (!dsp)
				return nullptr;

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[property_pool_t::PropertyCount];
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				writes[i].binding = i;
				writes[i].descriptorType = asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER;
				writes[i].count = 1u;
			}
			auto layout = createDescriptorSetLayout(device);
			if (!layout)
				return nullptr;

			auto ds = device->createGPUDescriptorSet(dsp.get(),std::move(layout));
			if (!ds)
				return nullptr;

			video::IGPUDescriptorSet::SDescriptorInfo infos[property_pool_t::PropertyCount];
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				writes[i].dstSet = ds.get();
				writes[i].arrayElement = 0u;
				writes[i].info = infos+i;

				const auto& block = pool->getPropertyMemoryBlock(i);
				infos[i].desc = block.buffer;
				infos[i].buffer.offset = block.offset;
				infos[i].buffer.size = block.size;
			}
			device->updateDescriptorSets(property_pool_t::PropertyCount,writes,0u,nullptr);

			auto* tt = new ITransformTree(std::move(pool),std::move(ds));
			return core::smart_refctd_ptr<ITransformTree>(tt,core::dont_grab);
		}
		
		//
		inline const auto* getNodePropertyPool() const {return m_nodeStorage.get();}

		//
		inline const auto* getNodePropertyDescriptorSet() const {return m_transformHierarchyDS.get();}

		//
		inline const asset::SBufferRange<video::IGPUBuffer>& getGlobalTransformationBufferRange() const
		{
			return m_nodeStorage->getPropertyMemoryBlock(global_transform_prop_ix);
		}

		// nodes array must be initialized with invalid_node
		inline bool allocateNodes(const core::SRange<ITransformTree::node_t>& outNodes)
		{
			if (outNodes.size()>m_nodeStorage->getFree())
				return false;

			return m_nodeStorage->allocateProperties(outNodes.begin(),outNodes.end());
		}

		// This removes all nodes in the hierarchy, if you want to remove individual nodes, use `ITransformTreeManager::removeNodes`
		inline void clearNodes()
		{
			m_nodeStorage->freeAllProperties();
		}

		//
		[[nodiscard]] inline bool copyGlobalTransforms(
			video::CPropertyPoolHandler* pphandler, video::StreamingTransientDataBufferMT<>* const upIndexBuff, video::IGPUBuffer* dest, const uint64_t destOffset,
			video::IGPUCommandBuffer* const cmdbuf, video::IGPUFence* const fence, const node_t* const nodesBegin, const node_t* const nodesEnd, system::logger_opt_ptr logger,
			const std::chrono::high_resolution_clock::time_point maxWaitPoint=std::chrono::high_resolution_clock::now()+std::chrono::microseconds(500u))
		{
			video::CPropertyPoolHandler::TransferRequest request;
			request.setFromPool(m_nodeStorage.get(),global_transform_prop_ix);
			request.flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			request.elementCount = nodesEnd - nodesBegin;
			request.srcAddresses = nodesBegin;
			request.dstAddresses = nullptr;
			request.buffer = dest;
			request.offset = destOffset;

			return pphandler->transferProperties(upIndexBuff, nullptr, cmdbuf, fence, &request, &request + 1u, logger, maxWaitPoint).transferSuccess;
		}

		//
		[[nodiscard]] inline auto downloadGlobalTransforms(
			video::CPropertyPoolHandler* pphandler, video::StreamingTransientDataBufferMT<>* const upIndexBuff, video::StreamingTransientDataBufferMT<>* const downBuff,
			video::IGPUCommandBuffer* const cmdbuf, video::IGPUFence* const fence, const node_t* const nodesBegin, const node_t* const nodesEnd, system::logger_opt_ptr logger,
			const std::chrono::high_resolution_clock::time_point maxWaitPoint=std::chrono::high_resolution_clock::now()+std::chrono::microseconds(500u))
		{
			video::CPropertyPoolHandler::TransferRequest request;
			request.setFromPool(m_nodeStorage.get(),global_transform_prop_ix);
			request.flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			request.elementCount = nodesEnd-nodesBegin;
			request.srcAddresses = nodesBegin;
			request.dstAddresses = nullptr;
			request.device2device = false;
			request.source = nullptr;
			return pphandler->transferProperties(upIndexBuff,downBuff,cmdbuf,fence,&request,&request+1u,logger,maxWaitPoint);
		}

	protected:
		ITransformTree(core::smart_refctd_ptr<property_pool_t>&& _nodeStorage, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS)
			: m_nodeStorage(std::move(_nodeStorage)), m_transformHierarchyDS(std::move(_transformHierarchyDS))
		{
			m_nodeStorage->getPropertyMemoryBlock(parent_prop_ix).buffer->setObjectDebugName("ITransformTree::parent_t");
			m_nodeStorage->getPropertyMemoryBlock(relative_transform_prop_ix).buffer->setObjectDebugName("ITransformTree::relative_transform_t");
			m_nodeStorage->getPropertyMemoryBlock(modified_stamp_prop_ix).buffer->setObjectDebugName("ITransformTree::modified_stamp_t");
			m_nodeStorage->getPropertyMemoryBlock(global_transform_prop_ix).buffer->setObjectDebugName("ITransformTree::global_transform_t");
			m_nodeStorage->getPropertyMemoryBlock(recomputed_stamp_prop_ix).buffer->setObjectDebugName("ITransformTree::recomputed_stamp_t");
		}
		~ITransformTree()
		{
			// everything drops itself automatically
		}

		friend class ITransformTreeManager;
		//
		inline auto* getNodePropertyPool() { return m_nodeStorage.get(); }

		core::smart_refctd_ptr<property_pool_t> m_nodeStorage;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_transformHierarchyDS;
		// TODO: do we keep a contiguous `node_t` array in-case we want to shortcut to full tree reevaluation when the number of relative transform modification requests > totalNodes*ratio (or overflows the temporary buffer we've provided) ?
};

} // end namespace nbl::scene

#endif

