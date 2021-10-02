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
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDescriptorSetLayout(video::ILogicalDevice* device, asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[property_pool_t::PropertyCount];
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = stageAccessFlags ? stageAccessFlags[i]:asset::ISpecializedShader::ESS_ALL;
				bindings[i].samplers = nullptr;
			}
			return device->createGPUDescriptorSetLayout(bindings,bindings+property_pool_t::PropertyCount);
		}

		// the creation is the same as that of a `video::CPropertyPool`
		template<typename... Args>
		static inline core::smart_refctd_ptr<ITransformTree> create(video::ILogicalDevice* device, core::smart_refctd_ptr<video::IGPURenderpass> gpuRenderpass, Args... args)
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

			auto ds = device->createGPUDescriptorSet(dsp.get(), core::smart_refctd_ptr(layout));
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

			auto* ttRaw = new ITransformTree(std::move(pool),std::move(ds));
			auto transformTree = core::smart_refctd_ptr<ITransformTree>(ttRaw, core::dont_grab);
			{
				auto system = device->getPhysicalDevice()->getSystem();

				auto createShader = [&system, &device](auto uniqueString, asset::ISpecializedShader::E_SHADER_STAGE type) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
				{
					auto glsl = system->loadBuiltinData<decltype(uniqueString)>();
					auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl), asset::IShader::buffer_contains_glsl_t{});
					auto gpuShader = device->createGPUShader(std::move(cpuShader));

					return device->createGPUSpecializedShader(gpuShader.get(), {nullptr, nullptr, "main", type});
				};

				constexpr uint16_t SHADER_COUNT = 2u;
				auto gpuDebugVertexShader = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/transform_tree/debug/debug_draw_node_line.vert")(), asset::ISpecializedShader::ESS_VERTEX);
				auto gpuDebugFragmentShader = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/transform_tree/debug/debug_draw.frag")(), asset::ISpecializedShader::ESS_FRAGMENT);

				if (!gpuDebugVertexShader || !gpuDebugFragmentShader)
					return nullptr;

				video::IGPUSpecializedShader* gpuShaders[] = {gpuDebugVertexShader.get(), gpuDebugFragmentShader.get()};

				asset::SVertexInputParams vertexInputParams;
				vertexInputParams.bindings[DEBUG_GLOBAL_NODE_ID_AND_SCALE_BINDING].inputRate = asset::EVIR_PER_INSTANCE;
				vertexInputParams.bindings[DEBUG_GLOBAL_NODE_ID_AND_SCALE_BINDING].stride = sizeof(DebugNodeVtxInput);

				vertexInputParams.attributes[DEBUG_GLOBAL_NODE_ID_ATTRIBUTE].binding = DEBUG_GLOBAL_NODE_ID_AND_SCALE_BINDING;
				vertexInputParams.attributes[DEBUG_GLOBAL_NODE_ID_ATTRIBUTE].format = asset::EF_R32_UINT;
				vertexInputParams.attributes[DEBUG_GLOBAL_NODE_ID_ATTRIBUTE].relativeOffset = offsetof(DebugNodeVtxInput, node);

				vertexInputParams.attributes[DEBUG_SCALE_NODE_ATTRIBUTE].binding = DEBUG_GLOBAL_NODE_ID_AND_SCALE_BINDING;
				vertexInputParams.attributes[DEBUG_SCALE_NODE_ATTRIBUTE].format = asset::EF_R32_SFLOAT;
				vertexInputParams.attributes[DEBUG_SCALE_NODE_ATTRIBUTE].relativeOffset = offsetof(DebugNodeVtxInput, scale);

				vertexInputParams.enabledBindingFlags |= 0x1u << DEBUG_GLOBAL_NODE_ID_AND_SCALE_BINDING;
				vertexInputParams.enabledAttribFlags |= 0x1u << DEBUG_GLOBAL_NODE_ID_ATTRIBUTE | 0x1u << DEBUG_SCALE_NODE_ATTRIBUTE;

				asset::SBlendParams blendParams;
				asset::SPrimitiveAssemblyParams primitiveAssemblyParams;
				primitiveAssemblyParams.primitiveType = asset::EPT_LINE_LIST;
				asset::SRasterizationParams rasterizationParams;

				asset::SPushConstantRange pcRange;
				pcRange.offset = 0u;
				pcRange.size = sizeof(DebugPushConstants);
				pcRange.stageFlags = static_cast<asset::ISpecializedShader::E_SHADER_STAGE>(asset::ISpecializedShader::ESS_VERTEX);

				auto gpuPipelineLayout = device->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(layout));
				auto gpuRenderpassIndependentPipeline = device->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), gpuShaders, gpuShaders + SHADER_COUNT, vertexInputParams, blendParams, primitiveAssemblyParams, rasterizationParams);
				
				if (!gpuRenderpassIndependentPipeline)
					return nullptr;

				transformTree->m_debugGpuRenderpassIndependentPipelineNode = std::move(gpuRenderpassIndependentPipeline);
				transformTree->m_debugGpuRenderpass = core::smart_refctd_ptr(gpuRenderpass);
				
				core::smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
				{
					nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
					graphicsPipelineParams.renderpassIndependent = transformTree->m_debugGpuRenderpassIndependentPipelineNode;
					graphicsPipelineParams.renderpass = core::smart_refctd_ptr(transformTree->m_debugGpuRenderpass);

					auto gpuGraphicsPipeline = device->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
					transformTree->m_debugGpuPipelineNode = std::move(gpuGraphicsPipeline);
				}
			}

			return transformTree;
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

		#include "nbl/nblpack.h"
		struct DebugNodeVtxInput
		{
			node_t node;
			float scale;
		} PACK_STRUCT;
		#include "nbl/nblunpack.h"

		#include "nbl/nblpack.h"
		struct DebugPushConstants
		{
			core::matrix4SIMD viewProjectionMatrix;
			core::vector4df_SIMD lineColor;
			core::vector4df_SIMD aabbColor;
			core::vector4df_SIMD minEdge;
			core::vector4df_SIMD maxEdge;
		} PACK_STRUCT;
		#include "nbl/nblunpack.h"

		inline void setDebugEnabledFlag(bool debugEnabled = true)
		{
			m_debugEnabled = debugEnabled;
		}

		inline void setDebugLiveAllocations(const core::vector<DebugNodeVtxInput>& nodes)
		{
			m_debugLiveAllocations = nodes;
		}

		void debugDraw(video::ILogicalDevice* device, video::IGPUCommandBuffer* commandBuffer, const DebugPushConstants& debugPushConstants)
		{
			if (!m_debugEnabled)
				return;

			if (m_debugLiveAllocationsGpuBuffer)
			{
				if (!m_debugLiveAllocationsGpuBuffer->getSize() || m_debugLiveAllocationsGpuBuffer->getSize() != m_debugLiveAllocations.size() * sizeof(DebugNodeVtxInput))
					return;
			}
			else
			{
				if (!m_debugLiveAllocations.size())
					return;

				auto localGPUMemoryReqs = device->getDeviceLocalGPUMemoryReqs();
				localGPUMemoryReqs.vulkanReqs.size = m_debugLiveAllocations.size() * sizeof(DebugNodeVtxInput);
				localGPUMemoryReqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
				m_debugLiveAllocationsGpuBuffer = std::move(device->createGPUBufferOnDedMem(video::IGPUBuffer::SCreationParams{}, localGPUMemoryReqs, true));

				commandBuffer->updateBuffer(m_debugLiveAllocationsGpuBuffer.get(), 0, m_debugLiveAllocationsGpuBuffer->getSize(), m_debugLiveAllocations.data());
			}
			
			#define LINE_VERTEX_COUNT 2u
			#define BOX_VERTEX_COUNT 24u

			_NBL_STATIC_INLINE_CONSTEXPR auto VERTEX_COUNT = LINE_VERTEX_COUNT + BOX_VERTEX_COUNT;
			const size_t INSTANCE_COUNT = m_debugLiveAllocations.size();

			const nbl::video::IGPUBuffer* gpuBufferBindings[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
			{
				for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
					gpuBufferBindings[i] = nullptr;

				gpuBufferBindings[DEBUG_GLOBAL_NODE_ID_AND_SCALE_BINDING] = m_debugLiveAllocationsGpuBuffer.get();
			}

			size_t bufferBindingsOffsets[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
			{
				for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
					bufferBindingsOffsets[i] = 0;
			}

			commandBuffer->bindGraphicsPipeline(m_debugGpuPipelineNode.get());
			const auto stage = core::bitflag(asset::ISpecializedShader::ESS_VERTEX) | core::bitflag(asset::ISpecializedShader::ESS_GEOMETRY);
			commandBuffer->pushConstants(m_debugGpuRenderpassIndependentPipelineNode->getLayout(), stage, 0u, sizeof(DebugPushConstants), &debugPushConstants);
			commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, m_debugGpuRenderpassIndependentPipelineNode->getLayout(), 0u, 1u, &m_transformHierarchyDS.get());

			commandBuffer->bindVertexBuffers(0, nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT, gpuBufferBindings, bufferBindingsOffsets);
			commandBuffer->bindIndexBuffer(nullptr, 0, nbl::asset::EIT_UNKNOWN);

			commandBuffer->draw(VERTEX_COUNT, INSTANCE_COUNT, 0, 0);
		}

	protected:
		ITransformTree(core::smart_refctd_ptr<property_pool_t>&& _nodeStorage, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS)
			: m_nodeStorage(std::move(_nodeStorage)), m_transformHierarchyDS(std::move(_transformHierarchyDS))
		{
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
	private:
		bool m_debugEnabled = false;

		_NBL_STATIC_INLINE_CONSTEXPR size_t DEBUG_GLOBAL_NODE_ID_AND_SCALE_BINDING = 15u;
		_NBL_STATIC_INLINE_CONSTEXPR size_t DEBUG_GLOBAL_NODE_ID_ATTRIBUTE = 0u;
		_NBL_STATIC_INLINE_CONSTEXPR size_t DEBUG_SCALE_NODE_ATTRIBUTE = 1u;

		core::vector<DebugNodeVtxInput> m_debugLiveAllocations;
		core::smart_refctd_ptr<video::IGPUBuffer> m_debugLiveAllocationsGpuBuffer;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_debugGpuPipelineNode;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> m_debugGpuRenderpassIndependentPipelineNode;
		core::smart_refctd_ptr<video::IGPURenderpass> m_debugGpuRenderpass;
};

} // end namespace nbl::scene

#endif

