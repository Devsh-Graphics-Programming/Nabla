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
		static inline core::smart_refctd_ptr<ITransformTree> create(video::ILogicalDevice* device, bool debugDraw, Args... args)
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

			auto* ttRaw = new ITransformTree(std::move(pool),std::move(ds), debugDraw);
			auto transformTree = core::smart_refctd_ptr<ITransformTree>(ttRaw, core::dont_grab);

			if(transformTree->m_debugEnabled)
			{
				auto system = device->getPhysicalDevice()->getSystem();

				auto createShader = [&system, &device](auto uniqueString, asset::ISpecializedShader::E_SHADER_STAGE type) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
				{
					auto glsl = system->loadBuiltinData<decltype(uniqueString)>();
					auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl), asset::IShader::buffer_contains_glsl_t{});
					return device->createGPUSpecializedShader(cpushader.get(), {nullptr, nullptr, "main", type});
				};

				auto gpuDebugVertexShader = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/transform_tree/debug/debugDrawNodeLine.vert")());
				auto gpuDebugFragmentShader = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/transform_tree/debug/debugDraw.frag")());

				if (!gpuDebugVertexShader && !gpuDebugFragmentShader)
					return nullptr;

				video::IGPUSpecializedShader* gpuShaders[] = {gpuDebugVertexShader.get(), gpuDebugFragmentShader.get()};

				asset::SVertexInputParams vertexInputParams;
				vertexInputParams.bindings[DEBUG_GLOBAL_NODE_ID_BINDING].inputRate = asset::EVIR_PER_INSTANCE;
				vertexInputParams.bindings[DEBUG_GLOBAL_NODE_ID_BINDING].stride = sizeof(uint32_t);
				vertexInputParams.attributes[DEBUG_GLOBAL_NODE_ID_ATTRIBUTE].binding = DEBUG_GLOBAL_NODE_ID_BINDING;
				vertexInputParams.attributes[DEBUG_GLOBAL_NODE_ID_ATTRIBUTE].format = asset::EF_R32_UINT;
				vertexInputParams.attributes[DEBUG_GLOBAL_NODE_ID_ATTRIBUTE].relativeOffset = 0u;

				vertexInputParams.enabledBindingFlags |= 0x1u << GLOBAL_NODE_ID_BINDING;
				vertexInputParams.enabledAttribFlags |= 0x1u << GLOBAL_NODE_ID_ATTRIBUTE;

				asset::SBlendParams blendParams;
				asset::SPrimitiveAssemblyParams primitiveAssemblyParams;
				primitiveAssemblyParams.primitiveType = asset::EPT_LINE_LIST;
				asset::SRasterizationParams rasterizationParams;

				core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout;
				{
					video::IGPUDescriptorSetLayout::SBinding sBindings[2];
					sBindings[0].binding = parent_prop_ix;
					sBindings[0].count = 1u;
					sBindings[0].type = asset::EDT_STORAGE_BUFFER;
					sBindings[0].stageFlags = video::IGPUSpecializedShader::ESS_VERTEX;
					sBindings[0].samplers = nullptr;

					sBindings[1] = sBindings[0];
					sBindings[1].binding = global_transform_prop_ix;
					descriptorSetLayout = device->createGPUDescriptorSetLayout(sBindings, sBindings + 2);
				}

				constexpr size_t DEBUG_LINE_PS_SIZE = sizeof(core::matrix4SIMD) + sizeof(core::vector4df_SIMD);
				asset::SPushConstantRange pcRange;
				pcRange.offset = 0u;
				pcRange.size = DEBUG_LINE_PS_SIZE;
				pcRange.stageFlags = asset::ISpecializedShader::ESS_VERTEX;

				auto gpuPipelineLayout = device->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(descriptorSetLayout));
				auto gpuRenderpassIndependentPipeline = device->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), gpuShaders, gpuShaders + 2, vertexInputParams, blendParams, primitiveAssemblyParams, rasterizationParams);
				
				if (!gpuRenderpassIndependentPipeline)
					return nullptr;

				transformTree->m_debugGpuRenderpassIndependentPipelineNodeLines = std::move(gpuRenderpassIndependentPipeline);
				{
					video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
					attachments[0].initialLayout = asset::EIL_UNDEFINED;
					attachments[0].finalLayout = asset::EIL_UNDEFINED;
					attachments[0].format = asset::EF_R8G8B8A8_SRGB;
					attachments[0].samples = asset::IImage::ESCF_1_BIT;
					attachments[0].loadOp = video::IGPURenderpass::ELO_CLEAR;
					attachments[0].storeOp = video::IGPURenderpass::ESO_STORE;

					attachments[1].initialLayout = asset::EIL_UNDEFINED;
					attachments[1].finalLayout = asset::EIL_UNDEFINED;
					attachments[1].format = asset::EF_D32_SFLOAT;
					attachments[1].samples = asset::IImage::ESCF_1_BIT;
					attachments[1].loadOp = video::IGPURenderpass::ELO_CLEAR;
					attachments[1].storeOp = video::IGPURenderpass::ESO_STORE;

					video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
					colorAttRef.attachment = 0u;
					colorAttRef.layout = asset::EIL_UNDEFINED;

					video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
					depthStencilAttRef.attachment = 1u;
					depthStencilAttRef.layout = asset::EIL_UNDEFINED;

					video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
					sp.colorAttachmentCount = 1u;
					sp.colorAttachments = &colorAttRef;
					sp.depthStencilAttachment = &depthStencilAttRef;
				
					sp.flags = video::IGPURenderpass::ESDF_NONE;
					sp.inputAttachmentCount = 0u;
					sp.inputAttachments = nullptr;
					sp.preserveAttachmentCount = 0u;
					sp.preserveAttachments = nullptr;
					sp.resolveAttachments = nullptr;

					video::IGPURenderpass::SCreationParams rp_params;
					rp_params.attachmentCount = 2u;
					rp_params.attachments = attachments;
					rp_params.dependencies = nullptr;
					rp_params.dependencyCount = 0u;
					rp_params.subpasses = &sp;
					rp_params.subpassCount = 1u;

					auto debugRenderpass = device->createGPURenderpass(rp_params);
					transformTree->m_debugGpuRenderpass = std::move(debugRenderpass);
				}

				core::smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
				{
					nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
					graphicsPipelineParams.renderpassIndependent = transformTree->m_debugGpuRenderpassIndependentPipelineNodeLines;
					graphicsPipelineParams.renderpass = transformTree->m_debugGpuRenderpass; // I'm not sure if I should go with second debug renderpass!

					auto gpuGraphicsPipeline = device->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
					transformTree->m_debugGpuPipelineNodeLines = std::move(gpuGraphicsPipeline);
				}
			}

			return transformTree;
		}

		inline void setDebugLiveAllocations(const core::unordered_set<node_t>&& nodes)
		{
			m_debugLiveAllocations = nodes;
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

		struct DebugPushConstants
		{
			core::matrix4SIMD viewProjectionMatrix;
			core::vector4df_SIMD color;
		};

		void debugDraw(video::ILogicalDevice* device, video::IGPUCommandBuffer* commandBuffer, const DebugPushConstants& debugPushConstants)
		{
			if (!m_debugEnabled)
				return;

			if (m_debugLiveAllocationsGpuBuffer)
				if (m_debugLiveAllocationsGpuBuffer->getSize() == 0 || m_debugLiveAllocationsGpuBuffer->getSize() != m_debugLiveAllocations.size() * sizeof(node_t))
					return;
			else
			{
				auto localGPUMemoryReqs = device->getDeviceLocalGPUMemoryReqs();
				localGPUMemoryReqs.vulkanReqs.size = m_debugLiveAllocations.size() * sizeof(node_t);
				localGPUMemoryReqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
				m_debugLiveAllocationsGpuBuffer = std::move(device->createGPUBufferOnDedMem(localGPUMemoryReqs, true));

				std::vector<node_t> debugLiveAllocationsData(m_debugLiveAllocations.size());
				std::copy_n(m_debugLiveAllocations.begin(), m_debugLiveAllocations.end(), debugLiveAllocationsData.data());

				commandBuffer->updateBuffer(m_debugLiveAllocationsGpuBuffer.get(), 0, m_debugLiveAllocationsGpuBuffer->getSize(), debugLiveAllocationsData.data());
			}

			//if (needFlush) what about this one?
			//	device->flushRanges();

			_NBL_STATIC_INLINE_CONSTEXPR auto VERTEX_COUNT = 2;
			const size_t INSTANCE_COUNT = m_debugLiveAllocations.size();

			const nbl::video::IGPUBuffer* gpuBufferBindings[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
			{
				for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
					gpuBufferBindings[i] = nullptr;

				gpuBufferBindings[DEBUG_GLOBAL_NODE_ID_BINDING] = m_debugLiveAllocationsGpuBuffer.get();
			}

			size_t bufferBindingsOffsets[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
			{
				for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
					bufferBindingsOffsets[i] = 0;
			}

			commandBuffer->bindGraphicsPipeline(m_debugGpuPipelineNodeLines.get());
			commandBuffer->pushConstants(m_debugGpuRenderpassIndependentPipelineNodeLines->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(debugPushConstants), &debugPushConstants);
			commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, m_debugGpuRenderpassIndependentPipelineNodeLines->getLayout(), 0u, 1u, &m_transformHierarchyDS.get());

			commandBuffer->bindVertexBuffers(0, nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT, gpuBufferBindings, bufferBindingsOffsets);
			commandBuffer->bindIndexBuffer(nullptr, 0, nbl::asset::EIT_UNKNOWN);

			commandBuffer->draw(VERTEX_COUNT, INSTANCE_COUNT, 0, 0);
		}

	protected:
		ITransformTree(core::smart_refctd_ptr<property_pool_t>&& _nodeStorage, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS, bool _debugDraw)
			: m_nodeStorage(std::move(_nodeStorage)), m_transformHierarchyDS(std::move(_transformHierarchyDS)), m_debugEnabled(_debugDraw)
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
		bool m_debugEnabled;

		_NBL_STATIC_INLINE_CONSTEXPR size_t DEBUG_GLOBAL_NODE_ID_BINDING = 15u;
		_NBL_STATIC_INLINE_CONSTEXPR size_t DEBUG_GLOBAL_NODE_ID_ATTRIBUTE = 0u;

		core::unordered_set<node_t> m_debugLiveAllocations;
		core::smart_refctd_ptr<video::IGPUBuffer> m_debugLiveAllocationsGpuBuffer;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_debugGpuPipelineNodeLines;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> m_debugGpuRenderpassIndependentPipelineNodeLines; // only lines at the moment
		core::smart_refctd_ptr<video::IGPURenderpass> m_debugGpuRenderpass;
};

} // end namespace nbl::scene

#endif

