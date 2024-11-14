// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_CULLING_LOD_SELECTION_SYSTEM_H_INCLUDED__
#define __NBL_SCENE_I_CULLING_LOD_SELECTION_SYSTEM_H_INCLUDED__

#include "nbl/video/utilities/CScanner.h"

#include "nbl/scene/ITransformTree.h"
#include "nbl/scene/ILevelOfDetailLibrary.h"

namespace nbl::scene
{
# if 0 // REDO
class ICullingLoDSelectionSystem : public virtual core::IReferenceCounted
{
	public:
		static void enableRequiredFeautres(video::SPhysicalDeviceFeatures& featuresToEnable)
		{
		}

		static void enablePreferredFeatures(const video::SPhysicalDeviceFeatures& availableFeatures, video::SPhysicalDeviceFeatures& featuresToEnable)
		{
		}

		//
		#define nbl_glsl_DispatchIndirectCommand_t asset::DispatchIndirectCommand_t
		#define uint uint32_t
		#include "nbl/builtin/glsl/culling_lod_selection/dispatch_indirect_params.glsl"
		#undef uint
		#undef nbl_glsl_DispatchIndirectCommand_t
		struct DispatchIndirectParams : nbl_glsl_culling_lod_selection_dispatch_indirect_params_t
		{
		};
		struct InstanceToCull
		{
			uint32_t instanceGUID;
			uint32_t lodTableUvec4Offset;
		};

		//
		static core::smart_refctd_ptr<video::IGPUBuffer> createDispatchIndirectBuffer(video::IUtilities* utils, video::IQueue* queue)
		{
			DispatchIndirectParams contents;
			auto setWorkgroups = [](asset::DispatchIndirectCommand_t& cmd)
			{
				cmd.num_groups_x = 1u;
				cmd.num_groups_y = 1u;
				cmd.num_groups_z = 1u;
			};
			setWorkgroups(contents.instanceCullAndLoDSelect);
			setWorkgroups(contents.instanceDrawCountPrefixSum);
			setWorkgroups(contents.instanceDrawCull);
			setWorkgroups(contents.instanceRefCountingSortScatter);
			setWorkgroups(contents.drawCompact);

			video::IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(contents);
			params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT)|asset::IBuffer::EUF_INDIRECT_BUFFER_BIT|asset::IBuffer::EUF_TRANSFER_DST_BIT;
			return utils->createFilledDeviceLocalBufferOnDedMem(queue,std::move(params),&contents);
		}

		// These buffer ranges can be safely discarded or reused after `processInstancesAndFillIndirectDraws` completes
		struct ScratchBufferRanges
		{
			asset::SBufferRange<video::IGPUBuffer> pvsInstances;
			asset::SBufferRange<video::IGPUBuffer> lodDrawCallCounts; // must clear first DWORD to 0
			asset::SBufferRange<video::IGPUBuffer> pvsInstanceDraws;
			asset::SBufferRange<video::IGPUBuffer> prefixSumScratch;  // clear a certain range to 0
		};
		static ScratchBufferRanges createScratchBuffer(video::CScanner* scanner, const uint32_t maxTotalInstances, const uint32_t maxTotalVisibleDrawcallInstances)
		{
			auto logicalDevice = scanner->getDevice();

			ScratchBufferRanges retval;
			{
				const auto& limits = logicalDevice->getPhysicalDevice()->getLimits();
				const auto ssboAlignment = limits.minSSBOAlignment;

				retval.pvsInstances.offset = 0u;
				retval.lodDrawCallCounts.offset = retval.pvsInstances.size = core::alignUp(maxTotalInstances*2u*sizeof(uint32_t),ssboAlignment);
				retval.lodDrawCallCounts.size = core::alignUp((maxTotalInstances+1u)*sizeof(uint32_t),ssboAlignment);
				retval.pvsInstanceDraws.offset = retval.lodDrawCallCounts.offset+retval.lodDrawCallCounts.size;
				retval.pvsInstanceDraws.size = core::alignUp(sizeof(uint32_t)+maxTotalVisibleDrawcallInstances*sizeof(PotentiallyVisisbleInstanceDraw),ssboAlignment);
				retval.prefixSumScratch.offset = retval.pvsInstanceDraws.offset+retval.pvsInstanceDraws.size;
				{
					video::CScanner::Parameters params;
					auto schedulerParams = video::CScanner::SchedulerParameters(params,maxTotalInstances,scanner->getWorkgroupSize());
					retval.prefixSumScratch.size = params.getScratchSize(ssboAlignment);
				}
			}
			{
				video::IGPUBuffer::SCreationParams params = {};
				params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
				params.size = retval.prefixSumScratch.offset+retval.prefixSumScratch.size;
				auto gpubuffer = logicalDevice->createBuffer(std::move(params));
				retval.pvsInstances.buffer =
				retval.lodDrawCallCounts.buffer =
				retval.pvsInstanceDraws.buffer =
				retval.prefixSumScratch.buffer = gpubuffer;
				auto mreqs = gpubuffer->getMemoryReqs();
				mreqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto gpubufMem = logicalDevice->allocate(mreqs, gpubuffer.get());

				retval.pvsInstances.buffer->setObjectDebugName("Culling Scratch Buffer");
			}
			return retval;
		}
		// two buffers out of the scratch need to be pre-cleared to 0 at their starts to ensure atomics start counting from 0
		// you only need to call this on startup, the culling system cleans up after itself (resets to 0)
		inline void clearScratch(
			video::IGPUCommandBuffer* cmdbuf,
			const asset::SBufferBinding<video::IGPUBuffer>& lodDrawCallCounts,
			const asset::SBufferRange<video::IGPUBuffer>& prefixSumScratch
		)
		{
			cmdbuf->fillBuffer({lodDrawCallCounts.offset,sizeof(uint32_t),lodDrawCallCounts.buffer},0u);
			// if we allow for more than 2^31 theoretical pool allocator entries AND SSBOs bigger than 2GB we might have to change this logic
			static_assert(video::CScanner::Parameters::MaxScanLevels<=7u,"Max Scan Scheduling Hierarchy Tree Height has increased, logic needs update");
			const uint32_t schedulerSizeBound = core::min(sizeof(uint32_t)<<20u,prefixSumScratch.size);
			cmdbuf->fillBuffer({prefixSumScratch.offset,schedulerSizeBound,prefixSumScratch.buffer},0u);
		}

		// Per-View Per-Instance buffer should hold at least an MVP matrix
		template<typename PerViewPerInstanceDataType>
		static core::smart_refctd_ptr<video::IGPUBuffer> createPerViewPerInstanceDataBuffer(video::ILogicalDevice* logicalDevice, const uint32_t maxTotalInstances)
		{
			return createPerViewPerInstanceDataBuffer(logicalDevice,maxTotalInstances,sizeof(PerViewPerInstanceDataType));
		}
		static core::smart_refctd_ptr<video::IGPUBuffer> createPerViewPerInstanceDataBuffer(video::ILogicalDevice* logicalDevice, const uint32_t maxTotalInstances, const uint32_t perViewPerInstanceDataSize)
		{
			video::IGPUBuffer::SCreationParams params = {};
			params.size = perViewPerInstanceDataSize*maxTotalInstances;
			params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
			auto buffer = logicalDevice->createBuffer(std::move(params));
			auto mreqs = buffer->getMemoryReqs();
			mreqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpubufMem = logicalDevice->allocate(mreqs, buffer.get());
			return buffer;
		}

		// Instance Redirect buffer holds a `uvec2` of `{instanceGUID,perViewPerInstanceDataID}` for each instace of a drawcall
		static core::smart_refctd_ptr<video::IGPUBuffer> createInstanceRedirectBuffer(video::ILogicalDevice* logicalDevice, const uint32_t maxTotalVisibleDrawcallInstances)
		{
			video::IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(uint32_t)*2u*maxTotalVisibleDrawcallInstances;
			params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT)|asset::IBuffer::EUF_VERTEX_BUFFER_BIT;
			auto buffer = logicalDevice->createBuffer(std::move(params));
			auto mreqs = buffer->getMemoryReqs();
			mreqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpubufMem = logicalDevice->allocate(mreqs, buffer.get());
			return buffer;
		}


		//
		static inline constexpr auto InputDescriptorBindingCount = 8u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createInputDescriptorSetLayout(video::ILogicalDevice* device, bool withMDICounts=false)
		{
			withMDICounts &= device->getPhysicalDevice()->getLimits().drawIndirectCount;

			video::IGPUDescriptorSetLayout::SBinding bindings[InputDescriptorBindingCount];
			for (auto i=0u; i<InputDescriptorBindingCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
				bindings[i].samplers = nullptr;
			}

			uint32_t count = InputDescriptorBindingCount;
			if (!withMDICounts)
				count--;
			return device->createDescriptorSetLayout(bindings,bindings+InputDescriptorBindingCount);
		}
		//
		static inline constexpr auto OutputDescriptorBindingCount = 4u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createOutputDescriptorSetLayout(video::ILogicalDevice* device, bool withMDICounts=false)
		{
			withMDICounts &= device->getPhysicalDevice()->getLimits().drawIndirectCount;

			video::IGPUDescriptorSetLayout::SBinding bindings[OutputDescriptorBindingCount];
			for (auto i=0u; i<OutputDescriptorBindingCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
				bindings[i].samplers = nullptr;
			}

			uint32_t count = OutputDescriptorBindingCount;
			if (!withMDICounts)
				count--;
			return device->createDescriptorSetLayout(bindings,bindings+count);
		}


		//
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSet> createInputDescriptorSet(
			video::ILogicalDevice* device, video::IDescriptorPool* pool,
			core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>&& layout,
			const asset::SBufferBinding<video::IGPUBuffer>& dispatchIndirect,
			const asset::SBufferRange<video::IGPUBuffer>& instanceList,
			const ScratchBufferRanges& scratchBufferRanges,
			const asset::SBufferRange<video::IGPUBuffer>& drawcallsToScan,
			const asset::SBufferRange<video::IGPUBuffer>& drawCountsToScan={}
		)
		{
			auto _layout = layout.get();
			auto ds = pool->createDescriptorSet(std::move(layout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[InputDescriptorBindingCount];
				video::IGPUDescriptorSet::SDescriptorInfo infos[InputDescriptorBindingCount] =
				{
					dispatchIndirect,
					instanceList,
					scratchBufferRanges.pvsInstances,
					scratchBufferRanges.lodDrawCallCounts,
					scratchBufferRanges.pvsInstanceDraws,
					scratchBufferRanges.prefixSumScratch,
					drawcallsToScan,
					drawCountsToScan
				};
				for (auto i=0u; i<InputDescriptorBindingCount; i++)
				{
					writes[i].dstSet = ds.get();
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
					writes[i].info = infos+i;
				}
				uint32_t count = InputDescriptorBindingCount;
				if (_layout->getTotalBindingCount()==InputDescriptorBindingCount)
				{
					assert(drawCountsToScan.buffer && drawCountsToScan.size!=0ull);
				}
				else
					count--;
				device->updateDescriptorSets(count,writes,0u,nullptr);
			}
			return ds;
		}
		//
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSet> createOutputDescriptorSet(
			video::ILogicalDevice* device, video::IDescriptorPool* pool,
			core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>&& layout,
			const asset::SBufferRange<video::IGPUBuffer>& drawCalls,
			const asset::SBufferRange<video::IGPUBuffer>& perViewPerInstance,
			const asset::SBufferRange<video::IGPUBuffer>& perInstanceRedirectAttribs,
			const asset::SBufferRange<video::IGPUBuffer>& drawCallCounts={}
		)
		{
			auto _layout = layout.get();
			auto ds = pool->createDescriptorSet(std::move(layout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[OutputDescriptorBindingCount];
				video::IGPUDescriptorSet::SDescriptorInfo infos[OutputDescriptorBindingCount] =
				{
					drawCalls,
					perViewPerInstance,
					perInstanceRedirectAttribs,
					drawCallCounts
				};
				for (auto i=0u; i<OutputDescriptorBindingCount; i++)
				{
					writes[i].dstSet = ds.get();
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
					writes[i].info = infos+i;
				}
				uint32_t count = OutputDescriptorBindingCount;
				if (_layout->getTotalBindingCount()==OutputDescriptorBindingCount)
				{
					assert(drawCallCounts.buffer && drawCallCounts.size!=0ull);
				}
				else
					count--;
				device->updateDescriptorSets(count,writes,0u,nullptr);
			}
			return ds;
		}

		//
		inline const video::IGPUPipelineLayout* getInstanceCullAndLoDSelectLayout() const
		{
			return m_instanceCullAndLoDSelectLayout.get();
		}

		//
		struct Params
		{
			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			asset::SBufferBinding<video::IGPUBuffer> indirectDispatchParams;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> lodLibraryDS;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> transientInputDS;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> transientOutputDS;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> customDS;
			// these are for the pipeline barriers
			asset::SBufferRange<video::IGPUBuffer> instanceList;
			ScratchBufferRanges scratchBufferRanges;
			asset::SBufferRange<video::IGPUBuffer> drawCalls;
			asset::SBufferRange<video::IGPUBuffer> perViewPerInstance;
			asset::SBufferRange<video::IGPUBuffer> perInstanceRedirectAttribs;
			asset::SBufferRange<video::IGPUBuffer> drawCounts = {};
			uint32_t drawcallCount : 26;
			uint32_t indirectInstanceCull : 1;
		};
		void processInstancesAndFillIndirectDraws(const Params& params, const uint32_t directInstanceCullInstanceCount=0u)
		{
			assert(false);
#if 0 // TODO: redo
			auto cmdbuf = params.cmdbuf;
			const auto queueFamilyIndex = cmdbuf->getPool()->getQueueFamilyIndex();
			const asset::SBufferRange<video::IGPUBuffer> indirectRange = {
				params.indirectDispatchParams.offset,sizeof(DispatchIndirectParams),
				params.indirectDispatchParams.buffer
			};

			constexpr auto MaxBufferBarriers = 6u;
			video::IGPUCommandBuffer::SBufferMemoryBarrier barriers[MaxBufferBarriers];
			for (auto i=0u; i<MaxBufferBarriers; i++)
			{
				barriers[i].srcQueueFamilyIndex = queueFamilyIndex;
				barriers[i].dstQueueFamilyIndex = queueFamilyIndex;
			}
			auto setBarrierBuffer = [](
				video::IGPUCommandBuffer::SBufferMemoryBarrier& barrier, const asset::SBufferRange<video::IGPUBuffer>& range,
				core::bitflag<asset::E_ACCESS_FLAGS> srcAccessMask,
				core::bitflag<asset::E_ACCESS_FLAGS> dstAccessMask//=asset::EAF_SHADER_READ_BIT
			) -> void
			{
				barrier.barrier.srcAccessMask = srcAccessMask;
				barrier.barrier.dstAccessMask = dstAccessMask;
				barrier.buffer = range.buffer;
				barrier.offset = range.offset;
				barrier.size = range.size;
			};
			const auto wAccessMask = core::bitflag(asset::EAF_SHADER_WRITE_BIT);
			const auto rwAccessMask = core::bitflag(asset::EAF_SHADER_READ_BIT)|wAccessMask;
			const auto indirectAccessMask = core::bitflag(asset::EAF_INDIRECT_COMMAND_READ_BIT)|rwAccessMask;
			{
				setBarrierBuffer(barriers[0],params.drawCalls,indirectAccessMask,wAccessMask);
				setBarrierBuffer(barriers[1],params.perViewPerInstance,asset::EAF_SHADER_READ_BIT,wAccessMask);
				setBarrierBuffer(barriers[2],params.perInstanceRedirectAttribs,asset::EAF_VERTEX_ATTRIBUTE_READ_BIT,wAccessMask);
				cmdbuf->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,asset::EDF_NONE,0u,nullptr,3u,barriers,0u,nullptr);
			}
			setBarrierBuffer(barriers[0],indirectRange,indirectAccessMask,indirectAccessMask);
			
			const auto internalStageFlags = core::bitflag(asset::PIPELINE_STAGE_FLAGS::DRAW_INDIRECT_BIT)|asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			cmdbuf->bindComputePipeline(m_instanceCullAndLoDSelect.get());
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_instanceCullAndLoDSelectLayout.get(),0u,4u,&params.lodLibraryDS.get());
			if (params.indirectInstanceCull)
				cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,instanceCullAndLoDSelect));
			else
			{
				const auto& limits = m_scanner->getDevice()->getPhysicalDevice()->getLimits();
				uint32_t wgCount;
				if (directInstanceCullInstanceCount)
					wgCount = limits.computeOptimalPersistentWorkgroupDispatchSize(directInstanceCullInstanceCount,m_workgroupSize);
				else
					wgCount = limits.maxResidentInvocations/limits.maxOptimallyResidentWorkgroupInvocations;
				cmdbuf->dispatch(wgCount,1u,1u);
			}
			{
				setBarrierBuffer(barriers[1],params.drawCalls,wAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[2],params.scratchBufferRanges.pvsInstances,wAccessMask,asset::EAF_SHADER_READ_BIT);
				setBarrierBuffer(barriers[3],params.scratchBufferRanges.pvsInstanceDraws,rwAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[4],params.scratchBufferRanges.lodDrawCallCounts,rwAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[5],params.perViewPerInstance,wAccessMask,asset::EAF_SHADER_READ_BIT);
				cmdbuf->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,internalStageFlags,asset::EDF_NONE,0u,nullptr, MaxBufferBarriers,barriers,0u,nullptr);
			}

			cmdbuf->bindComputePipeline(m_instanceDrawCountPrefixSum.get());
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_instanceDrawCullLayout.get(),0u,3u,&params.lodLibraryDS.get());
			cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,instanceDrawCountPrefixSum));
			{
				setBarrierBuffer(barriers[1],params.scratchBufferRanges.lodDrawCallCounts,rwAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[2],params.scratchBufferRanges.prefixSumScratch,rwAccessMask,rwAccessMask);
				cmdbuf->pipelineBarrier(internalStageFlags,internalStageFlags,asset::EDF_NONE,0u,nullptr,3u,barriers,0u,nullptr);
			}

			cmdbuf->bindComputePipeline(m_instanceDrawCull.get());
			{
				const auto maxTotalVisibleDrawcallInstances = (params.scratchBufferRanges.pvsInstanceDraws.size-sizeof(uint32_t))/sizeof(PotentiallyVisisbleInstanceDraw);
				video::CScanner::Parameters scanParams;
				auto schedulerParams = video::CScanner::SchedulerParameters(scanParams,maxTotalVisibleDrawcallInstances,m_scanner->getWorkgroupSize());
				cmdbuf->pushConstants(m_instanceDrawCullLayout.get(),asset::IShader::ESS_COMPUTE,0u,sizeof(uint32_t),scanParams.temporaryStorageOffset);
			}
			cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,instanceDrawCull));
			{
				setBarrierBuffer(barriers[1],params.scratchBufferRanges.prefixSumScratch,rwAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[2],params.drawCalls,rwAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[3],params.scratchBufferRanges.pvsInstanceDraws,rwAccessMask,rwAccessMask);
				cmdbuf->pipelineBarrier(internalStageFlags,internalStageFlags,asset::EDF_NONE,0u,nullptr,4u,barriers,0u,nullptr);
			}

			cmdbuf->bindComputePipeline(m_drawInstanceCountPrefixSum.get());
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_instanceRefCountingSortPipelineLayout.get(),1u,2u,&params.transientInputDS.get());
			{
				video::CScanner::DispatchInfo dispatchInfo;
				{
					video::CScanner::DefaultPushConstants pushConstants;
					m_scanner->buildParameters(params.drawcallCount,pushConstants,dispatchInfo);
					cmdbuf->pushConstants(m_instanceRefCountingSortPipelineLayout.get(),asset::IShader::ESS_COMPUTE,0u,sizeof(pushConstants),&pushConstants);
				}
				cmdbuf->dispatch(dispatchInfo.wg_count,1u,1u);
			}
			{
				setBarrierBuffer(barriers[1],params.drawCalls,rwAccessMask,indirectAccessMask);
				setBarrierBuffer(barriers[2],params.scratchBufferRanges.prefixSumScratch,rwAccessMask,rwAccessMask);
				cmdbuf->pipelineBarrier(internalStageFlags,internalStageFlags,asset::EDF_NONE,0u,nullptr,3u,barriers,0u,nullptr);
			}

			cmdbuf->bindComputePipeline(m_instanceRefCountingSortScatter.get());
			cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,instanceRefCountingSortScatter));
			{
				setBarrierBuffer(barriers[1],params.scratchBufferRanges.lodDrawCallCounts,wAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[2],params.scratchBufferRanges.prefixSumScratch,rwAccessMask,rwAccessMask);
				setBarrierBuffer(barriers[3],params.perInstanceRedirectAttribs,wAccessMask,asset::EAF_VERTEX_ATTRIBUTE_READ_BIT);
				cmdbuf->pipelineBarrier(internalStageFlags,internalStageFlags|asset::PIPELINE_STAGE_FLAGS::VERTEX_INPUT_BIT,asset::EDF_NONE,0u,nullptr,4u,barriers,0u,nullptr);
			}
			// drawcall compaction
			if (params.transientOutputDS->getLayout()->getTotalBindingCount()==OutputDescriptorBindingCount)
			{
				cmdbuf->bindComputePipeline(drawCompact.get());
				cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,drawCompact));
			}
#			endif
		}

		// `perViewPerInstanceDefinition` must contain the definition of a data type `nbl_glsl_PerViewPerInstance_t` 
		// and `cullAndLoDSelectFuncDefinitions` must define the 4 functions used by
		// "nbl/builtin/glsl/culling_lod_selection/instance_cull_and_lod_select.comp"
		static core::smart_refctd_ptr<ICullingLoDSelectionSystem> create(
			core::smart_refctd_ptr<video::CScanner>&& _scanner,
			const asset::SPushConstantRange* cullAndLoDSelectPCBegin, const asset::SPushConstantRange* cullAndLoDSelectPCEnd,
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>&& customExtraDSLayout,
			const system::path& cwdForShaderCompilation, const std::string& perViewPerInstanceDefinition, const std::string& cullAndLoDSelectFuncDefinitions,
			uint32_t workgroupSize=0u
		)
		{
			auto device = _scanner->getDevice();

			auto lodLibraryDSLayout = ILevelOfDetailLibrary::createDescriptorSetLayout(device);
			auto transientInputDSLayout = createInputDescriptorSetLayout(device);
			auto transientOutputDSLayout = createOutputDescriptorSetLayout(device);
			if (!lodLibraryDSLayout || !transientInputDSLayout || !transientOutputDSLayout)
				return nullptr;

			auto instanceCullAndLoDSelectLayout = device->createPipelineLayout(
				cullAndLoDSelectPCBegin,cullAndLoDSelectPCEnd,
				core::smart_refctd_ptr(lodLibraryDSLayout),
				core::smart_refctd_ptr(transientInputDSLayout),
				core::smart_refctd_ptr(transientOutputDSLayout),
				std::move(customExtraDSLayout)
			);
			const asset::SPushConstantRange singleUintRange = {asset::IShader::ESS_COMPUTE,0u,sizeof(uint32_t)};
			auto instanceDrawCullLayout = device->createPipelineLayout(
				&singleUintRange,&singleUintRange+1u,
				core::smart_refctd_ptr(lodLibraryDSLayout),
				core::smart_refctd_ptr(transientInputDSLayout),
				core::smart_refctd_ptr(transientOutputDSLayout)
			);
			auto instanceRefCountingSortPushConstants = _scanner->getDefaultPipelineLayout()->getPushConstantRanges();
			auto instanceRefCountingSortPipelineLayout = device->createPipelineLayout(
				instanceRefCountingSortPushConstants.begin(),instanceRefCountingSortPushConstants.end(),
				nullptr,
				core::smart_refctd_ptr(transientInputDSLayout),
				core::smart_refctd_ptr(transientOutputDSLayout)
			);
			if (!instanceCullAndLoDSelectLayout || !instanceDrawCullLayout || !instanceRefCountingSortPipelineLayout)
				return nullptr;

			if (workgroupSize==0u)
				workgroupSize = device->getPhysicalDevice()->getLimits().maxOptimallyResidentWorkgroupInvocations;
			else if (core::isNPoT(workgroupSize))
				return nullptr;
			
			using shader_source_and_path = std::pair<core::smart_refctd_ptr<asset::ICPUShader>,system::path>;
			auto getShader = [device]<core::StringLiteral Path>() -> shader_source_and_path
			{
				auto system = device->getPhysicalDevice()->getSystem();

				auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
				{
					nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
					system->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
					if (future.wait())
						return future.copy();
					return nullptr;
				};

				auto glslFile = loadBuiltinData(Path.value);
				core::smart_refctd_ptr<asset::ICPUBuffer> glsl;
				{
					glsl = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
					memcpy(glsl->getPointer(), glslFile->getMappedPointer(), glsl->getSize());
				}
				return {core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, Path.value), Path.value};
			};
			auto overrideShader = [device,&cwdForShaderCompilation,workgroupSize,_scanner](shader_source_and_path&& baseShader, std::string additionalCode)
			{
				additionalCode = "\n#define _NBL_GLSL_CULLING_LOD_SELECTION_CULL_WORKGROUP_SIZE_ "+std::to_string(workgroupSize)+"\n"+
					"\n#define _NBL_GLSL_CULLING_LOD_SELECTION_SCAN_WORKGROUP_SIZE_ "+std::to_string(_scanner->getWorkgroupSize())+"\n"+
					additionalCode;
				auto& path = baseShader.second;
				path = cwdForShaderCompilation / path.filename();
				baseShader.first->setFilePathHint(path.string());
				baseShader.first->setShaderStage(asset::IShader::ESS_COMPUTE);
				auto shader =  device->createShader(
					asset::CGLSLCompiler::createOverridenCopy(baseShader.first.get(),"\n%s\n",additionalCode.c_str())
				);
				return device->createSpecializedShader(shader.get(),{nullptr,nullptr,"main"});
			};

			const std::string workgroupSizeDef = "\n#define _NBL_GLSL_WORKGROUP_SIZE_ _NBL_GLSL_CULLING_LOD_SELECTION_CULL_WORKGROUP_SIZE_\n";
			
			auto firstShader = getShader.operator()<core::StringLiteral("nbl/builtin/glsl/culling_lod_selection/instance_cull_and_lod_select.comp")>();
			auto instanceCullAndLoDSelect = device->createComputePipeline(
				nullptr,core::smart_refctd_ptr(instanceCullAndLoDSelectLayout),
				overrideShader(std::move(firstShader),workgroupSizeDef+perViewPerInstanceDefinition+cullAndLoDSelectFuncDefinitions)
			);
			auto instanceDrawCountPrefixSum = device->createComputePipeline(
				nullptr,core::smart_refctd_ptr(instanceDrawCullLayout),
				overrideShader(
					{
						_scanner->getIndirectShader(video::CScanner::EST_INCLUSIVE,video::CScanner::EDT_UINT,video::CScanner::EO_ADD),
						"ICullingLodSelectionSystem::instanceDrawCountPrefixSum"
					},
					"\n#include <nbl/builtin/glsl/culling_lod_selection/instance_draw_count_scan_override.glsl>\n"
				)
			);
			
			auto instanceDrawCull = device->createComputePipeline(
				nullptr,core::smart_refctd_ptr(instanceDrawCullLayout),
				overrideShader(getShader.operator()<core::StringLiteral("nbl/builtin/glsl/culling_lod_selection/instance_draw_cull.comp")>(),workgroupSizeDef+perViewPerInstanceDefinition)
			);

			auto drawInstanceCountPrefixSum = device->createComputePipeline(
				nullptr,core::smart_refctd_ptr(instanceRefCountingSortPipelineLayout),
				overrideShader(
					{
						core::smart_refctd_ptr<asset::ICPUShader>(_scanner->getDefaultShader(video::CScanner::EST_EXCLUSIVE,video::CScanner::EDT_UINT,video::CScanner::EO_ADD)),
						"ICullingLodSelectionSystem::drawInstanceCountPrefixSum"
					},
					"\n#include <nbl/builtin/glsl/culling_lod_selection/draw_instance_count_scan_override.glsl>\n"
				)
			);
			auto instanceRefCountingSortScatter = device->createComputePipeline(
				nullptr,core::smart_refctd_ptr(instanceRefCountingSortPipelineLayout),
				overrideShader(getShader.operator()<core::StringLiteral("nbl/builtin/glsl/culling_lod_selection/instance_ref_counting_sort_scatter.comp")>(),workgroupSizeDef)
			);

			//auto drawCompact = ?;

			if (!instanceCullAndLoDSelect || !instanceDrawCountPrefixSum || !instanceDrawCull || !drawInstanceCountPrefixSum || !instanceRefCountingSortScatter)// || !drawCompact)
				return nullptr;

			return core::smart_refctd_ptr<ICullingLoDSelectionSystem>(new ICullingLoDSelectionSystem(
				std::move(_scanner),workgroupSize,std::move(instanceCullAndLoDSelectLayout),std::move(instanceDrawCullLayout),std::move(instanceRefCountingSortPipelineLayout),
				std::move(instanceCullAndLoDSelect),std::move(instanceDrawCountPrefixSum),std::move(instanceDrawCull),std::move(drawInstanceCountPrefixSum),std::move(instanceRefCountingSortScatter)
			),core::dont_grab);
		}

	protected:
		ICullingLoDSelectionSystem(
			core::smart_refctd_ptr<video::CScanner>&& _scanner, const uint32_t workgroupSize,
			core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _instanceCullAndLoDSelectLayout, core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _instanceDrawCullLayout, core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _instanceRefCountingSortPipelineLayout,
			core::smart_refctd_ptr<video::IGPUComputePipeline>&& instanceCullAndLoDSelect, core::smart_refctd_ptr<video::IGPUComputePipeline>&& _instanceDrawCountPrefixSum, core::smart_refctd_ptr<video::IGPUComputePipeline>&& _instanceDrawCull,
			core::smart_refctd_ptr<video::IGPUComputePipeline>&& _drawInstanceCountPrefixSum, core::smart_refctd_ptr<video::IGPUComputePipeline>&& _instanceRefCountingSortScatter)
			:	m_scanner(std::move(_scanner)), m_instanceCullAndLoDSelectLayout(std::move(_instanceCullAndLoDSelectLayout)), m_instanceDrawCullLayout(std::move(_instanceDrawCullLayout)), m_instanceRefCountingSortPipelineLayout(std::move(_instanceRefCountingSortPipelineLayout)),
				m_instanceCullAndLoDSelect(std::move(instanceCullAndLoDSelect)), m_instanceDrawCountPrefixSum(std::move(_instanceDrawCountPrefixSum)), m_instanceDrawCull(std::move(_instanceDrawCull)),
				m_drawInstanceCountPrefixSum(std::move(_drawInstanceCountPrefixSum)), m_instanceRefCountingSortScatter(std::move(_instanceRefCountingSortScatter)),// m_drawCompact(std::move(_drawCompact)),
				m_workgroupSize(workgroupSize)
		{
		}
		
		#include "nbl/builtin/glsl/culling_lod_selection/potentially_visible_instance_draw_struct.glsl"
		using PotentiallyVisisbleInstanceDraw = nbl_glsl_culling_lod_selection_PotentiallyVisibleInstanceDraw_t;

		core::smart_refctd_ptr<video::CScanner> m_scanner;
		core::smart_refctd_ptr<video::IGPUPipelineLayout> m_instanceCullAndLoDSelectLayout,m_instanceDrawCullLayout,m_instanceRefCountingSortPipelineLayout;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_instanceCullAndLoDSelect,m_instanceDrawCountPrefixSum,m_instanceDrawCull;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_drawInstanceCountPrefixSum,m_instanceRefCountingSortScatter,m_drawCompact;

		const uint32_t m_workgroupSize;
};

#endif
} // end namespace nbl::scene

#endif

