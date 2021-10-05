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

class ICullingLoDSelectionSystem : public virtual core::IReferenceCounted
{
	public:
		static inline constexpr uint32_t DefaultWorkGroupSize = 256u;

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
			uint32_t lodTableID;
		};

		//
		static core::smart_refctd_ptr<video::IGPUBuffer> createDispatchIndirectBuffer(video::IUtilities* utils, video::IGPUQueue* queue)
		{
			DispatchIndirectParams contents;
			auto setWorkgroups = [](asset::DispatchIndirectCommand_t& cmd)
			{
				cmd.num_groups_x = 0u;
				cmd.num_groups_y = 1u;
				cmd.num_groups_z = 1u;
			};
			setWorkgroups(contents.instanceCullAndLoDSelect);
			setWorkgroups(contents.instanceDrawCountPrefixSum);
			setWorkgroups(contents.instanceDrawCull);
			setWorkgroups(contents.instanceRefCountingSortScatter);
			setWorkgroups(contents.drawCompact);
			// TODO: get rid of this
			contents.instanceRefCountingSortScatter.num_groups_x = 1u;

            video::IGPUBuffer::SCreationParams params;
            params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT)|asset::IBuffer::EUF_INDIRECT_BUFFER_BIT;
            return utils->createFilledDeviceLocalGPUBufferOnDedMem(queue,sizeof(contents),&contents);
		}

		// These buffer ranges can be safely discarded or reused after `processInstancesAndFillIndirectDraws` completes
		struct ScratchBufferRanges
		{
			asset::SBufferRange<video::IGPUBuffer> lodDrawCallOffsets;
			asset::SBufferRange<video::IGPUBuffer> lodDrawCallCounts;
			asset::SBufferRange<video::IGPUBuffer> pvsInstanceDraws;
			asset::SBufferRange<video::IGPUBuffer> prefixSumScratch;
		};
		static ScratchBufferRanges createScratchBuffer(
			video::ILogicalDevice* logicalDevice,
			const uint32_t maxTotalInstances,
			const uint32_t maxTotalDrawcallInstances,
			const uint32_t wg_size=DefaultWorkGroupSize
		)
		{
			ScratchBufferRanges retval;
			{
				const auto& limits = logicalDevice->getPhysicalDevice()->getLimits();
				const auto ssboAlignment = limits.SSBOAlignment;

				retval.lodDrawCallOffsets.offset = 0u;
				retval.lodDrawCallCounts.offset = retval.lodDrawCallOffsets.size = core::alignUp(maxTotalInstances*sizeof(uint32_t),ssboAlignment);
				retval.lodDrawCallCounts.size = retval.lodDrawCallOffsets.size;
				retval.pvsInstanceDraws.offset = retval.lodDrawCallCounts.offset+retval.lodDrawCallCounts.size;
				retval.pvsInstanceDraws.size = core::alignUp(maxTotalDrawcallInstances*sizeof(uint32_t)*4u,ssboAlignment);
				retval.prefixSumScratch.offset = retval.pvsInstanceDraws.offset+retval.pvsInstanceDraws.size;
				{
					video::CScanner::Parameters params;
					auto schedulerParams = video::CScanner::SchedulerParameters(params,maxTotalDrawcallInstances,wg_size);
					retval.prefixSumScratch.size = params.getScratchSize(ssboAlignment);
				}
			}
			{
				video::IGPUBuffer::SCreationParams params;
				params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
				retval.lodDrawCallOffsets.buffer = 
				retval.lodDrawCallCounts.buffer =
				retval.pvsInstanceDraws.buffer =
				retval.prefixSumScratch.buffer =
					logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,retval.prefixSumScratch.offset+retval.prefixSumScratch.size);
			}
			return retval;
		}

		// Per-View Per-Instance buffer should hold at least an MVP matrix
		template<typename PerViewPerInstanceDataType>
		static core::smart_refctd_ptr<video::IGPUBuffer> createPerViewPerInstanceDataBuffer(video::ILogicalDevice* logicalDevice, const uint32_t maxTotalDrawcallInstances)
		{
			return createPerViewPerInstanceDataBuffer(logicalDevice,maxTotalDrawcallInstances,sizeof(PerViewPerInstanceDataType));
		}
		static core::smart_refctd_ptr<video::IGPUBuffer> createPerViewPerInstanceDataBuffer(video::ILogicalDevice* logicalDevice, const uint32_t maxTotalDrawcallInstances, const uint32_t perViewPerInstanceDataSize)
		{
            video::IGPUBuffer::SCreationParams params;
			params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
            return logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,perViewPerInstanceDataSize*maxTotalDrawcallInstances);
		}

		// Instance Redirect buffer holds a `uvec2` of `{instanceGUID,perViewPerInstanceDataID}` for each instace of a drawcall
		static core::smart_refctd_ptr<video::IGPUBuffer> createInstanceRedirectBuffer(video::ILogicalDevice* logicalDevice, const uint32_t maxTotalDrawcallInstances)
		{
            video::IGPUBuffer::SCreationParams params;
            params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT)|asset::IBuffer::EUF_VERTEX_BUFFER_BIT;
            return logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(uint32_t)*2u*maxTotalDrawcallInstances);
		}


		//
		static inline constexpr auto InputDescriptorBindingCount = 8u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createInputDescriptorSetLayout(video::ILogicalDevice* device, bool withMDICounts=false)
		{
			withMDICounts &= device->getPhysicalDevice()->getFeatures().multiDrawIndirect;
			withMDICounts &= device->getPhysicalDevice()->getFeatures().drawIndirectCount;

			video::IGPUDescriptorSetLayout::SBinding bindings[InputDescriptorBindingCount];
			for (auto i=0u; i<InputDescriptorBindingCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::EDT_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
				bindings[i].samplers = nullptr;
			}

			uint32_t count = InputDescriptorBindingCount;
			if (!withMDICounts)
				count--;
			return device->createGPUDescriptorSetLayout(bindings,bindings+InputDescriptorBindingCount);
		}
		//
		static inline constexpr auto OutputDescriptorBindingCount = 4u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createOutputDescriptorSetLayout(video::ILogicalDevice* device, bool withMDICounts=false)
		{
			withMDICounts &= device->getPhysicalDevice()->getFeatures().multiDrawIndirect;
			withMDICounts &= device->getPhysicalDevice()->getFeatures().drawIndirectCount;

			video::IGPUDescriptorSetLayout::SBinding bindings[OutputDescriptorBindingCount];
			for (auto i=0u; i<OutputDescriptorBindingCount; i++)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::EDT_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
				bindings[i].samplers = nullptr;
			}

			uint32_t count = OutputDescriptorBindingCount;
			if (!withMDICounts)
				count--;
			return device->createGPUDescriptorSetLayout(bindings,bindings+count);
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
			auto ds = device->createGPUDescriptorSet(pool,std::move(layout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[InputDescriptorBindingCount];
				video::IGPUDescriptorSet::SDescriptorInfo infos[InputDescriptorBindingCount] =
				{
					dispatchIndirect,
					instanceList,
					scratchBufferRanges.lodDrawCallOffsets,
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
					writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
					writes[i].info = infos+i;
				}
				uint32_t count = InputDescriptorBindingCount;
				if (_layout->getBindings().size()==InputDescriptorBindingCount)
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
			auto ds = device->createGPUDescriptorSet(pool,std::move(layout));
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
					writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
					writes[i].info = infos+i;
				}
				uint32_t count = OutputDescriptorBindingCount;
				if (_layout->getBindings().size()==OutputDescriptorBindingCount)
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
		struct Params
		{
			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			asset::SBufferBinding<video::IGPUBuffer> indirectDispatchParams;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> lodLibraryDS;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> transientInputDS;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> transientOutputDS;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> customDS;
			uint32_t directInstanceCount; // set as 0u for indirect dispatch
			// these are for the pipeline barriers
			asset::SBufferRange<video::IGPUBuffer> instanceList;
			ScratchBufferRanges scratchBufferRanges;
			asset::SBufferRange<video::IGPUBuffer> drawCalls;
			asset::SBufferRange<video::IGPUBuffer> perViewPerInstance;
			asset::SBufferRange<video::IGPUBuffer> perInstanceRedirectAttribs;
			asset::SBufferRange<video::IGPUBuffer> drawCounts = {};
		};
		void processInstancesAndFillIndirectDraws(const Params& params)
		{
			auto cmdbuf = params.cmdbuf;
			const auto queueFamilyIndex = cmdbuf->getPool()->getQueueFamilyIndex();
			const asset::SBufferRange<video::IGPUBuffer> indirectRange = {
				params.indirectDispatchParams.offset,sizeof(DispatchIndirectParams),
				params.indirectDispatchParams.buffer
			};

			const auto internalStageFlags = core::bitflag(asset::EPSF_DRAW_INDIRECT_BIT)|asset::EPSF_COMPUTE_SHADER_BIT;

			constexpr auto MaxBufferBarriers = 5u;
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
			setBarrierBuffer(barriers[0],indirectRange,indirectAccessMask,indirectAccessMask);

#if 0
			auto srcStageFlags = core::bitflag(asset::EPSF_COMPUTE_SHADER_BIT);
			if (params.directInstanceCount)
			{
				cmdbuf->bindComputePipeline(directInstanceCullAndLoDSelect.get());
				cmdbuf->bindDescriptorSets(EPBP_COMPUTE,directInstanceCullAndLoDSelectLayout.get(),0u,4u,&params.lodLibraryDS);
				//cmdbuf->dispatch(,1u,1u); TODO
			}
			else
			{
				srcStageFlags |= asset::EPSF_DRAW_INDIRECT_BIT;
				cmdbuf->bindComputePipeline(indirectInstanceCullAndLoDSelect.get());
				cmdbuf->bindDescriptorSets(EPBP_COMPUTE,indirectInstanceCullAndLoDSelectLayout.get(),0u,4u,&params.lodLibraryDS);
				cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,instanceCullAndLoDSelect));
			}
			{
				setBarrierBuffer(barriers[1],params.lodDrawCallOffsets,core::bitflag(asset::EAF_SHADER_WRITE_BIT),core::bitflag(asset::EAF_SHADER_READ_BIT));
				setBarrierBuffer(barriers[2],params.lodDrawCallCount,rwAccessMask,core::bitflag(asset::EAF_SHADER_READ_BIT));
				setBarrierBuffer(barriers[3],params.prefixSumScratch,rwAccessMask,rwAccessMask);
				// TODO: perViewData
				cmdbuf->pipelineBarrier(srcStageFlags,internalStageFlags,asset::EDF_NONE,0u,nullptr,5u,barriers,0u,nullptr);
			}

			cmdbuf->bindComputePipeline(instanceDrawCull.get());
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,sharedPipelineLayout.get(),0u,4u,&params.lodLibraryDS.get());
			cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,instanceDrawCull));
			{
				//setBarrierBuffer(barriers[1],params.drawCalls,rwAccessMask,rwAccessMask);
				//setBarrierBuffer(barriers[2],params.unorderedDrawCalls,rwAccessMask,core::bitflag(asset::EAF_SHADER_READ_BIT));
				cmdbuf->pipelineBarrier(internalStageFlags,internalStageFlags,asset::EDF_NONE,0u,nullptr,3u,barriers,0u,nullptr);
			}
#endif

			cmdbuf->bindComputePipeline(drawInstanceCountPrefixSum.get());
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,instanceRefCountingSortPipelineLayout.get(),1u,2u,&params.transientInputDS.get());
			{
				video::CScanner::DispatchInfo dispatchInfo;
				{
					const auto drawCount = (params.drawCalls.size!=~0u ? params.drawCalls.size:params.drawCalls.buffer->getSize())/sizeof(uint32_t);
					video::CScanner::DefaultPushConstants pushConstants;
					m_scanner->buildParameters(drawCount,pushConstants,dispatchInfo);
					cmdbuf->pushConstants(instanceRefCountingSortPipelineLayout.get(),asset::ISpecializedShader::ESS_COMPUTE,0u,sizeof(pushConstants),&pushConstants);
				}
				//cmdbuf->dispatch(dispatchInfo.wg_count,1u,1u);
			}
			{
				setBarrierBuffer(barriers[1],params.drawCalls,rwAccessMask,indirectAccessMask);
				setBarrierBuffer(barriers[2],params.scratchBufferRanges.prefixSumScratch,rwAccessMask,wAccessMask);
				cmdbuf->pipelineBarrier(internalStageFlags,internalStageFlags,asset::EDF_NONE,0u,nullptr,3u,barriers,0u,nullptr);
			}

			cmdbuf->bindComputePipeline(instanceRefCountingSortScatter.get());
			cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,instanceRefCountingSortScatter));
			{
				setBarrierBuffer(barriers[1],params.perInstanceRedirectAttribs,wAccessMask,wAccessMask|asset::EAF_VERTEX_ATTRIBUTE_READ_BIT);
				setBarrierBuffer(barriers[2],params.scratchBufferRanges.prefixSumScratch,wAccessMask,rwAccessMask);
				cmdbuf->pipelineBarrier(internalStageFlags,internalStageFlags|asset::EPSF_VERTEX_INPUT_BIT,asset::EDF_NONE,0u,nullptr,3u,barriers,0u,nullptr);
			}
#if 0
			// drawcall compaction
			if (params.transientOutputDS->getLayout()->getBindings().size()==OutputDescriptorBindingCount)
			{
				cmdbuf->bindComputePipeline(drawCompact.get());
				cmdbuf->dispatchIndirect(indirectRange.buffer.get(),indirectRange.offset+offsetof(DispatchIndirectParams,drawCompact));
			}
#endif
		}

	//protected:
		ICullingLoDSelectionSystem(video::IUtilities* utils, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>&& customDSLayout, uint32_t wg_size=DefaultWorkGroupSize)
		{
			auto device = utils->getLogicalDevice();

			auto lodLibraryDSLayout = ILevelOfDetailLibrary::createDescriptorSetLayout(device);
			auto transientInputDSLayout = createInputDescriptorSetLayout(device);
			auto transientOutputDSLayout = createOutputDescriptorSetLayout(device);

			asset::SPushConstantRange range = {asset::ISpecializedShader::ESS_COMPUTE,0u,sizeof(uint32_t)};
			directInstanceCullAndLoDSelectLayout = device->createGPUPipelineLayout(
				&range,&range+1u,
				core::smart_refctd_ptr(lodLibraryDSLayout),
				core::smart_refctd_ptr(transientInputDSLayout),
				core::smart_refctd_ptr(transientOutputDSLayout),
				core::smart_refctd_ptr(customDSLayout)
			);
			indirectInstanceCullAndLoDSelectLayout = device->createGPUPipelineLayout(
				nullptr,nullptr,
				core::smart_refctd_ptr(lodLibraryDSLayout),
				core::smart_refctd_ptr(transientInputDSLayout),
				core::smart_refctd_ptr(transientOutputDSLayout),
				core::smart_refctd_ptr(customDSLayout)
			);
			m_scanner = core::smart_refctd_ptr<video::CScanner>(utils->getDefaultScanner());
			auto instanceRefCountingSortPushConstants = m_scanner->getDefaultPipelineLayout()->getPushConstantRanges();
			instanceRefCountingSortPipelineLayout = device->createGPUPipelineLayout(
				instanceRefCountingSortPushConstants.begin(),instanceRefCountingSortPushConstants.end(),
				nullptr,
				core::smart_refctd_ptr(transientInputDSLayout),
				core::smart_refctd_ptr(transientOutputDSLayout)
			);
			
			auto createOverridenScanShader = [device,this](const char* additionalCode) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
			{
				auto baseScanShader = m_scanner->getDefaultShader(video::CScanner::EST_EXCLUSIVE,video::CScanner::EDT_UINT,video::CScanner::EO_ADD);
				auto shader =  device->createGPUShader(
					asset::IGLSLCompiler::createOverridenCopy(baseScanShader,"\n%s\n",additionalCode)
				);
				return device->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});
			};
			auto createShader = [device,wg_size](auto uniqueString) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
			{
				auto system = device->getPhysicalDevice()->getSystem();
				auto glsl = system->loadBuiltinData<decltype(uniqueString)>();
				auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl),asset::IShader::buffer_contains_glsl_t{});
				auto shader = device->createGPUShader(asset::IGLSLCompiler::createOverridenCopy(cpushader.get(),"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n",wg_size));
				return device->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});
			};
			//directInstanceCullAndLoDSelect = device->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(directInstanceCullAndLoDSelectLayout),loadShader());
			//indirectInstanceCullAndLoDSelect = device->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(indirectInstanceCullAndLoDSelectLayout),loadShader());
			//instanceDrawCull = device->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),loadShader());
			drawInstanceCountPrefixSum = device->createGPUComputePipeline(
				nullptr,core::smart_refctd_ptr(instanceRefCountingSortPipelineLayout),
				createOverridenScanShader("#include <nbl/builtin/glsl/culling_lod_selection/draw_instance_count_scan_override.glsl>")
			);
			instanceRefCountingSortScatter = device->createGPUComputePipeline(
				nullptr,core::smart_refctd_ptr(instanceRefCountingSortPipelineLayout),
				createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/culling_lod_selection/instance_ref_counting_sort_scatter.comp")())
			);
			//drawCompact = device->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),loadShader());
		}
	protected:

		core::smart_refctd_ptr<video::CScanner> m_scanner;
		core::smart_refctd_ptr<video::IGPUPipelineLayout> directInstanceCullAndLoDSelectLayout,indirectInstanceCullAndLoDSelectLayout,instanceRefCountingSortPipelineLayout;
		core::smart_refctd_ptr<video::IGPUComputePipeline> directInstanceCullAndLoDSelect,indirectInstanceCullAndLoDSelect,instanceDrawCull,drawInstanceCountPrefixSum,instanceRefCountingSortScatter,drawCompact;
};


} // end namespace nbl::scene

#endif

