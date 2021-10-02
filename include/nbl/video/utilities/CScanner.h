// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_SCANNER_H_INCLUDED_
#define _NBL_VIDEO_C_SCANNER_H_INCLUDED_


#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/utilities/IDescriptorSetCache.h"


namespace nbl::video
{

#include "nbl/builtin/glsl/scan/parameters_struct.glsl"
#include "nbl/builtin/glsl/scan/default_scheduler.glsl"
static_assert(NBL_BUILTIN_MAX_SCAN_LEVELS&0x1,"NBL_BUILTIN_MAX_SCAN_LEVELS must be odd!");

//
class CScanner final : public core::IReferenceCounted
{
	public:
		static inline constexpr uint32_t DefaultWorkGroupSize = 256u;
		
		enum E_SCAN_TYPE : uint8_t
		{
			 EST_INCLUSIVE = _NBL_GLSL_SCAN_TYPE_INCLUSIVE_,
			 EST_EXCLUSIVE = _NBL_GLSL_SCAN_TYPE_EXCLUSIVE_,
			 EST_COUNT
		};
		enum E_DATA_TYPE : uint8_t
		{
			EDT_UINT=0u,
			EDT_INT,
			EDT_FLOAT,
			EDT_COUNT
		};
		enum E_OPERATOR : uint8_t
		{
			EO_AND = _NBL_GLSL_SCAN_OP_AND_,
			EO_XOR = _NBL_GLSL_SCAN_OP_XOR_,
			EO_OR = _NBL_GLSL_SCAN_OP_OR_,
			EO_ADD = _NBL_GLSL_SCAN_OP_ADD_,
			EO_MUL = _NBL_GLSL_SCAN_OP_MUL_,
			EO_MIN = _NBL_GLSL_SCAN_OP_MIN_,
			EO_MAX = _NBL_GLSL_SCAN_OP_MAX_,
			EO_COUNT = _NBL_GLSL_SCAN_OP_COUNT_
		};

		//
		struct Parameters : nbl_glsl_scan_Parameters_t
		{
			static inline constexpr uint32_t MaxScanLevels = NBL_BUILTIN_MAX_SCAN_LEVELS;

			Parameters()
			{
				std::fill_n(elementCount,MaxScanLevels,0u);
				std::fill_n(temporaryStorageOffset,MaxScanLevels/2-1,0u);
			}
			Parameters(const uint32_t _elementCount, const uint32_t wg_size=DefaultWorkGroupSize) : Parameters()
			{
				assert(_elementCount!=0u && "Input element count can't be 0!");
				const auto maxReductionLog2 = core::findMSB(wg_size)*(MaxScanLevels/2u+1u);
				assert(maxReductionLog2>=32u||((_elementCount-1u)>>maxReductionLog2)==0u && "Can't scan this many elements with such small workgroups!");

				elementCount[0u] = _elementCount;
				for (topLevel=0u; elementCount[topLevel]>wg_size;)
					elementCount[++topLevel] = (elementCount[topLevel]-1u)/wg_size+1u;
				std::reverse_copy(elementCount,elementCount+topLevel,elementCount+topLevel+1u);
				
				std::copy_n(elementCount+1u,topLevel,temporaryStorageOffset);
				std::exclusive_scan(temporaryStorageOffset,temporaryStorageOffset+sizeof(temporaryStorageOffset)/sizeof(uint32_t),temporaryStorageOffset,0u);
			}

			inline uint32_t getScratchSize(uint32_t ssboAlignment=256u)
			{
				uint32_t uint_count = 1u; // workgroup enumerator
				uint_count += temporaryStorageOffset[MaxScanLevels/2u-2u]; // last scratch offset
				uint_count += elementCount[topLevel]; // and its size
				return core::roundUp<uint32_t>(uint_count*sizeof(uint32_t),ssboAlignment);
			}
		};
		struct SchedulerParameters : nbl_glsl_scan_DefaultSchedulerParameters_t
		{
			SchedulerParameters()
			{
				std::fill_n(finishedFlagOffset,Parameters::MaxScanLevels-1,0u);
				std::fill_n(lastWorkgroupDependentCount,Parameters::MaxScanLevels/2,1u);
				std::fill_n(cumulativeWorkgroupCount,Parameters::MaxScanLevels,0u);
			}
			SchedulerParameters(Parameters& outScanParams, const uint32_t _elementCount, const uint32_t wg_size=DefaultWorkGroupSize) : SchedulerParameters()
			{
				outScanParams = Parameters(_elementCount,wg_size);
				const auto topLevel = outScanParams.topLevel;

				std::copy_n(outScanParams.elementCount+1u,topLevel,cumulativeWorkgroupCount);
				cumulativeWorkgroupCount[topLevel] = 1u;

				for (auto i=0u; i<topLevel; i++)
					lastWorkgroupDependentCount[i] = cumulativeWorkgroupCount[i]-(cumulativeWorkgroupCount[i+1u]-1u)*wg_size;
				std::reverse_copy(cumulativeWorkgroupCount,cumulativeWorkgroupCount+topLevel,cumulativeWorkgroupCount+topLevel+1u);

				std::copy_n(cumulativeWorkgroupCount+1u,topLevel,finishedFlagOffset);
				std::copy_n(cumulativeWorkgroupCount+topLevel,topLevel,finishedFlagOffset+topLevel);

				const auto finishedFlagCount = sizeof(finishedFlagOffset)/sizeof(uint32_t);
				const auto finishedFlagsSize = std::accumulate(finishedFlagOffset,finishedFlagOffset+finishedFlagCount,0u);
				std::exclusive_scan(finishedFlagOffset,finishedFlagOffset+finishedFlagCount,finishedFlagOffset,0u);
				for (auto i=0u; i<sizeof(Parameters::temporaryStorageOffset)/sizeof(uint32_t); i++)
					outScanParams.temporaryStorageOffset[i] += finishedFlagsSize;
					
				std::inclusive_scan(cumulativeWorkgroupCount,cumulativeWorkgroupCount+Parameters::MaxScanLevels,cumulativeWorkgroupCount);
			}
		};
		struct DefaultPushConstants
		{
			Parameters scanParams;
			SchedulerParameters schedulerParams;
		};
		struct DispatchInfo
		{
			DispatchInfo() : wg_count(0u)
			{
			}
			DispatchInfo(const uint32_t elementCount, const uint32_t wg_size=DefaultWorkGroupSize)
			{
				assert(elementCount!=0u && "Input element count can't be 0!");
				constexpr auto workgroupSpinningProtection = 4u;
				constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
				wg_count = core::min<uint32_t>(beefyGPUWorkgroupMaxOccupancy,(elementCount-1u)/(wg_size*workgroupSpinningProtection)+1u);
			}

			uint32_t wg_count;
		};

		//
		CScanner(core::smart_refctd_ptr<ILogicalDevice>&& device, const uint32_t wg_size=DefaultWorkGroupSize) : m_device(std::move(device)), m_wg_size(wg_size)
		{
			assert(core::isPoT(wg_size));

			const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE,0u,sizeof(DefaultPushConstants) };
			const IGPUDescriptorSetLayout::SBinding bindings[2] = {
				{ 0u, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr }, // main buffer
				{ 1u, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr } // scratch
			};

			m_ds_layout = m_device->createGPUDescriptorSetLayout(bindings,bindings+sizeof(bindings)/sizeof(IGPUDescriptorSetLayout::SBinding));
			m_pipeline_layout = m_device->createGPUPipelineLayout(&pc_range,&pc_range+1,core::smart_refctd_ptr(m_ds_layout));
		}

		//
		inline auto getDefaultDescriptorSetLayout() const { return m_ds_layout.get(); }

		//
		inline auto getDefaultPipelineLayout() const { return m_pipeline_layout.get(); }

		//
		IGPUSpecializedShader* getDefaultSpecializedShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op);

		//
		inline auto getDefaultPipeline(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op)
		{
			// ondemand
			if (!m_pipelines[scanType][dataType][op])
				m_pipelines[scanType][dataType][op] = m_device->createGPUComputePipeline(
					nullptr,core::smart_refctd_ptr(m_pipeline_layout),
					core::smart_refctd_ptr<IGPUSpecializedShader>(getDefaultSpecializedShader(scanType,dataType,op))
				);
			return m_pipelines[scanType][dataType][op].get();
		}

		//
		inline void buildParameters(const uint32_t elementCount, DefaultPushConstants& pushConstants, DispatchInfo& dispatchInfo)
		{
			pushConstants.schedulerParams = SchedulerParameters(pushConstants.scanParams,elementCount,m_wg_size);
			dispatchInfo = DispatchInfo(elementCount,m_wg_size);
		}

		//
		static inline void updateDescriptorSet(ILogicalDevice* device, IGPUDescriptorSet* set, const asset::SBufferRange<IGPUBuffer>& input_range, const asset::SBufferRange<IGPUBuffer>& scratch_range)
		{
			IGPUDescriptorSet::SDescriptorInfo infos[2];
			infos[0].desc = input_range.buffer;
			infos[0].buffer = {input_range.offset,input_range.size};
			infos[1].desc = scratch_range.buffer;
			infos[1].buffer = {scratch_range.offset,scratch_range.size};

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[2];
			for (auto i=0u; i<2u; i++)
			{
				writes[i].dstSet = set;
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
				writes[i].info = infos+i;
			}

			device->updateDescriptorSets(2,writes,0u,nullptr);
		}
		inline void updateDescriptorSet(IGPUDescriptorSet* set, const asset::SBufferRange<IGPUBuffer>& input_range, const asset::SBufferRange<IGPUBuffer>& scratch_range)
		{
			updateDescriptorSet(m_device.get(),set,input_range,scratch_range);
		}

		// Half and sizeof(uint32_t) of the scratch buffer need to be cleared to 0s
		static inline void dispatchHelper(
			IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* pipeline_layout, const DefaultPushConstants& pushConstants, const DispatchInfo& dispatchInfo,
			const asset::E_PIPELINE_STAGE_FLAGS srcStageMask, const uint32_t srcBufferBarrierCount, const IGPUCommandBuffer::SBufferMemoryBarrier* srcBufferBarriers,
			const asset::E_PIPELINE_STAGE_FLAGS dstStageMask, const uint32_t dstBufferBarrierCount, const IGPUCommandBuffer::SBufferMemoryBarrier* dstBufferBarriers
		)
		{
			cmdbuf->pushConstants(pipeline_layout,asset::ISpecializedShader::ESS_COMPUTE,0u,sizeof(DefaultPushConstants),&pushConstants);
			if (srcStageMask!=asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT&&srcBufferBarrierCount)
				cmdbuf->pipelineBarrier(srcStageMask,asset::EPSF_COMPUTE_SHADER_BIT,asset::EDF_NONE,0u,nullptr,srcBufferBarrierCount,srcBufferBarriers,0u,nullptr);
			cmdbuf->dispatch(dispatchInfo.wg_count,1u,1u);
			if (dstStageMask!=asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT&&dstBufferBarrierCount)
				cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT,dstStageMask,asset::EDF_NONE,0u,nullptr,dstBufferBarrierCount,dstBufferBarriers,0u,nullptr);
		}

    protected:
		~CScanner()
		{
			// all drop themselves automatically
		}

		core::smart_refctd_ptr<ILogicalDevice> m_device;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> m_ds_layout;
		core::smart_refctd_ptr<IGPUPipelineLayout> m_pipeline_layout;
		core::smart_refctd_ptr<IGPUSpecializedShader> m_specialized_shaders[EST_COUNT][EDT_COUNT][EO_COUNT];
		core::smart_refctd_ptr<IGPUComputePipeline> m_pipelines[EST_COUNT][EDT_COUNT][EO_COUNT];
		const uint32_t m_wg_size;
};


}

#endif