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
static_assert(NBL_BUILTIN_MAX_SCAN_LEVELS&0x1,"NBL_BUILTIN_MAX_SCAN_LEVELS must be odd!");

//
class CScanner final : public core::IReferenceCounted
{
	public:
		static inline constexpr uint32_t DefaultWorkGroupSize = 256u;

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
				elementCount = 0u;
				std::fill_n(cumulativeWorkgroupCount,MaxScanLevels,0u);
			}
			Parameters(const uint32_t _elementCount, const uint32_t wg_size=DefaultWorkGroupSize) : Parameters()
			{
				assert(_elementCount!=0u && "Input element count can't be 0!");
				const auto maxReductionLog2 = core::findMSB(wg_size)*(MaxScanLevels/2u+1u);
				assert(maxReductionLog2>=32u||((_elementCount-1u)>>maxReductionLog2)==0u && "Can't scan this many elements with such small workgroups!");
				elementCount = _elementCount;

				auto wgCountIt = cumulativeWorkgroupCount-1u; 
				while (true)
				{
					*(++wgCountIt) = ((*wgCountIt)-1u)/DefaultWorkGroupSize+1u;
					if ((*wgCountIt)==1u)
						break;
				}
				for (auto revIt=wgCountIt-1u; revIt!=&elementCount; revIt--)
					*(++wgCountIt) = *revIt;
				std::inclusive_scan(cumulativeWorkgroupCount,cumulativeWorkgroupCount+MaxScanLevels,cumulativeWorkgroupCount);
			}

			inline uint32_t getScratchSize(uint32_t ssboAlignment=256u)
			{
				uint32_t uint_count = 1u; // workgroup enumerator
				uint_count += cumulativeWorkgroupCount[MaxScanLevels]; // finished counters
				uint_count += cumulativeWorkgroupCount[MaxScanLevels]; // transient sweep output
				return core::roundUp<uint32_t>(uint_count*sizeof(uint32_t),ssboAlignment);
			}
		};
		struct DispatchInfo
		{
			DispatchInfo() : wg_count(0u)
			{
			}
			DispatchInfo(const uint32_t elementCount, const uint32_t wg_size=DefaultWorkGroupSize)
			{
				assert(elementCount!=0u && "Input element count can't be 0!");
				wg_count = core::min<uint32_t>((0x1u<<16)-1u,(elementCount-1u)/wg_size+1u);
			}

			uint32_t wg_count;
		};

		//
		CScanner(core::smart_refctd_ptr<ILogicalDevice>&& device, const uint32_t wg_size=DefaultWorkGroupSize) : m_device(std::move(device)), m_wg_size(wg_size)
		{
			assert(core::isPoT(wg_size));

			const asset::SPushConstantRange pc_range = { asset::ISpecializedShader::ESS_COMPUTE,0u,sizeof(Parameters) };
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
		IGPUSpecializedShader* getDefaultSpecializedShader(const E_DATA_TYPE dataType, const E_OPERATOR op);

		//
		inline auto getDefaultPipeline(const E_DATA_TYPE dataType, const E_OPERATOR op)
		{
			// ondemand
			if (!m_pipelines[dataType][op])
				m_pipelines[dataType][op] = m_device->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_pipeline_layout),core::smart_refctd_ptr<IGPUSpecializedShader>(getDefaultSpecializedShader(dataType,op)));
			return m_pipelines[dataType][op].get();
		}

		//
		inline void buildParameters(const uint32_t elementCount, Parameters& pushConstants, DispatchInfo& dispatchInfo)
		{
			pushConstants = Parameters(elementCount,m_wg_size);
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
			IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* pipeline_layout, const Parameters& params, const DispatchInfo& dispatchInfo,
			const asset::E_PIPELINE_STAGE_FLAGS srcStageMask, const uint32_t srcBufferBarrierCount, const IGPUCommandBuffer::SBufferMemoryBarrier* srcBufferBarriers,
			const asset::E_PIPELINE_STAGE_FLAGS dstStageMask, const uint32_t dstBufferBarrierCount, const IGPUCommandBuffer::SBufferMemoryBarrier* dstBufferBarriers
		)
		{
			cmdbuf->pushConstants(pipeline_layout,asset::ISpecializedShader::ESS_COMPUTE,0u,sizeof(Parameters),&params);
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
		core::smart_refctd_ptr<IGPUSpecializedShader> m_specialized_shaders[EDT_COUNT][EO_COUNT];
		core::smart_refctd_ptr<IGPUComputePipeline> m_pipelines[EDT_COUNT][EO_COUNT];
		const uint32_t m_wg_size;
};


}

#endif