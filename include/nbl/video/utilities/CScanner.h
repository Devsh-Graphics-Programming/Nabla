// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_SCANNER_H_INCLUDED_
#define _NBL_VIDEO_C_SCANNER_H_INCLUDED_


#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/IDescriptorSetCache.h"


namespace nbl::video
{

#include "nbl/builtin/glsl/scan/parameters_struct.glsl"
#include "nbl/builtin/glsl/scan/default_scheduler.glsl"
static_assert(NBL_BUILTIN_MAX_SCAN_LEVELS&0x1,"NBL_BUILTIN_MAX_SCAN_LEVELS must be odd!");

/**
Utility class to help you perform the equivalent of `std::inclusive_scan` and `std::exclusive_scan` with data on the GPU.

The basic building block is a Blelloch-Scan, the `nbl_glsl_workgroup{Add/Mul/And/Xor/Or/Min/Max}{Exclusive/Inclusive}`:
https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
https://classes.engineering.wustl.edu/cse231/core/index.php/Scan
Also referred to as an "Upsweep-Downsweep Scan" due to the fact it computes Reductions hierarchically until there's only one block left,
then does Prefix Sums and propagates the results into more blocks until we're back at 1 element of a block for 1 element of the input.

The workgroup scan is itself probably built out of Hillis-Steele subgroup scans, we use `KHR_shader_subgroup_arithmetic` whenever available,
but fall back to our own "software" emulation of subgroup arithmetic using Hillis-Steele and some scratch shared memory.

The way the workgroup scans are combined is amongst the most advanced in its class, because it performs the scan as a single dispatch
via some clever scheduling which allows it to also be used in "indirect mode", which is when you don't know the number of elements
that you'll be scanning on the CPU side. This is why it provides two flavours of the compute shader.

The scheduling relies on two principles:
- Virtual and Persistent Workgroups
- Atomic Counters as Sempahores

## Virtual Workgroups TODO: Move this Paragraph somewhere else.
Generally speaking, launching a new workgroup has non-trivial overhead.

Also most IHVs, especially AMD have silly limits on the ranges of dispatches (like 64k workgroups), which also apply to 1D dispatches.

It becomes impossible to keep a simple 1 invocation to 1 data element relationship when processing a large buffer without reusing workgroups.

Virtual Persistent Workgroups is a $20 term for a $0.25 idea, its simply to do the following:
1. Launch a 1D dispatch "just big enough" to saturate your GPU, `SPhysicalDeviceLimits::maxResidentInvocations` helps you figure out this number
2. Make a single workgroup perform the task of multiple workgroups by repeating itself
3. Instead of relying on `gl_WorkGroupID` or `gl_GlobalInvocationID` to find your work items, use your own ID unique for the virtual workgroup

This usually has the form of
```glsl
for (uint virtualWorkgroupIndex=gl_GlobalInvocationID.x; virtualWorkgroupIndex<virtualWorkgroupCount; virtualWorkgroupIndex++)
{
   // do actual work for a single workgroup
}
```

This actually opens some avenues to abusing the system to achieve customized scheduling.

The GLSL and underlying spec give no guarantees and explicitly warn AGAINST assuming that a workgroup with a lower ID will begin executing
no later than a workgroup with a higher ID. Actually attempting to enforce this, such as this
```glsl
layout() buffer coherent Sched
{
   uint nextWorkgroup; // initial value is 0 before the dispatch
};

while (nextWorkgroup!=gl_GlobalInvocationID.x) {}
atomicMax(nextWorkgroup,gl_GlobalInvocationID.x+1);
```
has the potential to deadlock and TDR your GPU.

However if you use such an atomic to assign the `virtualWorkgroupIndex` in lieu of spinning
```glsl
uint virtualWorkgroupIndex;
for ((virtualWorkgroupIndex=atomicAdd(nextWorkgroup,1u))<virtualWorkgroupCount)
{
   // do actual work for a single workgroup
}
```
the ordering of starting work is now enforced (still wont guarantee the order of completion).

## Atomic Counters as Semaphores
To scan arbitrarily large arrays, we already use Virtual Workgroups.

For improved cache coherence and more bandwidth on the higher level reduction and scan blocks,
its best to use a temporary scratch buffer roughly of size `O(2 log_{WorkgroupSize}(n))`.

We can however turn the BrainWorm(TM) up to 11, and do the whole scan in a single dispatch.

First, we assign a Linear Index using the trick outlined in Virtual Workgroups section, to every scan block (workgroup)
such that if executed serially, **the lower index block would have finished before any higher index block.**
https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems3/elementLinks/39fig06.jpg
It would be also useful to keep some table that would let us map the Linear Index to the scan level.

Then we use a little bit more scratch for some atomic counters.

A naive scheduler would have one atomic counter per upsweep-downsweep level, which would be incremented AFTER the workgroup
is finished with the scan and writes its outputs, this would tell us how many workgroups have completed at the level so far.

Then we could figure out the scan level given workgroup Linear Index, then spinwait until the atomic counter mapped to the
previous scan level tells us all workgroups have completed.

HOWEVER, this naive approach is really no better than separate dispatches with pipeline barriers in the middle.

Subsequently we turn up the BrainWorm(TM) to 12, and notice that we don't really need to wait on an entire previous level,
just the workgroups that will produce the data that the current one will process.

So there's one atomic per workgroup above the second level while sweeping up (top block included as it waits for reduction results),
this atomic is incremented by immediately lower level workgroups which provide the inputs to the current workgroup. The current
workgroup will have to spinwait on its atomic until it reaches WORKGROUP_SIZE (with the exception of the last workgroup in the level,
where the value might be different when the number of workgroups in previous level is not divisible by WORKGROUP_SIZE).

In the downsweep phase (waiting for scan results), multiple lower level workgroups spin on the same atomic until it reaches 1, since
a single input is needed by multiple outputs.

## So what's an Indirect Scan?

It is when you don't know the count of the elements to scan, because lets say another GPU dispatch produces the list to scan and its
variable length, for example culling systems.

Naturally because of this, you won't know:
- the number of workgroups to dispatch, so DispatchIndirect is needed
- the number of upsweep/downsweep levels
- the number of workgroups in each level
- the size and offsets of the auxillary output data array for each level
- the size and offsets of the atomics for each level

## Further Work

We could reduce auxillary memory size some more, by noting that only two levels need to access the same intermediate result and
only workgroups from 3 immediately consecutive levels can ever work simultaneously due to our scheduler.

Right now we allocate and don't alias the auxillary memory used for storage of the intermediate workgroup results.

# I hear you say Nabla is too complex...

If you think that AAA Engines have similar and less complicated utilities, you're gravely mistaken, the AMD GPUs in the
Playstation and Xbox have hardware workgroup ordered dispatch and a `mbcnt` instruction which allows you do to single dispatch
prefix sum with subgroup sized workgroups at peak Bandwidth efficiency in about 6 lines of HLSL.

Console devs get to bring a gun to a knife fight...
**/
#if 0 // legacy & port to HLSL
class CScanner final : public core::IReferenceCounted
{
	public:		
		enum E_SCAN_TYPE : uint8_t
		{
			 // computes output[n] = Sum_{i<=n}(input[i])
			 EST_INCLUSIVE = _NBL_GLSL_SCAN_TYPE_INCLUSIVE_,
			 // computes output[n] = Sum_{i<n}(input[i]), meaning first element is identity
			 EST_EXCLUSIVE = _NBL_GLSL_SCAN_TYPE_EXCLUSIVE_,
			 EST_COUNT
		};
		// Only 4 byte wide data types supported due to need to trade the via shared memory,
		// different combinations of data type and operator have different identity elements.
		// `EDT_INT` and `EO_MIN` will have `INT_MAX` as identity, while `EDT_UINT` would have `UINT_MAX`  
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

		// This struct is only for managing where to store intermediate results of the scans
		struct Parameters : nbl_glsl_scan_Parameters_t // this struct and its methods are also available in GLSL so you can launch indirect dispatches
		{
			static inline constexpr uint32_t MaxScanLevels = NBL_BUILTIN_MAX_SCAN_LEVELS;

			Parameters()
			{
				std::fill_n(lastElement,MaxScanLevels/2+1,0u);
				std::fill_n(temporaryStorageOffset,MaxScanLevels/2,0u);
			}
			// build the constant tables for each level given the number of elements to scan and workgroupSize
			Parameters(const uint32_t _elementCount, const uint32_t workgroupSize) : Parameters()
			{
				assert(_elementCount!=0u && "Input element count can't be 0!");
				const auto maxReductionLog2 = hlsl::findMSB(workgroupSize)*(MaxScanLevels/2u+1u);
				assert(maxReductionLog2>=32u||((_elementCount-1u)>>maxReductionLog2)==0u && "Can't scan this many elements with such small workgroups!");

				lastElement[0u] = _elementCount-1u;
				for (topLevel=0u; lastElement[topLevel]>=workgroupSize;)
					temporaryStorageOffset[topLevel-1u] = lastElement[++topLevel] = lastElement[topLevel]/workgroupSize;
				
				std::exclusive_scan(temporaryStorageOffset,temporaryStorageOffset+sizeof(temporaryStorageOffset)/sizeof(uint32_t),temporaryStorageOffset,0u);
			}
                        // given already computed tables of lastElement indices per level, number of levels, and storage offsets, tell us total auxillary buffer size needed
			inline uint32_t getScratchSize(uint32_t ssboAlignment=256u)
			{
				uint32_t uint_count = 1u; // workgroup enumerator
				uint_count += temporaryStorageOffset[MaxScanLevels/2u-1u]; // last scratch offset
				uint_count += lastElement[topLevel]+1u; // and its size
				return core::roundUp<uint32_t>(uint_count*sizeof(uint32_t),ssboAlignment);
			}
		};
                // the default scheduler we provide works as described above in the big documentation block
		struct SchedulerParameters : nbl_glsl_scan_DefaultSchedulerParameters_t  // this struct and its methods are also available in GLSL so you can launch indirect dispatches
		{
			SchedulerParameters()
			{
				std::fill_n(finishedFlagOffset,Parameters::MaxScanLevels-1,0u);
				std::fill_n(cumulativeWorkgroupCount,Parameters::MaxScanLevels,0u);
			}
                        // given the number of elements and workgroup size, figure out how many atomics we need
                        // also account for the fact that we will want to use the same scratch buffer both for the
                        // scheduler's atomics and the aux data storage
			SchedulerParameters(Parameters& outScanParams, const uint32_t _elementCount, const uint32_t workgroupSize) : SchedulerParameters()
			{
				outScanParams = Parameters(_elementCount,workgroupSize);
				const auto topLevel = outScanParams.topLevel;

				std::copy_n(outScanParams.lastElement+1u,topLevel,cumulativeWorkgroupCount);
				for (auto i=0u; i<=topLevel; i++)
					cumulativeWorkgroupCount[i] += 1u;
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
                // push constants of the default direct scan pipeline provide both aux memory offset params and scheduling params
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
                        // in case we scan very few elements, you don't want to launch workgroups that wont do anything
			DispatchInfo(const IPhysicalDevice::SLimits& limits, const uint32_t elementCount, const uint32_t workgroupSize)
			{
				constexpr auto workgroupSpinningProtection = 4u; // to prevent first workgroup starving/idling on level 1 after finishing level 0 early
				wg_count = limits.computeOptimalPersistentWorkgroupDispatchSize(elementCount,workgroupSize,workgroupSpinningProtection);
			}

			uint32_t wg_count;
		};

		//
		CScanner(core::smart_refctd_ptr<ILogicalDevice>&& device) : CScanner(std::move(device),core::roundDownToPoT(device->getPhysicalDevice()->getLimits().maxOptimallyResidentWorkgroupInvocations)) {}
		//
		CScanner(core::smart_refctd_ptr<ILogicalDevice>&& device, const uint32_t workgroupSize) : m_device(std::move(device)), m_workgroupSize(workgroupSize)
		{
			assert(core::isPoT(m_workgroupSize));

			const asset::SPushConstantRange pc_range = { asset::IShader::ESS_COMPUTE,0u,sizeof(DefaultPushConstants) };
			const IGPUDescriptorSetLayout::SBinding bindings[2] = {
				{ 0u, asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, video::IGPUShader::ESS_COMPUTE, 1u, nullptr }, // main buffer
				{ 1u, asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, video::IGPUShader::ESS_COMPUTE, 1u, nullptr } // scratch
			};

			m_ds_layout = m_device->createDescriptorSetLayout(bindings,bindings+sizeof(bindings)/sizeof(IGPUDescriptorSetLayout::SBinding));
			m_pipeline_layout = m_device->createPipelineLayout(&pc_range,&pc_range+1,core::smart_refctd_ptr(m_ds_layout));
		}

		//
		inline auto getDefaultDescriptorSetLayout() const { return m_ds_layout.get(); }

		//
		inline auto getDefaultPipelineLayout() const { return m_pipeline_layout.get(); }

		// You need to override this shader with your own defintion of `nbl_glsl_scan_getIndirectElementCount` for it to even compile, so we always give you a new shader
		core::smart_refctd_ptr<asset::ICPUShader> getIndirectShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op) const
		{
			return createShader(true,scanType,dataType,op);
		}

		//
		inline asset::ICPUShader* getDefaultShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op)
		{
			if (!m_shaders[scanType][dataType][op])
				m_shaders[scanType][dataType][op] = createShader(false,scanType,dataType,op);
			return m_shaders[scanType][dataType][op].get();
		}
		//
		inline IGPUSpecializedShader* getDefaultSpecializedShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op)
		{
			if (!m_specialized_shaders[scanType][dataType][op])
			{
				auto cpuShader = core::smart_refctd_ptr<asset::ICPUShader>(getDefaultShader(scanType,dataType,op));
				cpuShader->setFilePathHint("nbl/builtin/glsl/scan/direct.comp");
				cpuShader->setShaderStage(asset::IShader::ESS_COMPUTE);

				auto gpushader = m_device->createShader(std::move(cpuShader));

				m_specialized_shaders[scanType][dataType][op] = m_device->createSpecializedShader(
					gpushader.get(),{nullptr,nullptr,"main"});
				// , asset::IShader::ESS_COMPUTE, "nbl/builtin/glsl/scan/direct.comp"
			}
			return m_specialized_shaders[scanType][dataType][op].get();
		}

		//
		inline auto getDefaultPipeline(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op)
		{
			// ondemand
			if (!m_pipelines[scanType][dataType][op])
				m_pipelines[scanType][dataType][op] = m_device->createComputePipeline(
					nullptr,core::smart_refctd_ptr(m_pipeline_layout),
					core::smart_refctd_ptr<IGPUSpecializedShader>(getDefaultSpecializedShader(scanType,dataType,op))
				);
			return m_pipelines[scanType][dataType][op].get();
		}

		//
		inline uint32_t getWorkgroupSize() const {return m_workgroupSize;}

		//
		inline void buildParameters(const uint32_t elementCount, DefaultPushConstants& pushConstants, DispatchInfo& dispatchInfo)
		{
			pushConstants.schedulerParams = SchedulerParameters(pushConstants.scanParams,elementCount,m_workgroupSize);
			dispatchInfo = DispatchInfo(m_device->getPhysicalDevice()->getLimits(),elementCount,m_workgroupSize);
		}

		//
		static inline void updateDescriptorSet(ILogicalDevice* device, IGPUDescriptorSet* set, const asset::SBufferRange<IGPUBuffer>& input_range, const asset::SBufferRange<IGPUBuffer>& scratch_range)
		{
			IGPUDescriptorSet::SDescriptorInfo infos[2];
			infos[0].desc = input_range.buffer;
			infos[0].info.buffer = {input_range.offset,input_range.size};
			infos[1].desc = scratch_range.buffer;
			infos[1].info.buffer = {scratch_range.offset,scratch_range.size};

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[2];
			for (auto i=0u; i<2u; i++)
			{
				writes[i].dstSet = set;
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				writes[i].info = infos+i;
			}

			device->updateDescriptorSets(2, writes, 0u, nullptr);
		}

		// Half and sizeof(uint32_t) of the scratch buffer need to be cleared to 0s
		static inline void dispatchHelper(
			IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* pipeline_layout, const DefaultPushConstants& pushConstants, const DispatchInfo& dispatchInfo,
			const uint32_t srcBufferBarrierCount, const IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>* srcBufferBarriers,
			const uint32_t dstBufferBarrierCount, const IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>* dstBufferBarriers
		)
		{
			cmdbuf->pushConstants(pipeline_layout,asset::IShader::ESS_COMPUTE,0u,sizeof(DefaultPushConstants),&pushConstants);
			if (srcBufferBarrierCount)
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
				info.bufBarrierCount = srcBufferBarrierCount;
				info.bufBarriers = srcBufferBarriers;
				cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE,info);
			}
			cmdbuf->dispatch(dispatchInfo.wg_count,1u,1u);
			if (srcBufferBarrierCount)
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
				info.bufBarrierCount = dstBufferBarrierCount;
				info.bufBarriers = dstBufferBarriers;
				cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE,info);
			}
		}

		inline ILogicalDevice* getDevice() const {return m_device.get();}

    protected:
		~CScanner()
		{
			// all drop themselves automatically
		}

		core::smart_refctd_ptr<asset::ICPUShader> createShader(const bool indirect, const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op) const;


		core::smart_refctd_ptr<ILogicalDevice> m_device;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> m_ds_layout;
		core::smart_refctd_ptr<IGPUPipelineLayout> m_pipeline_layout;
		core::smart_refctd_ptr<asset::ICPUShader> m_shaders[EST_COUNT][EDT_COUNT][EO_COUNT];
		core::smart_refctd_ptr<IGPUSpecializedShader> m_specialized_shaders[EST_COUNT][EDT_COUNT][EO_COUNT];
		core::smart_refctd_ptr<IGPUComputePipeline> m_pipelines[EST_COUNT][EDT_COUNT][EO_COUNT];
		const uint32_t m_workgroupSize;
};
#endif

}
#endif
