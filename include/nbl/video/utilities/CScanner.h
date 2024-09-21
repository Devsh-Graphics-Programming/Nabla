// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_SCANNER_H_INCLUDED_
#define _NBL_VIDEO_C_SCANNER_H_INCLUDED_


#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/IDescriptorSetCache.h"
#include "nbl/video/utilities/CArithmeticOps.h"

#include "nbl/builtin/hlsl/scan/declarations.hlsl"
static_assert(NBL_BUILTIN_MAX_LEVELS & 0x1, "NBL_BUILTIN_MAX_LEVELS must be odd!");

namespace nbl::video
{

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
class CScanner final : public CArithmeticOps
{
	public:		
		enum E_SCAN_TYPE : uint8_t
		{
			 // computes output[n] = Sum_{i<=n}(input[i])
			 EST_INCLUSIVE = 0u,
			 // computes output[n] = Sum_{i<n}(input[i]), meaning first element is identity
			 EST_EXCLUSIVE,
			 EST_COUNT
		};
		
		// You need to override this shader with your own defintion of `nbl_glsl_scan_getIndirectElementCount` for it to even compile, so we always give you a new shader
		//core::smart_refctd_ptr<asset::ICPUShader> getIndirectShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz) const
		//{
		//	return createShader(true,scanType,dataType,op,scratchSz);
		//}

		//
		CScanner(core::smart_refctd_ptr<ILogicalDevice>&& device) : CScanner(std::move(device), core::roundDownToPoT(device->getPhysicalDevice()->getLimits().maxOptimallyResidentWorkgroupInvocations)) {}
		CScanner(core::smart_refctd_ptr<ILogicalDevice>&& device, const uint32_t workgroupSize) : CArithmeticOps(core::smart_refctd_ptr(device), workgroupSize) {}
		asset::ICPUShader* getDefaultShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz);
		IGPUShader* getDefaultSpecializedShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz);
		IGPUComputePipeline* getDefaultPipeline(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz);
        core::smart_refctd_ptr<asset::ICPUShader> createShader(/*const bool indirect, */const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz) const;
    protected:
		~CScanner()
		{
			// all drop themselves automatically
		}

		core::smart_refctd_ptr<asset::ICPUShader> m_shaders[EST_COUNT][EDT_COUNT][EO_COUNT];
		core::smart_refctd_ptr < IGPUShader > m_specialized_shaders[EST_COUNT][EDT_COUNT][EO_COUNT];
		core::smart_refctd_ptr<IGPUComputePipeline> m_pipelines[EST_COUNT][EDT_COUNT][EO_COUNT];
};

}
#endif
