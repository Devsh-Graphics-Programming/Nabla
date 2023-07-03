// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "nbl/asset/IAccelerationStructure.h"

#include "nbl/video/IDeferredOperation.h"
#include "nbl/video/IGPUBuffer.h"

#include "nbl/builtin/hlsl/acceleration_structures.hlsl"


namespace nbl::video
{

class IGPUAccelerationStructure : public asset::IAccelerationStructure, public IBackendObject
{
	public:
		struct SCreationParams
		{
			asset::SBufferRange<IGPUBuffer> bufferRange;
			core::bitflag<CREATE_FLAGS> flags = CREATE_FLAGS::NONE;
		};
		inline const auto& getBufferRange() const {return m_bufferRange;}

		//! builds
		template<class BufferType>
		struct BuildInfo
		{
			public:
				BUILD_FLAGS							flags : 15 = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
				// implicitly satisfies:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-mode-04628
				uint8_t								isUpdate : 1 = false;
				asset::SBufferBinding<BufferType>	scratchAddr = {};

			protected:
				// List of things too expensive or impossible (without GPU Assist) to validate:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03403
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-dstAccelerationStructure-03698
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03663
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03664
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-dstAccelerationStructure-03701
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-dstAccelerationStructure-03702
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-dstAccelerationStructure-03703
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-scratchData-03704
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-scratchData-03705
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03667
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03758
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03759
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03761
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03762
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03763
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03764
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03765
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03766
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03767
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03768
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03769
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03770
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03675
				bool invalid(const IGPUAccelerationStructure* const src, const IGPUAccelerationStructure* const dst) const;
		};
		
		// for indirect builds
		using BuildRangeInfo = hlsl::acceleration_structures::BuildRangeInfo;

		// copies
		enum class COPY_MODE : uint8_t
		{
			CLONE = 0,
			COMPACT = 1,
			SERIALIZE = 2,
			DESERIALIZE = 3,
		};
		struct CopyInfo
		{
			const IGPUAccelerationStructure* src = nullptr;
			IGPUAccelerationStructure* dst = nullptr;
			COPY_MODE mode = COPY_MODE::CLONE;
		};
		template<typename BufferType>
		struct CopyToMemoryInfo
		{
			const IGPUAccelerationStructure* src = nullptr;
			asset::SBufferBinding<BufferType> dst = nullptr;
			COPY_MODE mode = COPY_MODE::SERIALIZE;
		};
		using DeviceCopyToMemoryInfo = CopyToMemoryInfo<IGPUBuffer>;
		using HostCopyToMemoryInfo = CopyToMemoryInfo<asset::ICPUBuffer>;
		template<typename BufferType>
		struct CopyFromMemoryInfo
		{
			asset::SBufferBinding<const BufferType> src = nullptr;
			IGPUAccelerationStructure* dst = nullptr;
			COPY_MODE mode = COPY_MODE::DESERIALIZE;
		};
		using DeviceCopyFromMemoryInfo = CopyFromMemoryInfo<IGPUBuffer>;
		using HostCopyFromMemoryInfo = CopyFromMemoryInfo<asset::ICPUBuffer>;

		// this will return false also if your deferred operation is not ready yet, so please use in combination with `isPending()`
		virtual bool wasCopySuccessful(const IDeferredOperation* const deferredOp) = 0;

		// Vulkan const VkAccelerationStructureKHR*
		virtual const void* getNativeHandle() const = 0;

	protected:
		inline IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params)
			: asset::IAccelerationStructure(params.flags), IBackendObject(std::move(dev)), m_bufferRange(std::move(params.bufferRange)) {}

		asset::SBufferRange<IGPUBuffer> m_bufferRange;
};
template class IGPUAccelerationStructure::BuildInfo<IGPUBuffer>;
template class IGPUAccelerationStructure::BuildInfo<asset::ICPUBuffer>;

// strong typing of acceleration structures implicitly satifies:
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-None-03407
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03699
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03700
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03760

class IGPUBottomLevelAccelerationStructure : public asset::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>
{
	public:
		template<class BufferType>
		struct BuildInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			// Returns 0 on failure, otherwise returns the number of `core::smart_refctd_ptr` to reserve for lifetime tracking
			// When validating for indirect builds pass `nullptr` as the argument
			uint32_t valid(const BuildRangeInfo* const buildRangeInfos) const;

			const IGPUBottomLevelAccelerationStructure* srcAS = nullptr;
			IGPUBottomLevelAccelerationStructure* dstAS = nullptr;
			core::SRange<const Geometry<BufferType>> geometries = {nullptr,nullptr};
		};
		using DeviceBuildInfo = BuildInfo<IGPUBuffer>;
		using HostBuildInfo = BuildInfo<asset::ICPUBuffer>;

		//! Fill your `ITopLevelAccelerationStructure::BuildGeometryInfo<IGPUBuffer>::instanceData` with `ITopLevelAccelerationStructure::Instance<device_op_ref_t>`
		struct device_op_ref_t
		{
			uint64_t deviceAddress;
		};
		virtual device_op_ref_t getReferenceForDeviceOperations() const = 0;

		//! Fill your `ITopLevelAccelerationStructure::BuildGeometryInfo<ICPUBuffer>::instanceData` with `ITopLevelAccelerationStructure::Instance<host_op_ref_t>`
		struct host_op_ref_t
		{
			uint64_t apiHandle;
		};
		virtual host_op_ref_t getReferenceForHostOperations() const = 0;

	protected:
		using asset::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>;
};
template class IGPUBottomLevelAccelerationStructure::BuildInfo<IGPUBuffer>;
template class IGPUBottomLevelAccelerationStructure::BuildInfo<asset::ICPUBuffer>;

class IGPUTopLevelAccelerationStructure : public asset::ITopLevelAccelerationStructure<IGPUAccelerationStructure>
{
	public:
		struct SCreationParams : IGPUAccelerationStructure::SCreationParams
		{
			// only relevant if `flag` contain `MOTION_BIT`
			uint32_t maxInstanceCount = 0u;
		};

		template<typename BufferType>
		struct BuildInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			// Returns 0 on failure, otherwise returns the number of `core::smart_refctd_ptr` to reserve for lifetime tracking
			// When validating for indirect builds pass `nullptr` as the argument
			// List of things too expensive or impossible (without GPU Assist) to validate:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-dstAccelerationStructure-03706
			uint32_t valid(const uint32_t* const instanceCounts) const;

			const IGPUTopLevelAccelerationStructure* srcAS = nullptr;
			IGPUTopLevelAccelerationStructure* dstAS = nullptr;
			core::SRange<const Geometry<BufferType>> geometries = {nullptr,nullptr};
		};
		using DeviceBuildInfo = BuildInfo<IGPUBuffer>;
		using HostBuildInfo = BuildInfo<asset::ICPUBuffer>;

		//! BEWARE, OUR RESOURCE LIFETIME TRACKING DOES NOT WORK ACROSS TLAS->BLAS boundaries with these types of BLAS references!
		// TODO: Investigate `EXT_private_data` to be able to go ` -> IGPUBottomLevelAccelerationStructure`
		using DeviceInstance = Instance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostInstance = Instance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		using DeviceStaticInstance = StaticInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostStaticInstance = StaticInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		using DeviceMatrixMotionInstance = MatrixMotionInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostMatrixMotionInstance = MatrixMotionInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		using DeviceSRTMotionInstance = SRTMotionInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostSRTMotionInstance = SRTMotionInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;

	protected:
		inline IGPUTopLevelAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params)
			: asset::ITopLevelAccelerationStructure<IGPUAccelerationStructure>(std::move(dev),std::move(params)),
			m_maxInstanceCount(getCreationFlags().hasFlags(CREATE_FLAGS::MOTION_BIT) ? params.maxInstanceCount:(~0u)) {}

		const uint32_t m_maxInstanceCount;
};
template class IGPUTopLevelAccelerationStructure::BuildInfo<IGPUBuffer>;
template class IGPUTopLevelAccelerationStructure::BuildInfo<asset::ICPUBuffer>;

}

#endif