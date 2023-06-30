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
		template<typename BufferType>
		struct BuildInfo
		{
			inline bool valid() const
			{
				return true;
			}

			BUILD_FLAGS							flags : 15 = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
			uint8_t								isUpdate : 1 = false;
			asset::SBufferBinding<BufferType>	scratchAddr = {};
		};
		
		// for indirect builds
		using BuildRangeInfo = hlsl::acceleration_structures::BuildRangeInfo;

		// returned by ILogicalDevice
		struct BuildSizes
		{
			size_t accelerationStructureSize = 0ull;
			size_t updateScratchSize = 0ull;
			size_t buildScratchSize = 0ull;
		};

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
		IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params)
			: asset::IAccelerationStructure(params.flags), IBackendObject(std::move(dev)), m_bufferRange(std::move(params.bufferRange)) {}

		asset::SBufferRange<IGPUBuffer> m_bufferRange;
};

class IGPUBottomLevelAccelerationStructure : public asset::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>
{
	public:
		template<typename BufferType>
		struct BuildGeometryInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			const IGPUBottomLevelAccelerationStructure* srcAS = nullptr;
			IGPUBottomLevelAccelerationStructure* dstAS = nullptr;
			core::SRange<BuildGeometryInfo<BufferType>> geometries = {};
		};
		using DeviceBuildGeometryInfo = BuildGeometryInfo<IGPUBuffer>;
		using HostBuildGeometryInfo = BuildGeometryInfo<asset::ICPUBuffer>;

		//! Function used for getting the reference to set `Instance::blas` to
		virtual uint64_t getReferenceForDeviceOperations() const = 0;
		virtual uint64_t getReferenceForHostOperations() const = 0;

	protected:
		using asset::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>;
};

class IGPUTopLevelAccelerationStructure : public asset::ITopLevelAccelerationStructure<IGPUAccelerationStructure>
{
	public:
		struct SCreationParams : IGPUAccelerationStructure::SCreationParams
		{
			// only relevant if `flag` contain `MOTION_BIT`
			uint32_t maxInstanceCount = 0u;
		};

		template<typename BufferType>
		struct BuildGeometryInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			inline bool valid() const
			{
				return true;
			}

			const IGPUTopLevelAccelerationStructure* srcAS = nullptr;
			IGPUTopLevelAccelerationStructure* dstAS = nullptr;
			core::SRange<BuildGeometryInfo<BufferType>> geometries = {};
		};
		using DeviceBuildGeometryInfo = BuildGeometryInfo<IGPUBuffer>;
		using HostBuildGeometryInfo = BuildGeometryInfo<asset::ICPUBuffer>;

		//! BEWARE, OUR RESOURCE LIFETIME TRACKING DOES NOT WORK ACROSS TLAS->BLAS boundaries with these types of BLAS references!
		using DeviceInstance = Instance<uint64_t>;
		using HostInstance = Instance<uint64_t>;
		using DeviceStaticInstance = StaticInstance<uint64_t>;
		using HostStaticInstance = StaticInstance<uint64_t>;
		using DeviceMatrixMotionInstance = MatrixMotionInstance<uint64_t>;
		using HostMatrixMotionInstance = MatrixMotionInstance<uint64_t>;
		using DeviceSRTMotionInstance = SRTMotionInstance<uint64_t>;
		using HostSRTMotionInstance = SRTMotionInstance<uint64_t>;

	protected:
		using asset::ITopLevelAccelerationStructure<IGPUAccelerationStructure>::ITopLevelAccelerationStructure<IGPUAccelerationStructure>;
};

}

#endif