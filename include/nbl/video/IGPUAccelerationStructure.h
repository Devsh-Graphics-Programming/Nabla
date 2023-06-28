// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "nbl/asset/IAccelerationStructure.h"

#include "nbl/video/IGPUBuffer.h"

#include "nbl/builtin/hlsl/acceleration_structures.hlsl"


namespace nbl::video
{

class IGPUAccelerationStructure : public IBackendObject
{
		using PseudoBase = asset::IAccelerationStructure;

	public:
		struct SCreationParams
		{
			asset::SBufferRange<IGPUBuffer> bufferRange;
			core::bitflag<PseudoBase::CREATE_FLAGS> flags = PseudoBase::CREATE_FLAGS::NONE;
		};

		// a few aliases to make usage simpler
		using CREATE_FLAGS = PseudoBase::CREATE_FLAGS;
		using BUILD_FLAGS = PseudoBase::BUILD_FLAGS;
		using GEOMETRY_FLAGS = PseudoBase::GEOMETRY_FLAGS;

		//! builds
		template<typename BufferType>
		struct BuildInfo
		{
			PseudoBase::BUILD_FLAGS				flags : 15 = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
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
			const IGPUAccelerationStructure* src;
			IGPUAccelerationStructure* dst;
			COPY_MODE copyMode;
		};
		template<typename BufferType>
		struct CopyToMemoryInfo
		{
			const IGPUAccelerationStructure* src;
			asset::SBufferBinding<BufferType> dst;
			COPY_MODE copyMode;
		};
		using DeviceCopyToMemoryInfo = CopyToMemoryInfo<IGPUBuffer>;
		using HostCopyToMemoryInfo = CopyToMemoryInfo<asset::ICPUBuffer>;
		template<typename BufferType>
		struct CopyFromMemoryInfo
		{
			asset::SBufferBinding<const BufferType> src;
			IGPUAccelerationStructure* dst;
			COPY_MODE copyMode;
		};
		using DeviceCopyFromMemoryInfo = CopyFromMemoryInfo<IGPUBuffer>;
		using HostCopyFromMemoryInfo = CopyFromMemoryInfo<asset::ICPUBuffer>;

	protected:
		IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : IBackendObject(std::move(dev)), m_bufferRange(std::move(params.bufferRange)) {}

		asset::SBufferBinding<IGPUBuffer> m_bufferRange;
};

class IGPUBottomLevelAccelerationStructure : public asset::IBottomLevelAccelerationStructure, public IGPUAccelerationStructure
{
		using PseudoBase = asset::IBottomLevelAccelerationStructure;

	public:
		template<typename BufferType>
		struct BuildGeometryInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			const IGPUBottomLevelAccelerationStructure* srcAS = nullptr;
			IGPUBottomLevelAccelerationStructure* dstAS = nullptr;
			core::SRange<PseudoBase::BuildGeometryInfo<BufferType>> geometries = {};
		};
		using DeviceBuildGeometryInfo = BuildGeometryInfo<IGPUBuffer>;
		using HostBuildGeometryInfo = BuildGeometryInfo<asset::ICPUBuffer>;

		//! Function used for getting the reference to set `Instance::blas` to
		virtual uint64_t getReferenceForDeviceOperations() const = 0;
		virtual uint64_t getReferenceForHostOperations() const = 0;

	protected:
		IGPUBottomLevelAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params)
			: PseudoBase(params.flags), IGPUAccelerationStructure(std::move(dev),std::move(params)) {}
};

class IGPUTopLevelAccelerationStructure : public asset::ITopLevelAccelerationStructure, public IGPUAccelerationStructure
{
		using PseudoBase = asset::ITopLevelAccelerationStructure;

	public:
		struct SCreationParams : IGPUAccelerationStructure::SCreationParams
		{
			// only relevant if `flag` contain `MOTION_BIT`
			uint32_t maxInstanceCount = 0u;
		};

		template<typename BufferType>
		struct BuildGeometryInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			const IGPUTopLevelAccelerationStructure* srcAS = nullptr;
			IGPUTopLevelAccelerationStructure* dstAS = nullptr;
			core::SRange<PseudoBase::BuildGeometryInfo<BufferType>> geometries = {};
		};
		using DeviceBuildGeometryInfo = BuildGeometryInfo<IGPUBuffer>;
		using HostBuildGeometryInfo = BuildGeometryInfo<asset::ICPUBuffer>;

		//! BEWARE, OUR RESOURCE LIFETIME TRACKING DOES NOT WORK ACROSS TLAS->BLAS boundaries with these types of BLAS references!
		using DeviceInstance = PseudoBase::Instance<uint64_t>;
		using HostInstance = PseudoBase::Instance<uint64_t>;
		using DeviceStaticInstance = PseudoBase::StaticInstance<uint64_t>;
		using HostStaticInstance = PseudoBase::StaticInstance<uint64_t>;
		using DeviceMatrixMotionInstance = PseudoBase::MatrixMotionInstance<uint64_t>;
		using HostMatrixMotionInstance = PseudoBase::MatrixMotionInstance<uint64_t>;
		using DeviceSRTMotionInstance = PseudoBase::SRTMotionInstance<uint64_t>;
		using HostSRTMotionInstance = PseudoBase::SRTMotionInstance<uint64_t>;

	protected:
		IGPUTopLevelAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params)
			: PseudoBase(params.flags), IGPUAccelerationStructure(std::move(dev),std::move(params)) {}
};

}

#endif