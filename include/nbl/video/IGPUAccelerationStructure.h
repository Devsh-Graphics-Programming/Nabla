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
	public:		
		//! builds
		template<typename BufferType>
		struct BuildGeometryInfo
		{
			using Geometry = IAccelerationStructure::Geometry<BufferType>;

			BUILD_FLAGS flags : 7 = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
			uint8_t isUpdate : 1 = false;
			IGPUAccelerationStructure* srcAS = nullptr;
			IGPUAccelerationStructure* dstAS = nullptr;
			core::SRange<Geometry> geometries = {};
			AddressType	scratchAddr = {};
		};
		using HostBuildGeometryInfo = BuildGeometryInfo<HostAddressType>;
		using DeviceBuildGeometryInfo = BuildGeometryInfo<DeviceAddressType>;
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
		
		enum E_COPY_MODE
		{
			ECM_CLONE = 0,
			ECM_COMPACT = 1,
			ECM_SERIALIZE = 2,
			ECM_DESERIALIZE = 3,
		};
		struct CopyInfo
		{
			const IGPUAccelerationStructure* src;
			IGPUAccelerationStructure* dst;
			E_COPY_MODE copyMode;
		};
		
		template<typename AddressType>
		struct CopyToMemoryInfo
		{
			const IGPUAccelerationStructure* src;
			AddressType dst;
			E_COPY_MODE copyMode;
		};
		using DeviceCopyToMemoryInfo = CopyToMemoryInfo<DeviceAddressType>;
		using HostCopyToMemoryInfo = CopyToMemoryInfo<HostAddressType>;

		
		template<typename AddressType>
		struct CopyFromMemoryInfo
		{
			AddressType src;
			IGPUAccelerationStructure* dst;
			E_COPY_MODE copyMode;
		};
		using DeviceCopyFromMemoryInfo = CopyFromMemoryInfo<DeviceAddressType>;
		using HostCopyFromMemoryInfo = CopyFromMemoryInfo<HostAddressType>;

		inline const auto& getCreationParameters() const
		{
			return params;
		}

		//!
		inline static bool validateCreationParameters(const SCreationParams& _params)
		{
			if(!_params.bufferRange.isValid()) {
				return false;
			}
			return true;
		}


		//! Function used for getting the reference to give 'Instance' as a parameter
		virtual uint64_t getReferenceForDeviceOperations() const = 0;
		virtual uint64_t getReferenceForHostOperations() const = 0;

	protected:
		virtual ~IGPUAccelerationStructure() = default;
};

class IGPUBottomLevelAccelerationStructure : public asset::IBottomLevelAccelerationStructure, public IGPUAccelerationStructure
{
		using Base = asset::IAccelerationStructure<IGPUAccelerationStructure>;

	public:		
		//! builds
		template<typename AddressType>
		struct BuildGeometryInfo
		{
			using Geometry = IAccelerationStructure::Geometry<AddressType>;

			FLAGS flags : 7 = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
			uint8_t isUpdate : 1 = false;
			IGPUAccelerationStructure* srcAS = nullptr;
			IGPUAccelerationStructure* dstAS = nullptr;
			core::SRange<Geometry> geometries = {};
			AddressType	scratchAddr = {};
		};
		using HostBuildGeometryInfo = BuildGeometryInfo<HostAddressType>;
		using DeviceBuildGeometryInfo = BuildGeometryInfo<DeviceAddressType>;
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
		
		enum E_COPY_MODE
		{
			ECM_CLONE = 0,
			ECM_COMPACT = 1,
			ECM_SERIALIZE = 2,
			ECM_DESERIALIZE = 3,
		};
		struct CopyInfo
		{
			const IGPUAccelerationStructure* src;
			IGPUAccelerationStructure* dst;
			E_COPY_MODE copyMode;
		};
		
		template<typename AddressType>
		struct CopyToMemoryInfo
		{
			const IGPUAccelerationStructure* src;
			AddressType dst;
			E_COPY_MODE copyMode;
		};
		using DeviceCopyToMemoryInfo = CopyToMemoryInfo<DeviceAddressType>;
		using HostCopyToMemoryInfo = CopyToMemoryInfo<HostAddressType>;

		
		template<typename AddressType>
		struct CopyFromMemoryInfo
		{
			AddressType src;
			IGPUAccelerationStructure* dst;
			E_COPY_MODE copyMode;
		};
		using DeviceCopyFromMemoryInfo = CopyFromMemoryInfo<DeviceAddressType>;
		using HostCopyFromMemoryInfo = CopyFromMemoryInfo<HostAddressType>;

		inline const auto& getCreationParameters() const
		{
			return params;
		}

		//!
		inline static bool validateCreationParameters(const SCreationParams& _params)
		{
			if(!_params.bufferRange.isValid()) {
				return false;
			}
			return true;
		}


		//! Function used for getting the reference to give 'Instance' as a parameter
		virtual uint64_t getReferenceForDeviceOperations() const = 0;
		virtual uint64_t getReferenceForHostOperations() const = 0;

	protected:
		IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& _params) : IBackendObject(std::move(dev)), params(std::move(_params)) {}
		virtual ~IGPUAccelerationStructure() = default;

	private:
		 SCreationParams params;
};

}

#endif