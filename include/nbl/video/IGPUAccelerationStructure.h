// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "nbl/asset/IAccelerationStructure.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/video/IGPUBuffer.h"

namespace nbl::video
{
class NBL_API IGPUAccelerationStructure : public asset::IAccelerationStructure, public IBackendObject
{
	using Base = asset::IAccelerationStructure;
	public:
		
		struct SCreationParams
		{
			E_CREATE_FLAGS					flags;
			Base::E_TYPE					type;
			asset::SBufferRange<IGPUBuffer> bufferRange;
			bool operator==(const SCreationParams& rhs) const
			{
				return flags == rhs.flags && type == rhs.type && bufferRange == rhs.bufferRange;
			}
			bool operator!=(const SCreationParams& rhs) const
			{
				return !operator==(rhs);
			}
		};
		
		using DeviceAddressType = asset::SBufferBinding<IGPUBuffer>;
		using HostAddressType = asset::SBufferBinding<asset::ICPUBuffer>;

		template<typename AddressType>
		struct BuildGeometryInfo
		{
			using Geometry = IAccelerationStructure::Geometry<AddressType>;
			BuildGeometryInfo() 
				: type(static_cast<Base::E_TYPE>(0u))
				, buildFlags(static_cast<E_BUILD_FLAGS>(0u))
				, buildMode(static_cast<E_BUILD_MODE>(0u))
				, srcAS(nullptr)
				, dstAS(nullptr)
				, geometries(core::SRange<Geometry>(nullptr, nullptr))
				, scratchAddr({})
			{}
			~BuildGeometryInfo() = default;
			Base::E_TYPE	type; // TODO: Can deduce from creationParams.type?
			E_BUILD_FLAGS	buildFlags;
			E_BUILD_MODE	buildMode;
			IGPUAccelerationStructure * srcAS;
			IGPUAccelerationStructure * dstAS;
			core::SRange<Geometry> geometries;
			AddressType		scratchAddr;
		};

		using HostBuildGeometryInfo = BuildGeometryInfo<HostAddressType>;
		using DeviceBuildGeometryInfo = BuildGeometryInfo<DeviceAddressType>;

		struct BuildSizes
		{
			uint64_t accelerationStructureSize;
			uint64_t updateScratchSize;
			uint64_t buildScratchSize;
		};

		struct CopyInfo
		{
			IGPUAccelerationStructure * src;
			IGPUAccelerationStructure * dst;
			E_COPY_MODE copyMode;
		};
		
		template<typename AddressType>
		struct CopyToMemoryInfo
		{
			IGPUAccelerationStructure * src;
			AddressType dst;
			E_COPY_MODE copyMode;
		};
		using DeviceCopyToMemoryInfo = CopyToMemoryInfo<DeviceAddressType>;
		using HostCopyToMemoryInfo = CopyToMemoryInfo<HostAddressType>;

		
		template<typename AddressType>
		struct CopyFromMemoryInfo
		{
			AddressType src;
			IGPUAccelerationStructure * dst;
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

		// Function used for getting the reference to give 'Instance' as a parameter
		virtual uint64_t getReferenceForDeviceOperations() const = 0;
		virtual uint64_t getReferenceForHostOperations() const = 0;

	protected:
		IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& _params) 
			: IBackendObject(std::move(dev)) 
			, params(std::move(_params))
		{}
		virtual ~IGPUAccelerationStructure() = default;
	private:
		 SCreationParams params;
};
}

#endif