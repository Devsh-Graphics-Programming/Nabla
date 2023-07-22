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
			enum class FLAGS : uint8_t
			{
				NONE = 0u,
				//DEVICE_ADDRESS_CAPTURE_REPLAY_BIT	= 0x1u<<0u, for tools only
				// Provided by VK_NV_ray_tracing_motion_blur
				MOTION_BIT = 0x1u << 1u,
			};

			asset::SBufferRange<IGPUBuffer> bufferRange;
			core::bitflag<FLAGS> flags = FLAGS::NONE;
		};
		inline const SCreationParams& getCreationParams() const {return m_params;}

#if 0 // TODO: need a non-refcounting `SBufferBinding` and `SBufferRange` variants first
		//! special binding value which you can fill your Geometry<BufferType> fields with before you call `ILogicalDevice::getAccelerationStructureBuildSizes` 
		template<class BufferType>
		static inline asset::SBufferBinding<const BufferType> getDummyBindingForBuildSizeQuery()
		{
			constexpr size_t Invalid = 0xdeadbeefBADC0FFEull;
			return {reinterpret_cast<const BufferType*>(Invalid),Invalid};
		}
#endif

		//! builds
		template<class BufferType>
		struct BuildInfo
		{
			public:
				asset::SBufferBinding<BufferType>	scratchAddr = {};
				// implicitly satisfies: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-mode-04628
				bool								isUpdate = false;

			protected:
				BuildInfo() = default;
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
		inline IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : IBackendObject(std::move(dev)), m_params(std::move(params)) {}

		const SCreationParams m_params;
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
		// read the comments in the .hlsl file, AABB builds ignore certain fields
		using BuildRangeInfo = hlsl::acceleration_structures::bottom_level::BuildRangeInfo;

		template<class BufferType>
		struct BuildInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			inline uint32_t inputCount() const
			{
				return buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT) ? aabbs.size():triangles.size();
			}

			// Returns 0 on failure, otherwise returns the number of `core::smart_refctd_ptr` to reserve for lifetime tracking
			// When validating for indirect builds pass `nullptr` as the argument
			uint32_t valid(const BuildRangeInfo* const buildRangeInfos) const;

			core::bitflag<BUILD_FLAGS> buildFlags = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
			const IGPUBottomLevelAccelerationStructure* srcAS = nullptr;
			IGPUBottomLevelAccelerationStructure* dstAS = nullptr;
			// please interpret based on `buildFlags.hasFlags(GEOMETRY_TYPE_IS_AABB_BIT)`
			union
			{
				core::SRange<const Triangles<BufferType>> triangles = {nullptr,nullptr};
				core::SRange<const AABBs<BufferType>> aabbs;
			};
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
		//
		inline uint32_t getMaxInstanceCount() const {return m_maxInstanceCount;}

		// read the comments in the .hlsl file, TLAS builds ignore certain fields
		using BuildRangeInfo = hlsl::acceleration_structures::top_level::BuildRangeInfo;

		template<typename BufferType>
		struct BuildInfo : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			inline uint32_t inputCount() const {return 1u;}

			// When validating for indirect builds pass the maximum possible instance count as the argument
			// List of things too expensive or impossible (without GPU Assist) to validate:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-dstAccelerationStructure-03706
			inline bool valid(const uint32_t instanceCount) const
			{
				if (IGPUAccelerationStructure::BuildInfo<BufferType>::invalid(srcAS,dstAS))
					return false;
				if (instanceCount>dstAs->getMaxInstanceCount())
					return false;
				
				constexpr bool HostBuild = std::is_same_v<BufferType,asset::ICPUBuffer>;
				// I'm not gonna do the `std::conditional_t<HostBuild,,>` to get the correct Instance struct type as they're the same size essentially
				const size_t instanceSize = buildFlags.hasFlags(BUILD_FLAGS::INSTANCE_TYPE_ENCODED_IN_POINTER_LSB) ? sizeof(void*):(
					dstAs->getCreationParams().flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT) ? sizeof(DevicePolymorphicInstance):sizeof(DeviceStaticInstance)
				);
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03778
				if (!instanceData.isValid())
					return false;
				else if (instanceData.offset+instanceSize*instanceCount>instanceData.buffer->getSize())
					return false;

				#ifdef _NBL_DEBUG
				/* TODO: with `EXT_private_data
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03724
				if constexpr (HostBuild)
				{
					for (auto 
					if (device->invalidAccelerationStructureForHostOperations(getAccelerationStructureFromReference(geometry.instanceData.blas)))
						return false;
				}
				*/
				#endif

				// destination, scratch and instanceData are required, source is optional
				return isUpdate ? 4u:3u;
			}
			
			core::bitflag<BUILD_FLAGS> buildFlags = BUILD_FLAGS::PREFER_FAST_BUILD_BIT;
			const IGPUTopLevelAccelerationStructure* srcAS = nullptr;
			IGPUTopLevelAccelerationStructure* dstAS = nullptr;
			// depending on the presence certain bits in `buildFlags` this buffer will be filled with:
			// - addresses to `StaticInstance`, `MatrixMotionInstance`, `SRTMotionInstance` packed in upper 60 bits and struct type in lower 4 bits if and only if `buildFlags.hasFlags(INSTANCE_TYPE_ENCODED_IN_POINTER_LSB)`, otherwise
			// - an array of `PolymorphicInstance` if our `SCreationParams::flags.hasFlags(MOTION_BIT)`, otherwise
			// - an array of `StaticInstance`
			asset::SBufferBinding<const BufferType> instanceData = {};
		};
		using DeviceBuildInfo = BuildInfo<IGPUBuffer>;
		using HostBuildInfo = BuildInfo<asset::ICPUBuffer>;

		//! BEWARE, OUR RESOURCE LIFETIME TRACKING DOES NOT WORK ACROSS TLAS->BLAS boundaries with these types of BLAS references!
		// TODO: Investigate `EXT_private_data` to be able to go ` -> IGPUBottomLevelAccelerationStructure`
		using DeviceInstance = Instance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostInstance = Instance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		static_assert(sizeof(DeviceInstance)==sizeof(HostInstance));
		// other typedefs for convenience
		using DeviceStaticInstance = StaticInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostStaticInstance = StaticInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		static_assert(sizeof(DeviceStaticInstance)==sizeof(HostStaticInstance));
		using DeviceMatrixMotionInstance = MatrixMotionInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostMatrixMotionInstance = MatrixMotionInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		static_assert(sizeof(DeviceMatrixMotionInstance)==sizeof(HostMatrixMotionInstance));
		using DeviceSRTMotionInstance = SRTMotionInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostSRTMotionInstance = SRTMotionInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		static_assert(sizeof(DeviceSRTMotionInstance)==sizeof(HostSRTMotionInstance));

		// defined exactly as Vulkan wants it, byte for byte
		template<typename blas_ref_t>
		struct PolymorphicInstance final
		{
				// make sure we're not trying to memcpy smartpointers
				static_assert(!std::is_same_v<core::smart_refctd_ptr<const asset::ICPUBottomLevelAccelerationStructure>,blas_ref_t>);

			public:
				PolymorphicInstance() = default;
				PolymorphicInstance(const PolymorphicInstance<blas_ref_t>&) = default;
				PolymorphicInstance(PolymorphicInstance<blas_ref_t>&&) = default;

				// I made all these assignment operators because of the `core::matrix3x4SIMD` alignment and keeping `type` correct at all times
				inline PolymorphicInstance<blas_ref_t>& operator=(const StaticInstance<blas_ref_t>& _static)
				{
					type = INSTANCE_TYPE::STATIC;
					memcpy(&largestUnionMember,&_static,sizeof(_static));
					return *this;
				}
				inline PolymorphicInstance<blas_ref_t>& operator=(const MatrixMotionInstance<blas_ref_t>& matrixMotion)
				{
					type = INSTANCE_TYPE::MATRIX_MOTION;
					memcpy(&largestUnionMember,&matrixMotion,sizeof(matrixMotion));
					return *this;
				}
				inline PolymorphicInstance<blas_ref_t>& operator=(const SRTMotionInstance<blas_ref_t>& srtMotion)
				{
					type = INSTANCE_TYPE::SRT_MOTION;
					largestUnionMember = srtMotion;
					return *this;
				}

				inline INSTANCE_TYPE getType() const {return type;}

				template<template<class> class InstanceT>
				inline InstanceT<blas_ref_t> copy() const
				{
					InstanceT<blas_ref_t> retval;
					memcpy(&retval,largestUnionMember,sizeof(retval));
					return retval;
				}

			private:
				INSTANCE_TYPE type = INSTANCE_TYPE::STATIC;
				static_assert(std::is_same_v<std::underlying_type_t<INSTANCE_TYPE>,uint32_t>);
				// these must be 0 as per vulkan spec
				uint32_t reservedMotionFlags = 0u;
				// I don't do an actual union because the preceeding members don't play nicely with alignment of `core::matrix3x4SIMD` and Vulkan requires this struct to be packed
				SRTMotionInstance<blas_ref_t> largestUnionMember = {};
				static_assert(alignof(largestUnionMember)==8ull);
		};
		using DevicePolymorphicInstance = PolymorphicInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostPolymorphicInstance = PolymorphicInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		static_assert(sizeof(DevicePolymorphicInstance)==sizeof(HostPolymorphicInstance));

	protected:
		inline IGPUTopLevelAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params)
			: asset::ITopLevelAccelerationStructure<IGPUAccelerationStructure>(std::move(dev),std::move(params)), m_maxInstanceCount(params.maxInstanceCount) {}

		const uint32_t m_maxInstanceCount;
};
template class IGPUTopLevelAccelerationStructure::BuildInfo<IGPUBuffer>;
template class IGPUTopLevelAccelerationStructure::BuildInfo<asset::ICPUBuffer>;

}

#endif