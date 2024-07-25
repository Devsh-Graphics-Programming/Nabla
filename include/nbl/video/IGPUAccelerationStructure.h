// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "nbl/asset/asset.h"

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
				asset::SBufferBinding<BufferType>	scratch = {};
				// implicitly satisfies: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-mode-04628
				bool								isUpdate = false;

			protected:
				inline BuildInfo() = default;
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
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03769
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03675
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03726
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03651
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03652
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03653
				NBL_API2 bool invalid(const IGPUAccelerationStructure* const src, const IGPUAccelerationStructure* const dst) const;

				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673
				static inline bool invalidInputBuffer(const asset::SBufferBinding<const BufferType>& binding, const size_t byteOffset, const size_t count, const size_t elementSize, const size_t alignment)
				{
					if (!binding.buffer || binding.offset+byteOffset+count*elementSize<binding.buffer->getSize())
						return true;

					if constexpr (std::is_same_v<BufferType,IGPUBuffer>)
					{
						const auto deviceAddr = binding.buffer->getDeviceAddress();
						if (!deviceAddr==0ull || !core::is_aligned_to(deviceAddr,alignment))
							return true;

						if (!binding.buffer->getCreationParams().usage.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT))
							return true;
					}

					return false;
				}
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

		// this will return false also if your deferred operation is not ready yet, so please use in combination with `isPending()`
		virtual bool wasBuildSuccessful(const IDeferredOperation* const deferredOp) = 0;

		// Vulkan const VkAccelerationStructureKHR*
		virtual const void* getNativeHandle() const = 0;

	protected:
		inline IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : IBackendObject(std::move(dev)), m_params(std::move(params)) {}

		const SCreationParams m_params;
};

// strong typing of acceleration structures implicitly satifies:
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-None-03407
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03699
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03700
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03760
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03789
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03790

class IGPUBottomLevelAccelerationStructure : public asset::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>
{
		using Base = asset::IBottomLevelAccelerationStructure<IGPUAccelerationStructure>;

	public:
		static inline bool validBuildFlags(const core::bitflag<BUILD_FLAGS> flags, const SPhysicalDeviceFeatures& enabledFeatures)
		{
			if (!Base::validBuildFlags(flags))
				return false;
			/* TODO
			if (flags.hasFlags(build_flags_t::ALLOW_OPACITY_MICROMAP_UPDATE_BIT|build_flags_t::ALLOW_DISABLE_OPACITY_MICROMAPS_BIT|build_flags_t::ALLOW_OPACITY_MICROMAP_DATA_UPDATE_BIT) && !enabledFeatures.??????????)
				return false;
			if (flags.hasFlags(build_flags_t::ALLOW_DISPLACEMENT_MICROMAP_UPDATE_BIT) && !enabledFeatures.???????????)
				return false;
			if (flags.hasFlags(build_flags_t::ALLOW_DATA_ACCESS) && !enabledFeatures.???????????)
				return false;
			*/
			return true;
		}

		// read the comments in the .hlsl file, AABB builds ignore certain fields
		using BuildRangeInfo = hlsl::acceleration_structures::bottom_level::BuildRangeInfo;
		using DirectBuildRangeRangeInfos = const BuildRangeInfo* const*;
		using MaxInputCounts = const uint32_t* const;

		template<class BufferType>
		struct BuildInfo final : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			private:
				using Base = IGPUAccelerationStructure::BuildInfo<BufferType>;

			public:
				inline uint32_t inputCount() const {return geometryCount;}

				// Returns 0 on failure, otherwise returns the number of `core::smart_refctd_ptr` to reserve for lifetime tracking
				// List of things too expensive or impossible (without GPU Assist) to validate:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03763
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03764
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03765
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03766
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03767
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03768
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03770
				template<typename T>// requires nbl::is_any_of_v<T,std::conditional_t<std::is_same_v<BufferType,IGPUBuffer>,uint32_t,BuildRangeInfo>,BuildRangeInfo>
				NBL_API2 uint32_t valid(const T* const buildRangeInfosOrMaxPrimitiveCounts) const;

				// really expensive to call, `valid` only calls it when `_NBL_DEBUG` is defined
				inline bool validGeometry(size_t& totalPrims, const AABBs<const BufferType>& geometry, const BuildRangeInfo& buildRangeInfo) const
				{
					constexpr size_t AABBalignment = 8ull;
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03659
					if (!core::is_aligned_to(buildRangeInfo.primitiveByteOffset,AABBalignment))
						return false;
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03811
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03812
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03814
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03774
					if (Base::invalidInputBuffer(geometry.data,buildRangeInfo.primitiveByteOffset,buildRangeInfo.primitiveCount,sizeof(AABB_t),AABBalignment))
						return false;
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545
					if (!core::is_aligned_to(geometry.stride,AABBalignment))
						return false;

					totalPrims += buildRangeInfo.primitiveCount;
					return true;
				}
				inline bool validGeometry(size_t& totalPrims, const Triangles<const BufferType>& geometry, const BuildRangeInfo& buildRangeInfo) const
				{
					//
					if (!dstAS->validVertexFormat(geometry.vertexFormat))
						return false;

					const size_t vertexSize = asset::getTexelOrBlockBytesize(geometry.vertexFormat);
					// TODO: improve in line with the spec https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03711
					const size_t vertexAlignment = core::max(vertexSize/4u,1ull);
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexStride-03735
					if (!core::is_aligned_to(geometry.vertexStride,vertexAlignment))
						return false;

					//
					const bool hasMotion = geometry.vertexData[1].buffer && dstAS->getCreationParams().flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT);

					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03712
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03806
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03807
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03771
					// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03772
					const size_t indexCount = 3ull*buildRangeInfo.primitiveCount;
					const size_t firstVertexByteOffset = size_t(buildRangeInfo.firstVertex)*geometry.vertexStride;
					if (geometry.indexType!=asset::EIT_UNKNOWN)
					{
						const size_t indexSize = geometry.indexType==asset::EIT_16BIT ? sizeof(uint16_t):sizeof(uint32_t);
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03656
						if (!core::is_aligned_to(buildRangeInfo.primitiveByteOffset,indexSize))
							return false;
						if (Base::invalidInputBuffer(geometry.indexData,buildRangeInfo.primitiveByteOffset,indexCount,indexSize,indexSize))
							return false;
						
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03804
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03805
						for (auto i=0u; i<(hasMotion ? 2u:1u); i++)
						if (Base::invalidInputBuffer(geometry.vertexData[i],firstVertexByteOffset,buildRangeInfo.firstVertex+geometry.maxVertex,geometry.vertexStride,vertexAlignment))
							return false;
					}
					else
					{
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03657
						if (!core::is_aligned_to(buildRangeInfo.primitiveByteOffset,vertexAlignment))
							return false;
						
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03804
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03805
						for (auto i=0u; i<(hasMotion ? 2u:1u); i++)
						if (Base::invalidInputBuffer(geometry.vertexData[i],buildRangeInfo.primitiveByteOffset+firstVertexByteOffset,indexCount,geometry.vertexStride,vertexAlignment))
							return false;
					}

					if (geometry.hasTransform())
					{
						if constexpr (std::is_same_v<BufferType,IGPUBuffer>)
						{
							// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03808
							// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03809
							// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03810
							// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03773
							if (Base::invalidInputBuffer(geometry.transform,buildRangeInfo.transformByteOffset,1u,sizeof(core::matrix3x4SIMD),sizeof(core::vectorSIMDf)))
								return false;
						}
						else
						{
							const hlsl::float32_t3x4& transform = geometry.transform;
							const auto upper3x3 = hlsl::float32_t3x3(transform);
							const float det = hlsl::determinant(upper3x3);
							// weird use of ! because we want to handle NaN as well
							if (!(std::abs(det)>std::numeric_limits<float>::min()))
								return false;
						}
					}

					totalPrims += buildRangeInfo.primitiveCount;
					return true;
				}

				inline core::smart_refctd_ptr<const IReferenceCounted>* fillTracking(core::smart_refctd_ptr<const IReferenceCounted>* oit) const
				{
					*(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(Base::scratch.buffer);
					if (Base::isUpdate)
						*(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(srcAS);
					*(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(dstAS);

					if (buildFlags.hasFlags(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
					{
						for (auto i=0u; i<geometryCount; i++)
							*(oit++) = aabbs[i].data.buffer;
					}
					else
					{
						for (auto i=0u; i<geometryCount; i++)
						{
							const auto& geometry = triangles[i];
							*(oit++) = geometry.vertexData[0].buffer;
							if (geometry.vertexData[1].buffer)
								*(oit++) = geometry.vertexData[1].buffer;
							if (geometry.indexData.buffer)
								*(oit++) = geometry.indexData.buffer;
							if constexpr (std::is_same_v<BufferType,IGPUBuffer>)
							if (geometry.hasTransform())
								*(oit++) = geometry.transform.buffer;
						}
					}

					return oit;
				}

				core::bitflag<BUILD_FLAGS> buildFlags = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
				uint32_t geometryCount = 0u;
				const IGPUBottomLevelAccelerationStructure* srcAS = nullptr;
				IGPUBottomLevelAccelerationStructure* dstAS = nullptr;
				// please interpret based on `buildFlags.hasFlags(GEOMETRY_TYPE_IS_AABB_BIT)`
				union
				{
					const Triangles<const BufferType>* triangles = nullptr;
					const AABBs<const BufferType>* aabbs;
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
		using Base::Base;

	private:
		bool validVertexFormat(const asset::E_FORMAT format) const;
};

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

		// read the comments in the .hlsl file
		using BuildRangeInfo = hlsl::acceleration_structures::top_level::BuildRangeInfo;
		using DirectBuildRangeRangeInfos = const BuildRangeInfo*;
		using MaxInputCounts = const uint32_t;

		template<typename BufferType>
		struct BuildInfo final : IGPUAccelerationStructure::BuildInfo<BufferType>
		{
			private:
				using Base = IGPUAccelerationStructure::BuildInfo<BufferType>;

			public:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03791
				inline uint32_t inputCount() const {return 1u;}

				// Returns 0 on failure, otherwise returns the number of `core::smart_refctd_ptr` to reserve for lifetime tracking
				// List of things too expensive or impossible (without GPU Assist) to validate:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03706
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03709
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03671
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03672
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-06707
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-dstAccelerationStructure-03706
				template<typename T> requires nbl::is_any_of_v<T,std::conditional_t<std::is_same_v<BufferType,IGPUBuffer>,uint32_t,BuildRangeInfo>,BuildRangeInfo>
				inline uint32_t valid(const T& buildRangeInfo) const
				{
					if constexpr (std::is_same_v<T,uint32_t>)
						return valid<BuildRangeInfo>({.instanceCount=buildRangeInfo,.instanceByteOffset=0});
					else
					{
						if (IGPUAccelerationStructure::BuildInfo<BufferType>::invalid(srcAS,dstAS))
							return false;
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03801
						if (buildRangeInfo.instanceCount>dstAS->getMaxInstanceCount())
							return false;
				
						const bool arrayOfPointers = buildFlags.hasFlags(BUILD_FLAGS::INSTANCE_DATA_IS_POINTERS_TYPE_ENCODED_LSB);
						constexpr bool HostBuild = std::is_same_v<BufferType,asset::ICPUBuffer>;
						// I'm not gonna do the `std::conditional_t<HostBuild,,>` to get the correct Instance struct type as they're the same size essentially
						const size_t instanceSize = arrayOfPointers ? sizeof(void*):(
							dstAS->getCreationParams().flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT) ? sizeof(DevicePolymorphicInstance):sizeof(DeviceStaticInstance)
						);
				
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03660
						if (!core::is_aligned_to(buildRangeInfo.instanceByteOffset,16ull))
							return false;
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03715
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03716
						const size_t instanceAlignment = arrayOfPointers ? 16u:sizeof(void*);
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03813
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03814
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03778
						if (Base::invalidInputBuffer(instanceData,buildRangeInfo.instanceByteOffset,buildRangeInfo.instanceCount,instanceSize,instanceAlignment))
							return false;

						#ifdef _NBL_DEBUG
						/* TODO: with `EXT_private_data
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03724
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03779
						// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-04930
						if constexpr (HostBuild)
						{
							for (auto 
							if (device->invalidAccelerationStructureForHostOperations(getAccelerationStructureFromReference(geometry.instanceData.blas)))
								return false;
						}
						*/
						#endif

						// destination, scratch and instanceData are required, source is optional
						return Base::isUpdate ? 4u:3u;
					}
				}

				inline core::smart_refctd_ptr<const IReferenceCounted>* fillTracking(core::smart_refctd_ptr<const IReferenceCounted>* oit) const
				{
					*(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(Base::scratch.buffer);
					if (Base::isUpdate)
						*(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(srcAS);
					*(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(dstAS);

					*(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(instanceData.buffer);

					return oit;
				}


				core::bitflag<BUILD_FLAGS> buildFlags = BUILD_FLAGS::PREFER_FAST_BUILD_BIT;
				const IGPUTopLevelAccelerationStructure* srcAS = nullptr;
				IGPUTopLevelAccelerationStructure* dstAS = nullptr;
				// depending on the presence certain bits in `buildFlags` this buffer will be filled with:
				// - addresses to `StaticInstance`, `MatrixMotionInstance`, `SRTMotionInstance` packed in upper 60 bits 
				//   and struct type in lower 4 bits if and only if `buildFlags.hasFlags(INSTANCE_DATA_IS_POINTERS_TYPE_ENCODED_LSB)`, otherwise:
				//	+ an array of `PolymorphicInstance` if our `SCreationParams::flags.hasFlags(MOTION_BIT)`, otherwise
				//	+ an array of `StaticInstance`
				asset::SBufferBinding<const BufferType> instanceData = {};
		};
		using DeviceBuildInfo = BuildInfo<IGPUBuffer>;
		using HostBuildInfo = BuildInfo<asset::ICPUBuffer>;

		//! BEWARE, OUR RESOURCE LIFETIME TRACKING DOES NOT WORK ACROSS TLAS->BLAS boundaries with these types of BLAS references!
		// TODO: Investigate `EXT_private_data` to be able to go ` -> IGPUBottomLevelAccelerationStructure` on Host Builds
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
				inline PolymorphicInstance() = default;
				inline PolymorphicInstance(const PolymorphicInstance<blas_ref_t>&) = default;
				inline PolymorphicInstance(PolymorphicInstance<blas_ref_t>&&) = default;

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
				static_assert(alignof(SRTMotionInstance<blas_ref_t>)==8ull);
		};
		using DevicePolymorphicInstance = PolymorphicInstance<IGPUBottomLevelAccelerationStructure::device_op_ref_t>;
		using HostPolymorphicInstance = PolymorphicInstance<IGPUBottomLevelAccelerationStructure::host_op_ref_t>;
		static_assert(sizeof(DevicePolymorphicInstance)==sizeof(HostPolymorphicInstance));

	protected:
		inline IGPUTopLevelAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params)
			: asset::ITopLevelAccelerationStructure<IGPUAccelerationStructure>(std::move(dev),std::move(params)), m_maxInstanceCount(params.maxInstanceCount) {}

		const uint32_t m_maxInstanceCount;
};

}

#endif