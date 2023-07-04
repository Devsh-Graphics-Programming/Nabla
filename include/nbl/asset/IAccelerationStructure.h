// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "aabbox3d.h"

#include <compare>

#include "nbl/asset/ECommonEnums.h"
#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/IBuffer.h"
#include "nbl/asset/format/EFormat.h"


namespace nbl::asset
{

class IAccelerationStructure : public IDescriptor
{
	public:
		// we don't expose the GENERIC type because Vulkan only intends it for API-translation layers like VKD3D or MoltenVK
		virtual bool isBLAS() const = 0;

		//!
		inline E_CATEGORY getTypeCategory() const override { return EC_ACCELERATION_STRUCTURE; }

	protected:
		IAccelerationStructure() = default;
};

template<class AccelerationStructure>
class IBottomLevelAccelerationStructure : public AccelerationStructure
{
		static_assert(std::is_base_of_v<IAccelerationStructure,AccelerationStructure>);
	public:
		inline bool isBLAS() const override {return true;}

		// build flags, we don't expose flags that don't make sense for certain levels
		enum class BUILD_FLAGS : uint16_t
		{
			ALLOW_UPDATE_BIT = 0x1u<<0u,
			ALLOW_COMPACTION_BIT = 0x1u<<1u,
			PREFER_FAST_TRACE_BIT = 0x1u<<2u,
			PREFER_FAST_BUILD_BIT = 0x1u<<3u,
			LOW_MEMORY_BIT = 0x1u<<4u,
			// Synthetic flag we use to indicate that the build data are AABBs instead of triangles, we've taken away the per-geometry choice thanks to:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03792
			GEOMETRY_TYPE_IS_AABB_BIT = 0x1u<<5u,
			// Provided by VK_NV_ray_tracing_motion_blur, but is ignored for BLASes
			//MOTION_BIT = 0x1u<<5u
			// Provided by VK_EXT_opacity_micromap
			ALLOW_OPACITY_MICROMAP_UPDATE_BIT = 0x1u<<6u,
			ALLOW_DISABLE_OPACITY_MICROMAPS_BIT = 0x1u<<7u,
			ALLOW_OPACITY_MICROMAP_DATA_UPDATE_BIT = 0x1u<<8u,
			// Provided by VK_NV_displacement_micromap
			ALLOW_DISPLACEMENT_MICROMAP_UPDATE_BIT = 0x1u<<9u,
			// Provided by VK_KHR_ray_tracing_position_fetch
			ALLOW_DATA_ACCESS_KHR = 0x1u<<11u,
		};
		
		// Apparently Vulkan allows setting these on TLAS Geometry (which are instances) but applying them to a TLAS doesn't make any SENSE AT ALL!
		enum class GEOMETRY_FLAGS : uint8_t
		{
			NONE								= 0u,
			// means you don't ever want to invoke any-hit shaders on thie geometry, ever.
			OPAQUE_BIT							= 0x1u<<0u,
			// useful for dealing with transmissivity effects
			NO_DUPLICATE_ANY_HIT_INVOCATION_BIT	= 0x1u<<1u,
		};

		// Note that in Vulkan strides are 64-bit value but restricted to be 32-bit in range
		template<typename BufferType>
		struct Triangles
		{
			// vertexData[1] are the vertex positions at time 1.0, and only used for AccelerationStructures created with `MOTION_BIT`
			asset::SBufferBinding<const BufferType>	vertexData[2] = {{},{}};
			asset::SBufferBinding<const BufferType>	indexData = {};
			// optional, only useful for baking model transforms of multiple meshes into one BLAS
			asset::SBufferBinding<const BufferType>	transformData = {};
			uint32_t								maxVertex = 0u;
			uint32_t								vertexStride = sizeof(float);
			E_FORMAT								vertexFormat = EF_R32G32B32_SFLOAT;
			E_INDEX_TYPE							indexType = EIT_32BIT;
			core::bitflag<GEOMETRY_FLAGS>			geometryFlags = GEOMETRY_FLAGS::NONE;
			// TODO: opacity and displacement micromap buffers and shizz
		};

		//
		template<typename BufferType>
		struct AABBs
		{
			// for `MOTION_BIT` you don't get a second buffer for AABBs at different times because linear interpolation doesn't work
			asset::SBufferBinding<const BufferType>	data = {};
			uint32_t								stride = sizeof(AABB_t);
			core::bitflag<GEOMETRY_FLAGS>			geometryFlags = GEOMETRY_FLAGS::NONE;
		};
		// For filling the AABB data buffers
		using AABB_t = core::aabbox3d<float>;

	protected:
		using AccelerationStructure::AccelerationStructure;
		virtual ~IBottomLevelAccelerationStructure() = default;
};

template<class AccelerationStructure>
class ITopLevelAccelerationStructure : public AccelerationStructure
{
		static_assert(std::is_base_of_v<IAccelerationStructure,AccelerationStructure>);
	public:
		inline bool isBLAS() const override {return false;}

		// build flags, we don't expose flags that don't make sense for certain levels
		enum class BUILD_FLAGS : uint8_t
		{
			ALLOW_UPDATE_BIT = 0x1u<<0u,
			ALLOW_COMPACTION_BIT = 0x1u<<1u,
			PREFER_FAST_TRACE_BIT = 0x1u<<2u,
			PREFER_FAST_BUILD_BIT = 0x1u<<3u,
			LOW_MEMORY_BIT = 0x1u<<4u,
			// Synthetic flag we use to indicate `VkAccelerationStructureGeometryInstancesDataKHR::arrayOfPointers`
			INSTANCE_TYPE_ENCODED_IN_POINTER_LSB = 0x1u<<5u,
			// Provided by VK_NV_ray_tracing_motion_blur, but we always override and deduce from creation flag because of
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-dstAccelerationStructure-04927
			//MOTION_BIT = 0x1u<<5u,
		};

		enum class INSTANCE_FLAGS : uint8_t
		{
			NONE = 0u,
			TRIANGLE_FACING_CULL_DISABLE_BIT = 0x1u<<0u,
			// changes CCW to CW for backface culling
			TRIANGLE_FLIP_FACING_BIT = 0x1u<<1u,
			FORCE_OPAQUE_BIT = 0x1u<<2u,
			FORCE_NO_OPAQUE_BIT = 0x1u<<3u,
			// Provided by VK_EXT_opacity_micromap
			FORCE_OPACITY_MICROMAP_2_STATE_BIT = 0x1u<<4u,
			FORCE_DISABLE_OPACITY_MICROMAPS_BIT = 0x1u<<5u,
		};
		// Note: `core::matrix3x4SIMD` is equvalent to VkTransformMatrixKHR, 4x3 row_major matrix
		template<typename blas_ref_t>
		struct Instance final
		{
			static_assert(sizeof(blas_ref_t)==8 && alignof(blas_ref_t)==8);
			static_assert(std::is_same_v<core::IReferenceCounted<asset::ICPUBottomLevelAccelerationStructure>,blas_ref_t> || std::is_standard_layout_v<blas_ref_t>);

			uint32_t	instanceCustomIndex : 24 = 0u;
			uint32_t	mask : 8 = 0xFFu;
			uint32_t	instanceShaderBindingTableRecordOffset : 24 = 0u;
			uint32_t	flags : 8 = INSTANCE_FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT;
			blas_ref_t	blas = {};
		};
		template<typename blas_ref_t>
		struct StaticInstance final
		{
			core::matrix3x4SIMD	transform;
			Instance<blas_ref_t> base = {};
		};
		template<typename blas_ref_t>
		struct MatrixMotionInstance final
		{
			core::matrix3x4SIMD transform[2];
			Instance<blas_ref_t> base = {};
		};
		struct SRT
		{
			// TODO: some operators to convert back and forth from `core::matrix3x4SIMD

			float    sx;
			float    a;
			float    b;
			float    pvx;
			float    sy;
			float    c;
			float    pvy;
			float    sz;
			float    pvz;
			float    qx;
			float    qy;
			float    qz;
			float    qw;
			float    tx;
			float    ty;
			float    tz;
		};
		template<typename blas_ref_t>
		struct SRTMotionInstance final
		{
			alignas(8) SRT transform[2];
			Instance<blas_ref_t> base = {};

			static_assert(sizeof(base)==16ull);
			static_assert(alignof(base)==8ull);
		};

		// enum for distinguishing unions of Instance Types when there is no `INSTANCE_TYPE_ENCODED_IN_POINTER_LSB` in build flags
		enum class INSTANCE_TYPE : uint32_t
		{
			// StaticInstance
			STATIC,
			// MatrixMotionInstance
			MATRIX_MOTION,
			// SRTMotionInstance
			SRT_MOTION
		};

	protected:
		using AccelerationStructure::AccelerationStructure;
		virtual ~ITopLevelAccelerationStructure() = default;
};

} // end namespace nbl::asset

#endif


