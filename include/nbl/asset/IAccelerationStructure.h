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

		enum class CREATE_FLAGS : uint8_t
		{
			NONE								= 0u,
			//DEVICE_ADDRESS_CAPTURE_REPLAY_BIT	= 0x1u<<0u, for tools only
			// Provided by VK_NV_ray_tracing_motion_blur
			MOTION_BIT							= 0x1u<<1u,
		};
		inline core::bitflag<CREATE_FLAGS> getCreationFlags() const {return m_creationFlags;}

		// build flags
		enum class BUILD_FLAGS : uint16_t
		{
			ALLOW_UPDATE_BIT = 0x1u<<0u,
			ALLOW_COMPACTION_BIT = 0x1u<<1u,
			PREFER_FAST_TRACE_BIT = 0x1u<<2u,
			PREFER_FAST_BUILD_BIT = 0x1u<<3u,
			LOW_MEMORY_BIT = 0x1u<<4u,
			// Provided by VK_NV_ray_tracing_motion_blur
			MOTION_BIT = 0x1u<<5u,
			// Provided by VK_EXT_opacity_micromap
			ALLOW_OPACITY_MICROMAP_UPDATE_BIT = 0x1u<<6u,
			ALLOW_DISABLE_OPACITY_MICROMAPS_BIT = 0x1u<<7u,
			ALLOW_OPACITY_MICROMAP_DATA_UPDATE_BIT = 0x1u<<8u,
			// Provided by VK_NV_displacement_micromap
			ALLOW_DISPLACEMENT_MICROMAP_UPDATE_BIT = 0x1u<<9u,
			// Provided by VK_KHR_ray_tracing_position_fetch
			ALLOW_DATA_ACCESS_KHR = 0x1u<<11u
		};
		
		// we provide some level of type safety here
		enum class GEOMETRY_FLAGS : uint8_t
		{
			NONE								= 0u,
			// means you don't ever want to invoke any-hit shaders on thie geometry, ever.
			OPAQUE_BIT							= 0x1u<<0u,
			// useful for dealing with transmissivity effects
			NO_DUPLICATE_ANY_HIT_INVOCATION_BIT	= 0x1u<<1u,
		};

		//!
		inline E_CATEGORY getTypeCategory() const override { return EC_ACCELERATION_STRUCTURE; }

	protected:
		inline IAccelerationStructure(const core::bitflag<CREATE_FLAGS> flags) : m_creationFlags(flags) {}

	private:
		const core::bitflag<CREATE_FLAGS> m_creationFlags;
};

template<class AccelerationStructure>
class IBottomLevelAccelerationStructure : public AccelerationStructure
{
		static_assert(std::is_base_of_v<IAccelerationStructure,AccelerationStructure>);
	public:
		inline bool isBLAS() const override {return true;}

		template<typename BufferType>
		struct Geometry
		{
			// Note that in Vulkan strides are 64-bit value but restricted to be 32-bit in range
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
			};
			struct AABBs
			{
				// data[1] are the AABB extents at time 1.0, and only used for AccelerationStructures created with `MOTION_BIT`
				asset::SBufferBinding<const BufferType>	data[2] = {};
				uint32_t								stride = sizeof(AABB_t);
			};

			inline auto operator<=>(const Geometry& other) const
			{
				return std::memcmp(this, &other, sizeof(Geometry));
			}

			union
			{
				Triangles triangles = {};
				AABBs aabbs;
			};
			GEOMETRY_FLAGS flags : 15 = GEOMETRY_FLAGS::NONE;
			uint16_t isAABB : 1 = false;
		};

		// For filling the AABB Buffers
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

		template<typename BufferType>
		struct BuildGeometryInfo
		{
			enum class TYPE : uint8_t
			{
				// StaticInstance
				STATIC_INSTANCES,
				// MatrixMotionInstance
				MATRIX_MOTION_INSTANCES,
				// SRTMotionInstance
				SRT_MOTION_INSTANCES
			};

			inline auto operator<=>(const Geometry& other) const
			{
				return std::memcmp(this, &other, sizeof(Geometry));
			}

			asset::SBufferBinding<const BufferType>	instanceData = {};
			core::bitflag<GEOMETRY_FLAGS>			flags = GEOMETRY_FLAGS::NONE;
			core::bitflag<TYPE>						type = TYPE::STATIC_INSTANCES;
		};

		// For filling the Instance Buffers
		template<typename blas_ref_t>
		struct Instance
		{
			enum class FLAGS : uint32_t
			{
				NONE = 0u,
				TRIANGLE_FACING_CULL_DISABLE_BIT = 0x1u<<0u,
				// changes CCW to CW for backface culling
				TRIANGLE_FLIP_FACING_BIT = 0x1u<<1u,
				FORCE_OPAQUE_BIT = 0x1u<<2u,
				FORCE_NO_OPAQUE_BIT = 0x1u<<3u
			};
			uint32_t	instanceCustomIndex : 24 = 0u;
			uint32_t	mask : 8 = 0xFFu;
			uint32_t	instanceShaderBindingTableRecordOffset : 24 = 0u;
			FLAGS		flags : 8 = FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT;
			blas_ref_t	blas = {};
		};
		// core::matrix3x4SIMD is equvalent to VkTransformMatrixKHR, 4x3 row_major matrix
		template<typename blas_ref_t>
		struct StaticInstance
		{
			core::matrix3x4SIMD	transform;
			Instance<blas_ref_t> instance;
		};
		template<typename blas_ref_t>
		struct MatrixMotionInstance
		{
			core::matrix3x4SIMD transform[2];
			Instance<blas_ref_t> instance;
		};
		template<typename blas_ref_t>
		struct SRTMotionInstance
		{
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
			SRT transform[2];
			Instance<blas_ref_t> instance;
		};

	protected:
		using AccelerationStructure::AccelerationStructure;
		virtual ~ITopLevelAccelerationStructure() = default;
};

} // end namespace nbl::asset

#endif


