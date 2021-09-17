// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/ECommonEnums.h"
#include "nbl/asset/IBuffer.h"
#include "nbl/asset/format/EFormat.h"
#include "aabbox3d.h"
#include "matrix4x3.h"
#define uint uint32_t
#include "nbl/builtin/glsl/utils/acceleration_structures.glsl"
#undef uint

namespace nbl::asset
{
class IAccelerationStructure : public IDescriptor
{
	public:
		enum E_TYPE : uint32_t
		{
			ET_TOP_LEVEL = 0,
			ET_BOTTOM_LEVEL = 1,
			ET_GENERIC = 2,
		};
		enum E_CREATE_FLAGS : uint32_t
		{
			ECF_NONE								= 0u,
			ECF_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT	= 0x1u << 0u,
			ECF_MOTION_BIT_NV						= 0x1u << 1u, // Provided by VK_NV_ray_tracing_motion_blur
		};

		using BuildRangeInfo = nbl_glsl_BuildRangeInfo;
		enum E_BUILD_FLAGS
		{
			EBF_ALLOW_UPDATE_BIT = 0x1u << 0u,
			EBF_ALLOW_COMPACTION_BIT = 0x1u << 1u,
			EBF_PREFER_FAST_TRACE_BIT = 0x1u << 2u,
			EBF_PREFER_FAST_BUILD_BIT = 0x1u << 3u,
			EBF_LOW_MEMORY_BIT = 0x1u << 4u,
			EBF_MOTION_BIT_NV = 0x1u << 5u, // Provided by VK_NV_ray_tracing_motion_blur
		};
		enum E_BUILD_MODE
		{
			EBM_BUILD = 0,
			EBM_UPDATE = 1,
		};
		
		enum E_GEOM_TYPE
		{
			EGT_TRIANGLES = 0,
			EGT_AABBS = 1,
			EGT_INSTANCES = 2,
		};
		enum E_GEOM_FLAGS {
			EGF_NONE								= 0u,
			EGF_OPAQUE_BIT							= 0x1u << 0u,
			EGF_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT	= 0x1u << 1u,
		};
		enum E_INSTANCE_FLAGS
		{
			EIF_NONE								= 0u,
			EIF_TRIANGLE_FACING_CULL_DISABLE_BIT	= 0x1u << 0u,
			EIF_TRIANGLE_FLIP_FACING_BIT			= 0x1u << 1u,
			EIF_FORCE_OPAQUE_BIT					= 0x1u << 2u,
			EIF_FORCE_NO_OPAQUE_BIT					= 0x1u << 3u,
			EIF_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR = EIF_TRIANGLE_FLIP_FACING_BIT,
		};

		template<typename AddressType>
		struct GeometryData
		{
			GeometryData() 
			{
				std::memset(this, 0, sizeof(GeometryData));
			}
			~GeometryData() {}

			GeometryData(GeometryData& copy)
			{
				std::memmove(this, &copy, sizeof(GeometryData));
			}

			GeometryData(const GeometryData& copy)
			{
				std::memmove(this, &copy, sizeof(GeometryData));
			}

			GeometryData& operator=(GeometryData& copy)
			{
				std::memmove(this, &copy, sizeof(GeometryData));
				return *this;
			}

			GeometryData& operator=(const GeometryData& copy)
			{
				std::memmove(this, &copy, sizeof(GeometryData));
				return *this;
			}

			union
			{
				struct Triangles
				{
					E_FORMAT		vertexFormat;
					AddressType		vertexData;
					uint64_t		vertexStride;
					uint32_t		maxVertex;
					E_INDEX_TYPE	indexType;
					AddressType		indexData;
					AddressType		transformData;
				} triangles;

				struct AABBs
				{
					AddressType		data;
					size_t			stride;
				} aabbs;

				struct Instances
				{
					AddressType		data;
				} instances;
			};
		};
		
		template<typename AddressType>
		struct Geometry
		{
			Geometry()
				: type(static_cast<E_GEOM_TYPE>(0u))
				, flags(static_cast<E_GEOM_FLAGS>(0u))
			{};
			E_GEOM_TYPE					type;
			E_GEOM_FLAGS				flags; // change to core::bitflags later
			GeometryData<AddressType>	data;
		};

		// For Filling the Instances/AABBs Buffer
		using AABB_Position = core::aabbox3d<float>;

		struct Instance
		{
			Instance()
				: instanceCustomIndex(0u)
				, mask(0xFF)
				, instanceShaderBindingTableRecordOffset(0u)
				, flags(EIF_NONE)
				, accelerationStructureReference(0ull)
				, mat(core::matrix3x4SIMD())
			{}
			core::matrix3x4SIMD				mat; // equvalent to VkTransformMatrixKHR, 4x3 row_major matrix
			uint32_t						instanceCustomIndex:24;
			uint32_t						mask:8;
			uint32_t						instanceShaderBindingTableRecordOffset:24;
			E_INSTANCE_FLAGS				flags:8;
			uint64_t						accelerationStructureReference; // retrieve via `getReference` functions in IGPUAccelerationStructrue
		};
		
		enum E_COPY_MODE 
		{
			ECM_CLONE = 0,
			ECM_COMPACT = 1,
			ECM_SERIALIZE = 2,
			ECM_DESERIALIZE = 3,
		};

		//!
		E_CATEGORY getTypeCategory() const override { return EC_ACCELERATION_STRUCTURE; }

	protected:
		IAccelerationStructure() 
		{
		}

		virtual ~IAccelerationStructure()
		{}

	private:
};
} // end namespace nbl::asset

#endif


