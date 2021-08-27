// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/ECommonEnums.h"
#include "nbl/asset/IBuffer.h"
#include "nbl/asset/format/EFormat.h"

#define uint uint32_t
#include "nbl/builtin/glsl/utils/acceleration_structures.glsl"
#undef uint

namespace nbl::asset
{
template<typename AddressType>
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

		struct GeometryData
		{
			union
			{
				struct Triangles {
					E_FORMAT		vertexFormat;
					AddressType		vertexData;
					VkDeviceSize	vertexStride;
					uint32_t		maxVertex;
					VkIndexType		indexType;
					AddressType		indexData;
					AddressType		transformData;
				} triangles;
				struct AABBs {
					AddressType		data;
					size_t			stride;
				} aabbs;
				struct Instances {
					AddressType		data;
				} instances;
			};
		};

		struct Geometry
		{
			enum E_TYPE
			{
				ET_TRIANGLES = 0,
				ET_AABBS = 1,
				ET_INSTANCES = 2,
			};
			enum E_FLAGS {
				EF_OPAQUE_BIT							= 0x1u << 0u,
				EF_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT	= 0x1u << 1u,
			};
			E_TYPE			type;
			E_FLAGS			flags;
			GeometryData	data;
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


