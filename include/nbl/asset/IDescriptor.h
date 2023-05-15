// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_DESCRIPTOR_H_INCLUDED__
#define __NBL_ASSET_I_DESCRIPTOR_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl::asset
{

class IDescriptor : public virtual core::IReferenceCounted
{
	public:
		enum E_CATEGORY
		{
			EC_BUFFER = 0,
			EC_IMAGE,
			EC_BUFFER_VIEW,
			EC_ACCELERATION_STRUCTURE,
			EC_COUNT
		};

		enum class E_TYPE : uint8_t
		{
			ET_COMBINED_IMAGE_SAMPLER = 0,
			ET_STORAGE_IMAGE,
			ET_UNIFORM_TEXEL_BUFFER,
			ET_STORAGE_TEXEL_BUFFER,
			ET_UNIFORM_BUFFER,
			ET_STORAGE_BUFFER,
			ET_UNIFORM_BUFFER_DYNAMIC,
			ET_STORAGE_BUFFER_DYNAMIC,
			ET_INPUT_ATTACHMENT,
			// Provided by VK_KHR_acceleration_structure
			ET_ACCELERATION_STRUCTURE,

			// Support for the following is not available:
			// Provided by VK_EXT_inline_uniform_block
			// ET_INLINE_UNIFORM_BLOCK_EXT,
			// Provided by VK_NV_ray_tracing
			// ET_ACCELERATION_STRUCTURE_NV = 1000165000,
			// Provided by VK_VALVE_mutable_descriptor_type
			// ET_MUTABLE_VALVE = 1000351000,

			ET_COUNT
		};

		virtual E_CATEGORY getTypeCategory() const = 0;

	protected:
		virtual ~IDescriptor() = default;
};

}

#endif