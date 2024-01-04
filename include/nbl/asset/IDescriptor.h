// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_DESCRIPTOR_H_INCLUDED_
#define _NBL_ASSET_I_DESCRIPTOR_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

namespace nbl::asset
{

class IDescriptor : public virtual core::IReferenceCounted
{
	public:
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
		enum E_CATEGORY : uint8_t
		{
			EC_BUFFER = 0,
			EC_IMAGE,
			EC_BUFFER_VIEW,
			EC_ACCELERATION_STRUCTURE,
			EC_COUNT
		};
		static inline E_CATEGORY GetTypeCategory(const E_TYPE type)
		{
			switch (type)
			{
				case E_TYPE::ET_COMBINED_IMAGE_SAMPLER:
				case E_TYPE::ET_STORAGE_IMAGE:
				case E_TYPE::ET_INPUT_ATTACHMENT:
					return EC_IMAGE;
					break;
				case E_TYPE::ET_UNIFORM_TEXEL_BUFFER:
				case E_TYPE::ET_STORAGE_TEXEL_BUFFER:
					return EC_BUFFER_VIEW;
					break;
				case E_TYPE::ET_UNIFORM_BUFFER:
				case E_TYPE::ET_STORAGE_BUFFER:
				case E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC:
				case E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
					return EC_BUFFER;
					break;
				case E_TYPE::ET_ACCELERATION_STRUCTURE:
					return EC_ACCELERATION_STRUCTURE;
					break;
				default:
					break;
			}
			return EC_COUNT;
		}

		virtual E_CATEGORY getTypeCategory() const = 0;

	protected:
		virtual ~IDescriptor() = default;
};

}

#endif