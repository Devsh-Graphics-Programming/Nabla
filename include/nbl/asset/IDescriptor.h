// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_DESCRIPTOR_H_INCLUDED__
#define __NBL_ASSET_I_DESCRIPTOR_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl
{
namespace asset
{

class NBL_API IDescriptor : public virtual core::IReferenceCounted
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

		virtual E_CATEGORY getTypeCategory() const = 0;

	protected:
		virtual ~IDescriptor() = default;
};

}
}

#endif