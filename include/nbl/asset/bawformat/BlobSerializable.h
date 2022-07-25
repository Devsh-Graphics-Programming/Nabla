// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_BLOB_SERIALIZABLE_H_INCLUDED__
#define __NBL_ASSET_BLOB_SERIALIZABLE_H_INCLUDED__

namespace nbl
{
namespace asset
{

class NBL_API NBL_FORCE_EBO BlobSerializable
{
	public:
		virtual ~BlobSerializable() {}

		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const = 0;
};

}
} // nbl::asset

#endif
