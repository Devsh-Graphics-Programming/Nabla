// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672
#ifndef __IRR_BLOB_SERIALIZABLE_H_INCLUDED__
#define __IRR_BLOB_SERIALIZABLE_H_INCLUDED__

namespace irr
{
namespace asset
{

class IRR_FORCE_EBO BlobSerializable
{
	public:
		virtual ~BlobSerializable() {}

		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const = 0;
};

}
} // irr::asset

#endif
