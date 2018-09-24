// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef _C_IMAGE_WRITER_JPG_H_INCLUDED__
#define _C_IMAGE_WRITER_JPG_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_JPG_WRITER_

#include "IAssetWriter.h"

namespace irr
{
namespace video
{

class CImageWriterJPG : public asset::IAssetWriter
{
public:
	//! constructor
	CImageWriterJPG();

    virtual const char** getAssociatedFileExtensions() const
    {
        static const char* ext[]{ "jpg", "jpeg", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

    virtual uint32_t getSupportedFlags() override { return asset::EWF_COMPRESSED; }

    virtual uint32_t getForcedFlags() { return asset::EWF_BINARY; }

    virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};

}
}

#endif // _C_IMAGE_WRITER_JPG_H_INCLUDED__
#endif

