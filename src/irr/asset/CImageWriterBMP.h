// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef _C_IMAGE_WRITER_BMP_H_INCLUDED__
#define _C_IMAGE_WRITER_BMP_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_BMP_WRITER_

#include "irr/asset/IAssetWriter.h"

namespace irr
{
namespace asset
{

class CImageWriterBMP : public asset::IAssetWriter
{
public:
	//! constructor
	CImageWriterBMP();

    virtual const char** getAssociatedFileExtensions() const
    {
        static const char* ext[] {"bmp", nullptr};
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_SUB_IMAGE; }

    virtual uint32_t getSupportedFlags() override { return 0u; }

    virtual uint32_t getForcedFlags() { return asset::EWF_BINARY; }

	//! write image to file
	virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};

} // namespace video
} // namespace irr

#endif
#endif // _C_IMAGE_WRITER_BMP_H_INCLUDED__

