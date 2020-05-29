// Copyright (C) 2009-2012 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_WRITER_OPENEXR__
#define __C_IMAGE_WRITER_OPENEXR__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENEXR_WRITER_

#include "irr/asset/IImageWriter.h"

namespace irr
{
namespace asset
{

//! OpenEXR writer capable of saving .exr files
class CImageWriterOpenEXR final : public IImageWriter
{
	protected:
		~CImageWriterOpenEXR(){}

	public:
		CImageWriterOpenEXR(){}

		const char** getAssociatedFileExtensions() const override
		{
			static const char* extensions[]{ "exr", nullptr };
			return extensions;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }

		uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

		uint32_t getForcedFlags() { return asset::EWF_BINARY; }

		bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

	private:

		bool writeImageBinary(io::IWriteFile* file, const asset::ICPUImage* image);
};

}
}

#endif // _IRR_COMPILE_WITH_OPENEXR_WRITER_
#endif // __C_IMAGE_WRITER_OPENEXR__
