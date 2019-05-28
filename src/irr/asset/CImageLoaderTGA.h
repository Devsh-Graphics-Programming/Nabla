// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_LOADER_TGA_H_INCLUDED__
#define __C_IMAGE_LOADER_TGA_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/asset/IAssetLoader.h"


namespace irr
{
namespace asset
{

#if defined(_IRR_COMPILE_WITH_TGA_LOADER_) || defined(_IRR_COMPILE_WITH_TGA_WRITER_)

// byte-align structures
#include "irr/irrpack.h"

	// these structs are also used in the TGA writer
	struct STGAHeader{
		uint8_t IdLength;
		uint8_t ColorMapType;
		uint8_t ImageType;
		uint8_t FirstEntryIndex[2];
		uint16_t ColorMapLength;
		uint8_t ColorMapEntrySize;
		uint8_t XOrigin[2];
		uint8_t YOrigin[2];
		uint16_t ImageWidth;
		uint16_t ImageHeight;
		uint8_t PixelDepth;
		uint8_t ImageDescriptor;
	} PACK_STRUCT;
	
	// Note to maintainers: most of this (except gamma) goes unused, and the struct itself is ignored 
	// by most TGA readers/writers. But we have to use a full struct just to store gamma information, 
	// of course we can get around to just store gamma, but then it'd no longer be conformant to TGA standard.
	struct STGAExtensionArea {
		uint16_t ExtensionSize;
		char AuthorName[41];
		char AuthorComment[324];
		char DateTimeStamp[12];
		char JobID[41];
		char JobTime[6];
		char SoftwareID[41];
		char SoftwareVersion[3];
		uint32_t KeyColor;
		float PixelAspectRatio;
		float Gamma;
		uint32_t ColorCorrectionOffset;
		uint32_t PostageStampOffset;
		uint32_t ScanlineOffset;
		uint8_t AttributeType;
	} PACK_STRUCT;

	struct STGAFooter
	{
		uint32_t ExtensionOffset;
		uint32_t DeveloperOffset;
		char  Signature[18];
	} PACK_STRUCT;
	
// Default alignment
#include "irr/irrunpack.h"

#endif // compiled with loader or reader

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_

/*!
	Surface Loader for targa images
*/
class CImageLoaderTGA : public asset::IAssetLoader
{
public:
    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "tga", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

    virtual asset::IAsset* loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:

	//! loads a compressed tga. Was written and sent in by Jon Pry, thank you very much!
	uint8_t* loadCompressedImage(io::IReadFile *file, const STGAHeader& header) const;
};

#endif // compiled with loader

} // end namespace video
} // end namespace irr

#endif

