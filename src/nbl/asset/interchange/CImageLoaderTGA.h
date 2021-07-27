// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_IMAGE_LOADER_TGA_H_INCLUDED__
#define __NBL_ASSET_C_IMAGE_LOADER_TGA_H_INCLUDED__

#include "BuildConfigOptions.h"
#include "nbl/asset/interchange/IImageLoader.h"
#include "nbl/system/ISystem.h"

namespace nbl
{
namespace asset
{

#if defined(_NBL_COMPILE_WITH_TGA_LOADER_) || defined(_NBL_COMPILE_WITH_TGA_WRITER_)

// byte-align structures
#include "nbl/nblpack.h"
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

enum STANDARD_TGA_BITS
{
	STB_8_BITS = 8,
	STB_16_BITS = 16,
	STB_24_BITS = 24,
	STB_32_BITS = 32,
	STB_COUNT
};

enum STANDARD_TGA_IMAGE_TYPE
{
	STIT_NONE = 0,
	STIT_UNCOMPRESSED_COLOR_MAPPED_IMAGE = 1,
	STIT_UNCOMPRESSED_RGB_IMAGE = 2,
	STIT_UNCOMPRESSED_GRAYSCALE_IMAGE = 3,
	STIT_RLE_TRUE_COLOR_IMAGE = 10,
	STIT_COUNT
};
// Default alignment
#include "nbl/nblunpack.h"

#endif // compiled with loader or reader

#ifdef _NBL_COMPILE_WITH_TGA_LOADER_

/*!
	Surface Loader for targa images
*/
class CImageLoaderTGA final : public IImageLoader
{
	core::smart_refctd_ptr<system::ISystem> m_system;
	public:
		CImageLoaderTGA(core::smart_refctd_ptr<system::ISystem>&& sys) : m_system(std::move(sys)) {}
		virtual bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr& logger) const override;

		virtual const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "tga", nullptr };
			return ext;
		}

		virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

		virtual asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

	private:

		//! loads a compressed tga. Was written and sent in by Jon Pry, thank you very much!
		void loadCompressedImage(system::IFile *file, const STGAHeader& header, const uint32_t wholeSizeWithPitchInBytes, core::smart_refctd_ptr<ICPUBuffer>& bufferData) const;
};

#endif // compiled with loader

} // end namespace video
} // end namespace nbl

#endif
