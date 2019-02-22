// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_LOADER_BMP_H_INCLUDED__
#define __C_IMAGE_LOADER_BMP_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/asset/IAssetLoader.h"


namespace irr
{
namespace asset
{

#if defined(_IRR_COMPILE_WITH_BMP_LOADER_) || defined(_IRR_COMPILE_WITH_BMP_WRITER_)


// byte-align structures
#include "irr/irrpack.h"

	struct SBMPHeader
	{
		uint16_t	Id;					//	BM - Windows 3.1x, 95, NT, 98, 2000, ME, XP
											//	BA - OS/2 Bitmap Array
											//	CI - OS/2 Color Icon
											//	CP - OS/2 Color Pointer
											//	IC - OS/2 Icon
											//	PT - OS/2 Pointer
		uint32_t	FileSize;
		uint32_t	Reserved;
		uint32_t	BitmapDataOffset;
		uint32_t	BitmapHeaderSize;	// should be 28h for windows bitmaps or
											// 0Ch for OS/2 1.x or F0h for OS/2 2.x
		uint32_t Width;
		uint32_t Height;
		uint16_t Planes;
		uint16_t BPP;					// 1: Monochrome bitmap
											// 4: 16 color bitmap
											// 8: 256 color bitmap
											// 16: 16bit (high color) bitmap
											// 24: 24bit (true color) bitmap
											// 32: 32bit (true color) bitmap

		uint32_t  Compression;			// 0: none (Also identified by BI_RGB)
											// 1: RLE 8-bit / pixel (Also identified by BI_RLE4)
											// 2: RLE 4-bit / pixel (Also identified by BI_RLE8)
											// 3: Bitfields  (Also identified by BI_BITFIELDS)

		uint32_t  BitmapDataSize;		// Size of the bitmap data in bytes. This number must be rounded to the next 4 byte boundary.
		uint32_t  PixelPerMeterX;
		uint32_t  PixelPerMeterY;
		uint32_t  Colors;
		uint32_t  ImportantColors;
	} PACK_STRUCT;

// Default alignment
#include "irr/irrunpack.h"

#endif // defined with loader or writer

#ifdef _IRR_COMPILE_WITH_BMP_LOADER_

/*!
	Surface Loader for Windows bitmaps
*/
class CImageLoaderBMP : public asset::IAssetLoader
{
public:

	//! constructor
	CImageLoaderBMP();

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override
    {
        const size_t prevPos = _file->getPos();
        _file->seek(0u);
        
        uint16_t headerId;
        _file->read(&headerId, 2);

        _file->seek(prevPos);

        return (headerId == 0x4d42u);
    }

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "bmp", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

    virtual asset::IAsset* loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:

	void decompress8BitRLE(uint8_t*& BmpData, int32_t size, int32_t width, int32_t height, int32_t pitch) const;

	void decompress4BitRLE(uint8_t*& BmpData, int32_t size, int32_t width, int32_t height, int32_t pitch) const;
};


#endif // compiled with loader

} // end namespace video
} // end namespace irr

#endif

