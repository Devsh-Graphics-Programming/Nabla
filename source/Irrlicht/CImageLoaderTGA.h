// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_LOADER_TGA_H_INCLUDED__
#define __C_IMAGE_LOADER_TGA_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "IImageLoader.h"


namespace irr
{
namespace video
{

#if defined(_IRR_COMPILE_WITH_TGA_LOADER_) || defined(_IRR_COMPILE_WITH_TGA_WRITER_)

// byte-align structures
#include "irrpack.h"

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

	struct STGAFooter
	{
		uint32_t ExtensionOffset;
		uint32_t DeveloperOffset;
		char  Signature[18];
	} PACK_STRUCT;

// Default alignment
#include "irrunpack.h"

#endif // compiled with loader or reader

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_

/*!
	Surface Loader for targa images
*/
class CImageLoaderTGA : public IImageLoader
{
public:

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".tga")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! returns true if the file maybe is able to be loaded by this class
	virtual bool isALoadableFileFormat(io::IReadFile* file) const;

	//! creates a surface from the file
	virtual IImage* loadImage(io::IReadFile* file) const;

private:

	//! loads a compressed tga. Was written and sent in by Jon Pry, thank you very much!
	uint8_t* loadCompressedImage(io::IReadFile *file, const STGAHeader& header) const;
};

#endif // compiled with loader

} // end namespace video
} // end namespace irr

#endif

