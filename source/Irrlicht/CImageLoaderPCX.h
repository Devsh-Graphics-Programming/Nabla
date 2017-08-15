// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_LOADER_PCX_H_INCLUDED__
#define __C_IMAGE_LOADER_PCX_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "IImageLoader.h"

namespace irr
{
namespace video
{

#if defined(_IRR_COMPILE_WITH_PCX_LOADER_) || defined(_IRR_COMPILE_WITH_PCX_WRITER_)

// byte-align structures
#include "irrpack.h"

	struct SPCXHeader
	{
		uint8_t	Manufacturer;
		uint8_t	Version;
		uint8_t	Encoding;
		uint8_t	BitsPerPixel;
		uint16_t	XMin;
		uint16_t	YMin;
		uint16_t	XMax;
		uint16_t	YMax;
		uint16_t	HorizDPI;
		uint16_t	VertDPI;
		uint8_t	Palette[48];
		uint8_t	Reserved;
		uint8_t	Planes;
		uint16_t	BytesPerLine;
		uint16_t	PaletteType;
		uint16_t	HScrSize;
		uint16_t	VScrSize;
		uint8_t	Filler[54];
	} PACK_STRUCT;


// Default alignment
#include "irrunpack.h"

#endif // compile with loader or writer

#ifdef _IRR_COMPILE_WITH_PCX_LOADER_

/*!
	Image Loader for Windows PCX bitmaps.
	This loader was written and sent in by Dean P. Macri. I modified
	only some small bits of it.
*/
class CImageLoaderPCX : public IImageLoader
{
public:

	//! constructor
	CImageLoaderPCX();

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".tga")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! returns true if the file maybe is able to be loaded by this class
	virtual bool isALoadableFileFormat(io::IReadFile* file) const;

	//! creates a surface from the file
	virtual IImage* loadImage(io::IReadFile* file) const;

};

#endif // compile with loader

} // end namespace video
} // end namespace irr

#endif

