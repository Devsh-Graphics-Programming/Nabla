// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_LOADER_PSD_H_INCLUDED__
#define __C_IMAGE_LOADER_PSD_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_PSD_LOADER_

#include "IImageLoader.h"

namespace irr
{
namespace video
{


// byte-align structures
#include "irrpack.h"

	struct PsdHeader
	{
		char signature [4];	// Always equal to 8BPS.
		uint16_t version;		// Always equal to 1
		char reserved [6];	// Must be zero
		uint16_t channels;		// Number of any channels inc. alphas
		uint32_t height;		// Rows Height of image in pixel
		uint32_t width;		// Colums Width of image in pixel
		uint16_t depth;		// Bits/channel
		uint16_t mode;		// Color mode of the file (Bitmap/Grayscale..)
	} PACK_STRUCT;


// Default alignment
#include "irrunpack.h"

/*!
	Surface Loader for psd images
*/
class CImageLoaderPSD : public IImageLoader
{
public:

	//! constructor
	CImageLoaderPSD();

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".tga")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! returns true if the file maybe is able to be loaded by this class
	virtual bool isALoadableFileFormat(io::IReadFile* file) const;

	//! creates a surface from the file
	virtual IImage* loadImage(io::IReadFile* file) const;

private:

	bool readRawImageData(io::IReadFile* file, const PsdHeader& header, uint32_t* imageData) const;
	bool readRLEImageData(io::IReadFile* file, const PsdHeader& header, uint32_t* imageData) const;
	int16_t getShiftFromChannel(char channelNr, const PsdHeader& header) const;
};


} // end namespace video
} // end namespace irr

#endif
#endif

