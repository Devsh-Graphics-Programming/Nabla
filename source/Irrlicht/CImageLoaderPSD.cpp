// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderPSD.h"

#ifdef _IRR_COMPILE_WITH_PSD_LOADER_

#include "IReadFile.h"
#include "os.h"
#include "CImage.h"
#include "irrString.h"


namespace irr
{
namespace video
{


//! constructor
CImageLoaderPSD::CImageLoaderPSD()
{
	#ifdef _DEBUG
	setDebugName("CImageLoaderPSD");
	#endif
}


//! returns true if the file maybe is able to be loaded by this class
//! based on the file extension (e.g. ".tga")
bool CImageLoaderPSD::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension ( filename, "psd" );
}



//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderPSD::isALoadableFileFormat(io::IReadFile* file) const
{
	if (!file)
		return false;

	uint8_t type[3];
	file->read(&type, sizeof(uint8_t)*3);
	return (type[2]==2); // we currently only handle tgas of type 2.
}



//! creates a surface from the file
IImage* CImageLoaderPSD::loadImage(io::IReadFile* file) const
{
	uint32_t* imageData = 0;

	PsdHeader header;
	file->read(&header, sizeof(PsdHeader));

#ifndef __BIG_ENDIAN__
	header.version = os::Byteswap::byteswap(header.version);
	header.channels = os::Byteswap::byteswap(header.channels);
	header.height = os::Byteswap::byteswap(header.height);
	header.width = os::Byteswap::byteswap(header.width);
	header.depth = os::Byteswap::byteswap(header.depth);
	header.mode = os::Byteswap::byteswap(header.mode);
#endif

	if (header.signature[0] != '8' ||
		header.signature[1] != 'B' ||
		header.signature[2] != 'P' ||
		header.signature[3] != 'S')
		return 0;

	if (header.version != 1)
	{
		os::Printer::log("Unsupported PSD file version", file->getFileName().c_str(), ELL_ERROR);
		return 0;
	}

	if (header.mode != 3 || header.depth != 8)
	{
		os::Printer::log("Unsupported PSD color mode or depth.\n", file->getFileName().c_str(), ELL_ERROR);
		return 0;
	}

	// skip color mode data

	uint32_t l;
	file->read(&l, sizeof(uint32_t));
#ifndef __BIG_ENDIAN__
	l = os::Byteswap::byteswap(l);
#endif
	if (!file->seek(l, true))
	{
		os::Printer::log("Error seeking file pos to image resources.\n", file->getFileName().c_str(), ELL_ERROR);
		return 0;
	}

	// skip image resources

	file->read(&l, sizeof(uint32_t));
#ifndef __BIG_ENDIAN__
	l = os::Byteswap::byteswap(l);
#endif
	if (!file->seek(l, true))
	{
		os::Printer::log("Error seeking file pos to layer and mask.\n", file->getFileName().c_str(), ELL_ERROR);
		return 0;
	}

	// skip layer & mask

	file->read(&l, sizeof(uint32_t));
#ifndef __BIG_ENDIAN__
	l = os::Byteswap::byteswap(l);
#endif
	if (!file->seek(l, true))
	{
		os::Printer::log("Error seeking file pos to image data section.\n", file->getFileName().c_str(), ELL_ERROR);
		return 0;
	}

	// read image data

	uint16_t compressionType;
	file->read(&compressionType, sizeof(uint16_t));
#ifndef __BIG_ENDIAN__
	compressionType = os::Byteswap::byteswap(compressionType);
#endif

	if (compressionType != 1 && compressionType != 0)
	{
		os::Printer::log("Unsupported psd compression mode.\n", file->getFileName().c_str(), ELL_ERROR);
		return 0;
	}

	// create image data block

	imageData = new uint32_t[header.width * header.height];

	bool res = false;

	if (compressionType == 0)
		res = readRawImageData(file, header, imageData); // RAW image data
	else
		res = readRLEImageData(file, header, imageData); // RLE compressed data

	video::IImage* image = 0;

	if (res)
	{
		// create surface
		image = new CImage(ECF_A8R8G8B8,
			core::dimension2d<uint32_t>(header.width, header.height), imageData);
	}

	if (!image)
		delete [] imageData;
	imageData = 0;

	return image;
}


bool CImageLoaderPSD::readRawImageData(io::IReadFile* file, const PsdHeader& header, uint32_t* imageData) const
{
	uint8_t* tmpData = new uint8_t[header.width * header.height];

	for (int32_t channel=0; channel<header.channels && channel < 3; ++channel)
	{
		if (!file->read(tmpData, sizeof(char) * header.width * header.height))
		{
			os::Printer::log("Error reading color channel\n", file->getFileName().c_str(), ELL_ERROR);
			break;
		}

		int16_t shift = getShiftFromChannel((char)channel, header);
		if (shift != -1)
		{
			uint32_t mask = 0xff << shift;

			for (uint32_t x=0; x<header.width; ++x)
			{
				for (uint32_t y=0; y<header.height; ++y)
				{
					int32_t index = x + y*header.width;
					imageData[index] = ~(~imageData[index] | mask);
					imageData[index] |= tmpData[index] << shift;
				}
			}
		}

	}

	delete [] tmpData;
	return true;
}


bool CImageLoaderPSD::readRLEImageData(io::IReadFile* file, const PsdHeader& header, uint32_t* imageData) const
{
	/*	If the compression code is 1, the image data
		starts with the byte counts for all the scan lines in the channel
		(LayerBottom LayerTop), with each count stored as a two
		byte value. The RLE compressed data follows, with each scan line
		compressed separately. The RLE compression is the same compres-sion
		algorithm used by the Macintosh ROM routine PackBits, and
		the TIFF standard.
		If the Layer's Size, and therefore the data, is odd, a pad byte will
		be inserted at the end of the row.
	*/

	/*
	A pseudo code fragment to unpack might look like this:

	Loop until you get the number of unpacked bytes you are expecting:
		Read the next source byte into n.
		If n is between 0 and 127 inclusive, copy the next n+1 bytes literally.
		Else if n is between -127 and -1 inclusive, copy the next byte -n+1
		times.
		Else if n is -128, noop.
	Endloop

	In the inverse routine, it is best to encode a 2-byte repeat run as a replicate run
	except when preceded and followed by a literal run. In that case, it is best to merge
	the three runs into one literal run. Always encode 3-byte repeats as replicate runs.
	That is the essence of the algorithm. Here are some additional rules:
	- Pack each row separately. Do not compress across row boundaries.
	- The number of uncompressed bytes per row is defined to be (ImageWidth + 7)
	/ 8. If the uncompressed bitmap is required to have an even number of bytes per
	row, decompress into word-aligned buffers.
	- If a run is larger than 128 bytes, encode the remainder of the run as one or more
	additional replicate runs.
	When PackBits data is decompressed, the result should be interpreted as per com-pression
	type 1 (no compression).
	*/

	uint8_t* tmpData = new uint8_t[header.width * header.height];
	uint16_t *rleCount= new uint16_t [header.height * header.channels];

	int32_t size=0;

	for (uint32_t y=0; y<header.height * header.channels; ++y)
	{
		if (!file->read(&rleCount[y], sizeof(uint16_t)))
		{
			delete [] tmpData;
			delete [] rleCount;
			os::Printer::log("Error reading rle rows\n", file->getFileName().c_str(), ELL_ERROR);
			return false;
		}

#ifndef __BIG_ENDIAN__
		rleCount[y] = os::Byteswap::byteswap(rleCount[y]);
#endif
		size += rleCount[y];
	}

	int8_t *buf = new int8_t[size];
	if (!file->read(buf, size))
	{
		delete [] rleCount;
		delete [] buf;
		delete [] tmpData;
		os::Printer::log("Error reading rle rows\n", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}

	uint16_t *rcount=rleCount;

	int8_t rh;
	uint16_t bytesRead;
	uint8_t *dest;
	int8_t *pBuf = buf;

	// decompress packbit rle

	for (int32_t channel=0; channel<header.channels; channel++)
	{
		for (uint32_t y=0; y<header.height; ++y, ++rcount)
		{
			bytesRead=0;
			dest = &tmpData[y*header.width];

			while (bytesRead < *rcount)
			{
				rh = *pBuf++;
				++bytesRead;

				if (rh >= 0)
				{
					++rh;

					while (rh--)
					{
						*dest = *pBuf++;
						++bytesRead;
						++dest;
					}
				}
				else
				if (rh > -128)
				{
					rh = -rh +1;

					while (rh--)
					{
						*dest = *pBuf;
						++dest;
					}

					++pBuf;
					++bytesRead;
				}
			}
		}

		int16_t shift = getShiftFromChannel((char)channel, header);

		if (shift != -1)
		{
			uint32_t mask = 0xff << shift;

			for (uint32_t x=0; x<header.width; ++x)
				for (uint32_t y=0; y<header.height; ++y)
				{
					int32_t index = x + y*header.width;
					imageData[index] = ~(~imageData[index] | mask);
					imageData[index] |= tmpData[index] << shift;
				}
		}
	}

	delete [] rleCount;
	delete [] buf;
	delete [] tmpData;

	return true;
}


int16_t CImageLoaderPSD::getShiftFromChannel(char channelNr, const PsdHeader& header) const
{
	switch(channelNr)
	{
	case 0:
		return 16;  // red
	case 1:
		return 8;   // green
	case 2:
		return 0;   // blue
	case 3:
		return header.channels == 4 ? 24 : -1;	// ?
	case 4:
		return 24;  // alpha
	default:
		return -1;
	}
}



//! creates a loader which is able to load tgas
IImageLoader* createImageLoaderPSD()
{
	return new CImageLoaderPSD();
}


} // end namespace video
} // end namespace irr

#endif

