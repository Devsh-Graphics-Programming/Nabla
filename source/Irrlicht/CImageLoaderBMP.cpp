// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderBMP.h"

#ifdef _IRR_COMPILE_WITH_BMP_LOADER_

#include "IReadFile.h"
#include "SColor.h"
#include "CColorConverter.h"
#include "CImage.h"
#include "os.h"
#include "irrString.h"

namespace irr
{
namespace video
{


//! constructor
CImageLoaderBMP::CImageLoaderBMP()
{
	#ifdef _DEBUG
	setDebugName("CImageLoaderBMP");
	#endif
}


//! returns true if the file maybe is able to be loaded by this class
//! based on the file extension (e.g. ".tga")
bool CImageLoaderBMP::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension ( filename, "bmp" );
}


//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderBMP::isALoadableFileFormat(io::IReadFile* file) const
{
	uint16_t headerID;
	file->read(&headerID, sizeof(uint16_t));
#ifdef __BIG_ENDIAN__
	headerID = os::Byteswap::byteswap(headerID);
#endif
	return headerID == 0x4d42;
}


void CImageLoaderBMP::decompress8BitRLE(uint8_t*& bmpData, int32_t size, int32_t width, int32_t height, int32_t pitch) const
{
	uint8_t* p = bmpData;
	uint8_t* newBmp = new uint8_t[(width+pitch)*height];
	uint8_t* d = newBmp;
	uint8_t* destEnd = newBmp + (width+pitch)*height;
	int32_t line = 0;

	while (bmpData - p < size && d < destEnd)
	{
		if (*p == 0)
		{
			++p;

			switch(*p)
			{
			case 0: // end of line
				++p;
				++line;
				d = newBmp + (line*(width+pitch));
				break;
			case 1: // end of bmp
				delete [] bmpData;
				bmpData = newBmp;
				return;
			case 2:
				++p; d +=(uint8_t)*p;  // delta
				++p; d += ((uint8_t)*p)*(width+pitch);
				++p;
				break;
			default:
				{
					// absolute mode
					int32_t count = (uint8_t)*p; ++p;
					int32_t readAdditional = ((2-(count%2))%2);
					int32_t i;

					for (i=0; i<count; ++i)
					{
						*d = *p;
						++p;
						++d;
					}

					for (i=0; i<readAdditional; ++i)
						++p;
				}
			}
		}
		else
		{
			int32_t count = (uint8_t)*p; ++p;
			uint8_t color = *p; ++p;
			for (int32_t i=0; i<count; ++i)
			{
				*d = color;
				++d;
			}
		}
	}

	delete [] bmpData;
	bmpData = newBmp;
}


void CImageLoaderBMP::decompress4BitRLE(uint8_t*& bmpData, int32_t size, int32_t width, int32_t height, int32_t pitch) const
{
	int32_t lineWidth = (width+1)/2+pitch;
	uint8_t* p = bmpData;
	uint8_t* newBmp = new uint8_t[lineWidth*height];
	uint8_t* d = newBmp;
	uint8_t* destEnd = newBmp + lineWidth*height;
	int32_t line = 0;
	int32_t shift = 4;

	while (bmpData - p < size && d < destEnd)
	{
		if (*p == 0)
		{
			++p;

			switch(*p)
			{
			case 0: // end of line
				++p;
				++line;
				d = newBmp + (line*lineWidth);
				shift = 4;
				break;
			case 1: // end of bmp
				delete [] bmpData;
				bmpData = newBmp;
				return;
			case 2:
				{
					++p;
					int32_t x = (uint8_t)*p; ++p;
					int32_t y = (uint8_t)*p; ++p;
					d += x/2 + y*lineWidth;
					shift = x%2==0 ? 4 : 0;
				}
				break;
			default:
				{
					// absolute mode
					int32_t count = (uint8_t)*p; ++p;
					int32_t readAdditional = ((2-((count)%2))%2);
					int32_t readShift = 4;
					int32_t i;

					for (i=0; i<count; ++i)
					{
						int32_t color = (((uint8_t)*p) >> readShift) & 0x0f;
						readShift -= 4;
						if (readShift < 0)
						{
							++*p;
							readShift = 4;
						}

						uint8_t mask = 0x0f << shift;
						*d = (*d & (~mask)) | ((color << shift) & mask);

						shift -= 4;
						if (shift < 0)
						{
							shift = 4;
							++d;
						}

					}

					for (i=0; i<readAdditional; ++i)
						++p;
				}
			}
		}
		else
		{
			int32_t count = (uint8_t)*p; ++p;
			int32_t color1 = (uint8_t)*p; color1 = color1 & 0x0f;
			int32_t color2 = (uint8_t)*p; color2 = (color2 >> 4) & 0x0f;
			++p;

			for (int32_t i=0; i<count; ++i)
			{
				uint8_t mask = 0x0f << shift;
				uint8_t toSet = (shift==0 ? color1 : color2) << shift;
				*d = (*d & (~mask)) | (toSet & mask);

				shift -= 4;
				if (shift < 0)
				{
					shift = 4;
					++d;
				}
			}
		}
	}

	delete [] bmpData;
	bmpData = newBmp;
}



//! creates a surface from the file
IImage* CImageLoaderBMP::loadImage(io::IReadFile* file) const
{
	SBMPHeader header;

	file->read(&header, sizeof(header));

#ifdef __BIG_ENDIAN__
	header.Id = os::Byteswap::byteswap(header.Id);
	header.FileSize = os::Byteswap::byteswap(header.FileSize);
	header.BitmapDataOffset = os::Byteswap::byteswap(header.BitmapDataOffset);
	header.BitmapHeaderSize = os::Byteswap::byteswap(header.BitmapHeaderSize);
	header.Width = os::Byteswap::byteswap(header.Width);
	header.Height = os::Byteswap::byteswap(header.Height);
	header.Planes = os::Byteswap::byteswap(header.Planes);
	header.BPP = os::Byteswap::byteswap(header.BPP);
	header.Compression = os::Byteswap::byteswap(header.Compression);
	header.BitmapDataSize = os::Byteswap::byteswap(header.BitmapDataSize);
	header.PixelPerMeterX = os::Byteswap::byteswap(header.PixelPerMeterX);
	header.PixelPerMeterY = os::Byteswap::byteswap(header.PixelPerMeterY);
	header.Colors = os::Byteswap::byteswap(header.Colors);
	header.ImportantColors = os::Byteswap::byteswap(header.ImportantColors);
#endif

	int32_t pitch = 0;

	//! return if the header is false

	if (header.Id != 0x4d42)
		return 0;

	if (header.Compression > 2) // we'll only handle RLE-Compression
	{
		os::Printer::log("Compression mode not supported.", ELL_ERROR);
		return 0;
	}

	// adjust bitmap data size to dword boundary
	header.BitmapDataSize += (4-(header.BitmapDataSize%4))%4;

	// read palette

	long pos = file->getPos();
	int32_t paletteSize = (header.BitmapDataOffset - pos) / 4;

	int32_t* paletteData = 0;
	if (paletteSize)
	{
		paletteData = new int32_t[paletteSize];
		file->read(paletteData, paletteSize * sizeof(int32_t));
#ifdef __BIG_ENDIAN__
		for (int32_t i=0; i<paletteSize; ++i)
			paletteData[i] = os::Byteswap::byteswap(paletteData[i]);
#endif
	}

	// read image data

	if (!header.BitmapDataSize)
	{
		// okay, lets guess the size
		// some tools simply don't set it
		header.BitmapDataSize = static_cast<uint32_t>(file->getSize()) - header.BitmapDataOffset;
	}

	file->seek(header.BitmapDataOffset);

	float t = (header.Width) * (header.BPP / 8.0f);
	int32_t widthInBytes = (int32_t)t;
	t -= widthInBytes;
	if (t!=0.0f)
		++widthInBytes;

	int32_t lineData = widthInBytes + ((4-(widthInBytes%4)))%4;
	pitch = lineData - widthInBytes;

	uint8_t* bmpData = new uint8_t[header.BitmapDataSize];
	file->read(bmpData, header.BitmapDataSize);

	// decompress data if needed
	switch(header.Compression)
	{
	case 1: // 8 bit rle
		decompress8BitRLE(bmpData, header.BitmapDataSize, header.Width, header.Height, pitch);
		break;
	case 2: // 4 bit rle
		decompress4BitRLE(bmpData, header.BitmapDataSize, header.Width, header.Height, pitch);
		break;
	}

	// create surface

	// no default constructor from packed area! ARM problem!
	core::dimension2d<uint32_t> dim;
	dim.Width = header.Width;
	dim.Height = header.Height;

	IImage* image = 0;
	switch(header.BPP)
	{
	case 1:
		image = new CImage(ECF_A1R5G5B5, dim);
		if (image)
			CColorConverter::convert1BitTo16Bit(bmpData, (int16_t*)image->lock(), header.Width, header.Height, pitch, true);
		break;
	case 4:
		image = new CImage(ECF_A1R5G5B5, dim);
		if (image)
			CColorConverter::convert4BitTo16Bit(bmpData, (int16_t*)image->lock(), header.Width, header.Height, paletteData, pitch, true);
		break;
	case 8:
		image = new CImage(ECF_A1R5G5B5, dim);
		if (image)
			CColorConverter::convert8BitTo16Bit(bmpData, (int16_t*)image->lock(), header.Width, header.Height, paletteData, pitch, true);
		break;
	case 16:
		image = new CImage(ECF_A1R5G5B5, dim);
		if (image)
			CColorConverter::convert16BitTo16Bit((int16_t*)bmpData, (int16_t*)image->lock(), header.Width, header.Height, pitch, true);
		break;
	case 24:
		image = new CImage(ECF_R8G8B8, dim);
		if (image)
			CColorConverter::convert24BitTo24Bit(bmpData, (uint8_t*)image->lock(), header.Width, header.Height, pitch, true, true);
		break;
	case 32: // thx to Reinhard Ostermeier
		image = new CImage(ECF_A8R8G8B8, dim);
		if (image)
			CColorConverter::convert32BitTo32Bit((int32_t*)bmpData, (int32_t*)image->lock(), header.Width, header.Height, pitch, true);
		break;
	};
	if (image)
		image->unlock();

	// clean up

	delete [] paletteData;
	delete [] bmpData;

	return image;
}


//! creates a loader which is able to load windows bitmaps
IImageLoader* createImageLoaderBMP()
{
	return new CImageLoaderBMP;
}


} // end namespace video
} // end namespace irr

#endif

