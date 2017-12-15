// Copyright (C) 2007-2012 Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderPPM.h"

#ifdef _IRR_COMPILE_WITH_PPM_LOADER_

#include "IReadFile.h"
#include "CColorConverter.h"
#include "CImage.h"
#include "os.h"
#include "coreutil.h"
#include "string.h"

namespace irr
{
namespace video
{


//! constructor
CImageLoaderPPM::CImageLoaderPPM()
{
	#ifdef _DEBUG
	setDebugName("CImageLoaderPPM");
	#endif
}


//! returns true if the file maybe is able to be loaded by this class
//! based on the file extension (e.g. ".tga")
bool CImageLoaderPPM::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension ( filename, "ppm", "pgm", "pbm" );
}


//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderPPM::isALoadableFileFormat(io::IReadFile* file) const
{
	int8_t id[2]={0};
	file->read(&id, 2);
	return (id[0]=='P' && id[1]>'0' && id[1]<'7');
}


//! creates a surface from the file
std::vector<CImageData*> CImageLoaderPPM::loadImage(io::IReadFile* file) const
{
    std::vector<CImageData*> retval;
	if (file->getSize() < 12)
		return retval;

	int8_t id[2];
	file->read(&id, 2);

	if (id[0]!='P' || id[1]<'1' || id[1]>'6')
		return retval;

	const uint8_t format = id[1] - '0';
	const bool binary = format>3;

	std::string token;
	getNextToken(file, token);

    uint32_t width;
	sscanf(token.c_str(),"%u",&width);

	getNextToken(file, token);
	uint32_t height;
	sscanf(token.c_str(),"%u",&height);

	CImageData* image = NULL;
	uint32_t nullOffset[3] = {0,0,0};
	uint32_t imageSize[3] = {width,height,1};
	uint8_t* data = 0;
	const uint32_t size = width*height;
	if (format==1 || format==4)
	{
		skipToNextToken(file); // go to start of data

		const uint32_t bytesize = size/8+(size & 3)?1:0;
		if (binary)
		{
			if (file->getSize()-file->getPos() < (long)bytesize)
				return retval;
			data = new uint8_t[bytesize];
			file->read(data, bytesize);
		}
		else
		{
			if (file->getSize()-file->getPos() < (long)(2*size)) // optimistic test
				return retval;
			data = new uint8_t[bytesize];
			memset(data, 0, bytesize);
			uint32_t shift=0;
			for (uint32_t i=0; i<size; ++i)
			{
				getNextToken(file, token);
				if (token == "1")
					data[i/8] |= (0x01 << shift);
				if (++shift == 8)
					shift=0;
			}
		}
		image = new CImageData(NULL,nullOffset,imageSize,0,ECF_A1R5G5B5);
		if (image)
			CColorConverter::convert1BitTo16Bit(data, (int16_t*)image->getData(), width, height);
	}
	else
	{
		getNextToken(file, token);
		uint32_t maxDepth;
        sscanf(token.c_str(),"%u",&maxDepth);
		if (maxDepth > 255) // no double bytes yet
			return retval;

		skipToNextToken(file); // go to start of data

		if (format==2 || format==5)
		{
			if (binary)
			{
				if (file->getSize()-file->getPos() < (long)size)
					return retval;
				data = new uint8_t[size];
				file->read(data, size);
				image = new CImageData(NULL,nullOffset,imageSize,0,ECF_A8R8G8B8);
				if (image)
				{
					uint8_t* ptr = (uint8_t*)image->getData();
					for (uint32_t i=0; i<size; ++i)
					{
						*ptr++ = data[i];
						*ptr++ = data[i];
						*ptr++ = data[i];
						*ptr++ = 255;
					}
				}
			}
			else
			{
				if (file->getSize()-file->getPos() < (long)(2*size)) // optimistic test
					return retval;
				image = new CImageData(NULL,nullOffset,imageSize,0,ECF_A8R8G8B8);
				if (image)
				{
					uint8_t* ptr = (uint8_t*)image->getData();
					for (uint32_t i=0; i<size; ++i)
					{
						getNextToken(file, token);
						uint8_t num;
                        sscanf(token.c_str(),"%u",&num);
						*ptr++ = num;
						*ptr++ = num;
						*ptr++ = num;
						*ptr++ = 255;
					}
				}
			}
		}
		else
		{
			const uint32_t bytesize = 3*size;
			if (binary)
			{
				if (file->getSize()-file->getPos() < (long)bytesize)
					return retval;
				data = new uint8_t[bytesize];
				file->read(data, bytesize);
				image = new CImageData(NULL,nullOffset,imageSize,0,ECF_A8R8G8B8);
				if (image)
				{
					uint8_t* ptr = (uint8_t*)image->getData();
					for (uint32_t i=0; i<size; ++i)
					{
						*ptr++ = data[3*i];
						*ptr++ = data[3*i+1];
						*ptr++ = data[3*i+2];
						*ptr++ = 255;
					}
				}
			}
			else
			{
				if (file->getSize()-file->getPos() < (long)(2*bytesize)) // optimistic test
					return retval;
				image = new CImageData(NULL,nullOffset,imageSize,0,ECF_A8R8G8B8);
				if (image)
				{
					uint8_t* ptr = (uint8_t*)image->getData();
					for (uint32_t i=0; i<size; ++i)
					{
						getNextToken(file, token);
                        sscanf(token.c_str(),"%u",ptr++);
						getNextToken(file, token);
                        sscanf(token.c_str(),"%u",ptr++);
						getNextToken(file, token);
                        sscanf(token.c_str(),"%u",ptr++);
						*ptr++ = 255;
					}
				}
			}
		}
	}
	delete [] data;

	retval.push_back(image);
	return retval;
}


//! read the next token from file
void CImageLoaderPPM::getNextToken(io::IReadFile* file, std::string& token) const
{
	token = "";
	int8_t c;
	while(file->getPos()<file->getSize())
	{
		file->read(&c, 1);
		if (c=='#')
		{
			while (c!='\n' && c!='\r' && (file->getPos()<file->getSize()))
				file->read(&c, 1);
		}
		else if (!core::isspace(c))
		{
			token.push_back(c);
			break;
		}
	}
	while(file->getPos()<file->getSize())
	{
		file->read(&c, 1);
		if (c=='#')
		{
			while (c!='\n' && c!='\r' && (file->getPos()<file->getSize()))
				file->read(&c, 1);
		}
		else if (!core::isspace(c))
			token.push_back(c);
		else
			break;
	}
}


//! skip to next token (skip whitespace)
void CImageLoaderPPM::skipToNextToken(io::IReadFile* file) const
{
	int8_t c;
	while(file->getPos()<file->getSize())
	{
		file->read(&c, 1);
		if (c=='#')
		{
			while (c!='\n' && c!='\r' && (file->getPos()<file->getSize()))
				file->read(&c, 1);
		}
		else if (!core::isspace(c))
		{
			file->seek(-1, true); // put back
			break;
		}
	}
}


//! creates a loader which is able to load windows bitmaps
IImageLoader* createImageLoaderPPM()
{
	return new CImageLoaderPPM;
}


} // end namespace video
} // end namespace irr

#endif

