// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderTGA.h"

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_

#include "IReadFile.h"
#include "os.h"
#include "CColorConverter.h"
#include "irr/asset/CImageData.h"
#include "irr/asset/ICPUTexture.h"


namespace irr
{
namespace video
{

//! loads a compressed tga.
uint8_t *CImageLoaderTGA::loadCompressedImage(io::IReadFile *file, const STGAHeader& header) const
{
	// This was written and sent in by Jon Pry, thank you very much!
	// I only changed the formatting a little bit.

	int32_t bytesPerPixel = header.PixelDepth/8;
	int32_t imageSize =  header.ImageHeight * header.ImageWidth * bytesPerPixel;
	uint8_t* data = new uint8_t[imageSize];
	int32_t currentByte = 0;

	while(currentByte < imageSize)
	{
		uint8_t chunkheader = 0;
		file->read(&chunkheader, sizeof(uint8_t)); // Read The Chunk's Header

		if(chunkheader < 128) // If The Chunk Is A 'RAW' Chunk
		{
			chunkheader++; // Add 1 To The Value To Get Total Number Of Raw Pixels

			file->read(&data[currentByte], bytesPerPixel * chunkheader);
			currentByte += bytesPerPixel * chunkheader;
		}
		else
		{
			// thnx to neojzs for some fixes with this code

			// If It's An RLE Header
			chunkheader -= 127; // Subtract 127 To Get Rid Of The ID Bit

			int32_t dataOffset = currentByte;
			file->read(&data[dataOffset], bytesPerPixel);

			currentByte += bytesPerPixel;

			for(int32_t counter = 1; counter < chunkheader; counter++)
			{
				for(int32_t elementCounter=0; elementCounter < bytesPerPixel; elementCounter++)
					data[currentByte + elementCounter] = data[dataOffset + elementCounter];

				currentByte += bytesPerPixel;
			}
		}
	}

	return data;
}



//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderTGA::isALoadableFileFormat(io::IReadFile* _file) const
{
	if (!_file)
		return false;

    const size_t prevPos = _file->getPos();

	STGAFooter footer;
	memset(&footer, 0, sizeof(STGAFooter));
	_file->seek(_file->getSize()-sizeof(STGAFooter));
	_file->read(&footer, sizeof(STGAFooter));
    _file->seek(prevPos);

	if (strcmp(footer.Signature,"TRUEVISION-X_file.")) // very old tgas are refused.
	{
#ifdef _DEBUG
		os::Printer::log("Unsupported, very old TGA", _file->getFileName().c_str(), ELL_ERROR);
#endif // _DEBUG
	    return false;
	}
    else
        return true;
}



//! creates a surface from the file
asset::IAsset* CImageLoaderTGA::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	STGAHeader header;
	uint32_t *palette = 0;

	_file->read(&header, sizeof(STGAHeader));

	// skip image identification field
	if (header.IdLength)
		_file->seek(header.IdLength, true);

	if (header.ColorMapType)
	{
		// create 32 bit palette
		palette = new uint32_t[ header.ColorMapLength];

		// read color map
		uint8_t * colorMap = new uint8_t[header.ColorMapEntrySize/8 * header.ColorMapLength];
		_file->read(colorMap,header.ColorMapEntrySize/8 * header.ColorMapLength);

		// convert to 32-bit palette
		switch ( header.ColorMapEntrySize )
		{
			case 16:
				CColorConverter::convert_A1R5G5B5toA8R8G8B8(colorMap, header.ColorMapLength, palette);
				break;
			case 24:
				CColorConverter::convert_B8G8R8toA8R8G8B8(colorMap, header.ColorMapLength, palette);
				break;
			case 32:
				CColorConverter::convert_B8G8R8A8toA8R8G8B8(colorMap, header.ColorMapLength, palette);
				break;
		}
		delete [] colorMap;
	}

	core::vector<asset::CImageData*> images;
	// read image
	uint8_t* data = 0;

	if (	header.ImageType == 1 || // Uncompressed, color-mapped images.
			header.ImageType == 2 || // Uncompressed, RGB images
			header.ImageType == 3 // Uncompressed, black and white images
		)
	{
		const int32_t imageSize = header.ImageHeight * header.ImageWidth * header.PixelDepth/8;
		data = new uint8_t[imageSize];
	  	_file->read(data, imageSize);
	}
	else
	if(header.ImageType == 10)
	{
		// Runlength encoded RGB images
		data = loadCompressedImage(_file, header);
	}
	else
	{
		os::Printer::log("Unsupported TGA _file type", _file->getFileName().c_str(), ELL_ERROR);

		if (palette)
            delete [] palette;

		return nullptr;
	}

    asset::CImageData* image = 0;

	uint32_t nullOffset[3] = {0,0,0};
	uint32_t imageSize[3] = {header.ImageWidth,header.ImageHeight,1};

	switch(header.PixelDepth)
	{
	case 8:
		{
			if (header.ImageType==3) // grey image
			{
				image = new asset::CImageData(NULL,nullOffset,imageSize,0,EF_R8G8B8_UNORM);
				if (image)
					CColorConverter::convert8BitTo24Bit((uint8_t*)data,
						(uint8_t*)image->getData(),
						header.ImageWidth,header.ImageHeight,
						0, 0, (header.ImageDescriptor&0x20)==0);
			}
			else
			{
				image = new asset::CImageData(NULL,nullOffset,imageSize,0, EF_A1R5G5B5_UNORM_PACK16);
				if (image)
					CColorConverter::convert8BitTo16Bit((uint8_t*)data,
						(int16_t*)image->getData(),
						header.ImageWidth,header.ImageHeight,
						(int32_t*) palette, 0,
						(header.ImageDescriptor&0x20)==0);
			}
		}
		break;
	case 16:
		image = new asset::CImageData(NULL,nullOffset,imageSize,0, EF_A1R5G5B5_UNORM_PACK16);
		if (image)
			CColorConverter::convert16BitTo16Bit((int16_t*)data,
				(int16_t*)image->getData(), header.ImageWidth,	header.ImageHeight, 0, (header.ImageDescriptor&0x20)==0);
		break;
	case 24:
			image = new asset::CImageData(NULL,nullOffset,imageSize,0,EF_R8G8B8_UNORM);
			if (image)
				CColorConverter::convert24BitTo24Bit(
					(uint8_t*)data, (uint8_t*)image->getData(), header.ImageWidth, header.ImageHeight, 0, (header.ImageDescriptor&0x20)==0, true);
		break;
	case 32:
			image = new asset::CImageData(NULL,nullOffset,imageSize,0,EF_B8G8R8A8_UNORM);
			if (image)
				CColorConverter::convert32BitTo32Bit((int32_t*)data,
					(int32_t*)image->getData(), header.ImageWidth, header.ImageHeight, 0, (header.ImageDescriptor&0x20)==0);
		break;
	default:
		os::Printer::log("Unsupported TGA format", _file->getFileName().c_str(), ELL_ERROR);
		break;
	}
	images.push_back(image);


	delete [] data;
	delete [] palette;

    asset::ICPUTexture* tex = asset::ICPUTexture::create(images);
    for (auto img : images)
        img->drop();
    return tex;
}


} // end namespace video
} // end namespace irr

#endif

