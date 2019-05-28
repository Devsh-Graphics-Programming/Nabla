// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderTGA.h"

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_

#include "IReadFile.h"
#include "os.h"
#include "irr/video/convertColor.h"
#include "irr/asset/CImageData.h"
#include "irr/asset/ICPUTexture.h"


namespace irr
{
namespace asset
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
	_file->seek(_file->getSize() - sizeof(STGAFooter));
	_file->read(&footer, sizeof(STGAFooter));
	
	// 16 bytes for "TRUEVISION-XFILE", 17th byte is '.', and the 18th byte contains '\0'.
	if (strncmp(footer.Signature, "TRUEVISION-XFILE.", 18u) != 0)
	{
		os::Printer::log("Invalid (non-TGA) file!", ELL_ERROR);
		return false;
	}
	
	if (footer.ExtensionOffset == 0)
		os::Printer::log("Gamma information is not present!", ELL_ERROR);
	else
	{
		STGAExtensionArea extension;
		_file->seek(footer.ExtensionOffset);
		_file->read(&extension, sizeof(STGAExtensionArea));
		
		float gamma = extension.Gamma;
		
		if (gamma > 0.0f)
		{
			// TODO: Pass gamma to loadAsset()?
		}
		else
			os::Printer::log("Gamma information is not present!", ELL_ERROR);
	}
	
    _file->seek(prevPos);
	
	return true;
}

// convertColorFlip() does color conversion as well as taking care of properly flipping the given image.
template <typename T, E_FORMAT srcFormat, E_FORMAT destFormat>
static void convertColorFlip(asset::CImageData **image, const T *src, bool flip)
{
	const T *in = (const T *) src;
	T *out = (T *) (*image)->getData();
	
	auto size     = (*image)->getSize();
	auto stride   = (*image)->getPitchIncludingAlignment();
	auto channels = (*image)->getBitsPerPixel() / 8;
	
	if (flip)
		out += size.X * size.Y * channels;
	
	for (int y = 0; y < size.Y; ++y) {
		if (flip)
			out -= stride;
		
		const void *src_container[4] = {in, nullptr, nullptr, nullptr};
		video::convertColor<srcFormat, destFormat>(src_container, out, 1, size.X, size);
		in += stride;
		
		if (!flip)
			out += stride;
	}
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
		palette = new uint32_t[header.ColorMapLength];

		// read color map
		uint8_t * colorMap = new uint8_t[header.ColorMapEntrySize/8 * header.ColorMapLength];
		_file->read(colorMap,header.ColorMapEntrySize/8 * header.ColorMapLength);
		
		// convert to 32-bit palette
		const void *src_container[4] = {colorMap, nullptr, nullptr, nullptr};
		switch ( header.ColorMapEntrySize )
		{
			case 16:
				video::convertColor<EF_A1R5G5B5_UNORM_PACK16, EF_R8G8B8A8_SRGB>(src_container, palette, 1, header.ColorMapLength, 0u);
				break;
			case 24:
				video::convertColor<EF_B8G8R8_SRGB, EF_R8G8B8A8_SRGB>(src_container, palette, 1, header.ColorMapLength, 0u);
				break;
			case 32:
				video::convertColor<EF_B8G8R8A8_SRGB, EF_R8G8B8A8_SRGB>(src_container, palette, 1, header.ColorMapLength, 0u);
				break;
		}
		delete [] colorMap;
	}

	core::vector<asset::CImageData*> images;
	// read image
	uint8_t* data = 0;
	
	switch (header.ImageType)
	{
		case 1: // Uncompressed color-mapped image
		case 2: // Uncompressed RGB image
		case 3: // Uncompressed grayscale image
			{
				const int32_t imageSize = header.ImageHeight * header.ImageWidth * header.PixelDepth/8;
				data = new uint8_t[imageSize];
				_file->read(data, imageSize);
			}
			break;
		
		case 10: // Run-length encoded (RLE) true color image
			data = loadCompressedImage(_file, header);
			break;
		
		case 0:
			{
				os::Printer::log("The given TGA doesn't have image data", _file->getFileName().c_str(), ELL_ERROR);
				if (palette)
					delete [] palette;

				return nullptr;
			}
		
		default:
			{
				os::Printer::log("Unsupported TGA file type", _file->getFileName().c_str(), ELL_ERROR);
				if (palette)
					delete [] palette;

				return nullptr;
			}
	}

    asset::CImageData* image = 0;

	uint32_t nullOffset[3] = {0,0,0};
	uint32_t imageSize[3] = {header.ImageWidth,header.ImageHeight,1};
	bool flip = (header.ImageDescriptor & 0x20) == 0;
	
	switch(header.PixelDepth)
	{
		case 8:
			{
				if (header.ImageType != 3)
				{
					os::Printer::log("Loading 8-bit non-grayscale is NOT supported.", ELL_ERROR);
					if (palette) delete [] palette;
					if (data)    delete [] data;
					
					return nullptr;
				}
				
				image = new asset::CImageData(nullptr,nullOffset,imageSize,0,asset::EF_R8_SRGB);
				if (image) {
					// Targa formats needs two y-axis flips. The first is a flip to get the Y conforms to OpenGL coords.
					// The second flip is defined from within the .tga file itself (header.ImageDescriptor & 0x20).
					if (flip) {
						// Two flips (OpenGL + Targa) = no flipping. Don't flip the image at all in that case
						convertColorFlip<uint8_t, EF_R8_SRGB, EF_R8_SRGB>(&image, data, false);
					}
					else {
						// Do an OpenGL flip
						convertColorFlip<uint8_t, EF_R8_SRGB, EF_R8_SRGB>(&image, data, true);
					}
				}
			}
			break;
		case 16:
			{
				image = new asset::CImageData(nullptr,nullOffset,imageSize,0, asset::EF_A1R5G5B5_UNORM_PACK16);
				if (image) {
					if (flip)
						convertColorFlip<uint8_t, EF_A1R5G5B5_UNORM_PACK16, EF_A1R5G5B5_UNORM_PACK16>(&image, data, false);
					else
						convertColorFlip<uint8_t, EF_A1R5G5B5_UNORM_PACK16, EF_A1R5G5B5_UNORM_PACK16>(&image, data, true);
				}
			}
			break;
		case 24:
			{
				image = new asset::CImageData(nullptr,nullOffset,imageSize,0,asset::EF_R8G8B8_SRGB);
				if (image)
					if (flip)
						convertColorFlip<uint8_t, EF_B8G8R8_SRGB, EF_R8G8B8_SRGB>(&image, data, false);
					else
						convertColorFlip<uint8_t, EF_B8G8R8_SRGB, EF_R8G8B8_SRGB>(&image, data, true);
			}
			break;
		case 32:
			{
				image = new asset::CImageData(nullptr,nullOffset,imageSize,0,asset::EF_R8G8B8A8_SRGB);
				if (image)
					if (flip)
						convertColorFlip<uint8_t, EF_B8G8R8A8_SRGB, EF_R8G8B8A8_SRGB>(&image, data, false);
					else
						convertColorFlip<uint8_t, EF_B8G8R8A8_SRGB, EF_R8G8B8A8_SRGB>(&image, data, true);
			}
			break;
		default:
			os::Printer::log("Unsupported TGA format", _file->getFileName().c_str(), ELL_ERROR);
			break;
	}
	
	images.push_back(image);


	delete [] data;
	delete [] palette;

    asset::ICPUTexture* tex = asset::ICPUTexture::create(images);
    for (auto& img : images)
        img->drop();
    return tex;
}

} // end namespace video
} // end namespace irr

#endif

