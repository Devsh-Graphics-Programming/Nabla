// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderTGA.h"

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_

#include "IReadFile.h"
#include "os.h"
#include "irr/asset/format/convertColor.h"
#include "irr/asset/ICPUImageView.h"


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
	uint8_t* data = _IRR_NEW_ARRAY(uint8_t, imageSize);
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
static void convertColorFlip(const core::smart_refctd_ptr<ICPUImage> &image, const T *src, bool flip)
{
	const T *in = (const T *) src;
	T *out = (T *) image->getBuffer()->getPointer();
	const auto extent = image->getCreationParameters().extent;
	const auto regionBufferRowLenght = image->getRegions().begin()->bufferRowLength;

	irr::core::vector3d size = 
	{
		regionBufferRowLenght > 0 ? regionBufferRowLenght : extent.width,
		extent.height,
		extent.depth
	};

	auto stride = image->getRegions().begin()->bufferRowLength;   
	auto channels = getFormatChannelCount(image->getCreationParameters().format);
	
	if (flip)
		out += size.X * size.Y * channels;
	
	for (int y = 0; y < size.Y; ++y) {
		if (flip)
			out -= stride;
		
		const void *src_container[4] = {in, nullptr, nullptr, nullptr};
		video::convertColor<srcFormat, destFormat>(src_container, out, stride, size);
		in += stride;
		
		if (!flip)
			out += stride;
	}
}

//! creates a surface from the file
asset::SAssetBundle CImageLoaderTGA::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	STGAHeader header;
	uint32_t *palette = nullptr;

	_file->read(&header, sizeof(STGAHeader));

	// skip image identification field
	if (header.IdLength)
		_file->seek(header.IdLength, true);

	if (header.ColorMapType)
	{
		// create 32 bit palette
		palette = _IRR_NEW_ARRAY(uint32_t, header.ColorMapLength);

		// read color map
		uint8_t* colorMap = _IRR_NEW_ARRAY(uint8_t, header.ColorMapEntrySize / 8 * header.ColorMapLength);
		_file->read(colorMap,header.ColorMapEntrySize/8 * header.ColorMapLength);
		
		// convert to 32-bit palette
		const void *src_container[4] = {colorMap, nullptr, nullptr, nullptr};
		switch ( header.ColorMapEntrySize )
		{
			case 16:
				video::convertColor<EF_A1R5G5B5_UNORM_PACK16, EF_R8G8B8A8_SRGB>(src_container, palette, header.ColorMapLength, 0u);
				break;
			case 24:
				video::convertColor<EF_B8G8R8_SRGB, EF_R8G8B8A8_SRGB>(src_container, palette, header.ColorMapLength, 0u);
				break;
			case 32:
				video::convertColor<EF_B8G8R8A8_SRGB, EF_R8G8B8A8_SRGB>(src_container, palette, header.ColorMapLength, 0u);
				break;
		}
		delete [] colorMap;
	}

	ICPUImage::SCreationParams imgInfo;
	imgInfo.type = ICPUImage::ET_2D;
	imgInfo.extent.width = header.ImageWidth;
	imgInfo.extent.height = header.ImageHeight;
	imgInfo.extent.depth = 1u; // not sure about it
	imgInfo.mipLevels = 1u;
	imgInfo.arrayLayers = 1u;
	imgInfo.samples = ICPUImage::ESCF_1_BIT;
	imgInfo.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);

	core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imgInfo));

	if (!image)
		return {};

	auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
	core::smart_refctd_ptr<ICPUBuffer> texelBuffer = nullptr;

	ICPUImage::SBufferCopy& region = regions->front();

	region.imageSubresource.mipLevel = 0u;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.bufferOffset = 0u;
	region.bufferImageHeight = 0u;
	region.imageOffset = { 0u, 0u, 0u };
	region.imageExtent = image->getCreationParameters().extent;

	// read image
	uint8_t* data = nullptr;
	size_t endBufferSize = {};
	
	switch (header.ImageType)
	{
		case 1: // Uncompressed color-mapped image
		case 2: // Uncompressed RGB image
		case 3: // Uncompressed grayscale image
			{
				region.bufferRowLength = calcPitchInBlocks(region.imageExtent.width, getTexelOrBlockBytesize(EF_R8G8B8_SRGB));
				const int32_t imageSize = endBufferSize = region.imageExtent.height * region.bufferRowLength * header.PixelDepth / 8;
				texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(imageSize);
				data = _IRR_NEW_ARRAY(uint8_t, imageSize);
				_file->read(data, imageSize);
			}
			break;
		
		case 10: // Run-length encoded (RLE) true color image
		{
			region.bufferRowLength = calcPitchInBlocks(region.imageExtent.width, getTexelOrBlockBytesize(EF_A1R5G5B5_UNORM_PACK16));
			const auto bufferSize = endBufferSize = region.imageExtent.height * region.bufferRowLength * header.PixelDepth / 8;
			texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(bufferSize);
			data = loadCompressedImage(_file, header);
			break;
		}
		
		case 0:
			{
				os::Printer::log("The given TGA doesn't have image data", _file->getFileName().c_str(), ELL_ERROR);
				if (palette)
					delete [] palette;

                return {};
			}
		
		default:
			{
				os::Printer::log("Unsupported TGA file type", _file->getFileName().c_str(), ELL_ERROR);
				if (palette)
					delete [] palette;

                return {};
			}
	}

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
					
                    return {};
				}
				
				imgInfo.format = EF_R8G8B8_SRGB;

				// Targa formats needs two y-axis flips. The first is a flip to get the Y conforms to OpenGL coords.
				// The second flip is defined from within the .tga file itself (header.ImageDescriptor & 0x20).
				if (flip) {
					// Two flips (OpenGL + Targa) = no flipping. Don't flip the image at all in that case
					convertColorFlip<uint8_t, EF_R8_SRGB, EF_R8_SRGB>(image, data, false);
				}
				else {
					// Do an OpenGL flip
					convertColorFlip<uint8_t, EF_R8_SRGB, EF_R8_SRGB>(image, data, true);
				}

				const void* planarData[] = { data , nullptr, nullptr, nullptr};
				const size_t wholeSize = region.imageExtent.height * region.bufferRowLength * header.PixelDepth / 8;
				const auto wholeSizeInBytes = wholeSize * getTexelOrBlockBytesize(EF_R8G8B8_SRGB);
				void* outRGBData = _IRR_NEW_ARRAY(uint8_t, wholeSizeInBytes);

				video::convertColor<EF_R8_SRGB, EF_R8G8B8_SRGB>(planarData, outRGBData, wholeSize, *reinterpret_cast<core::vector3d<uint32_t>*>(&region.imageExtent));

				memcpy(data, outRGBData, wholeSizeInBytes);
				_IRR_DELETE_ARRAY(outRGBData, wholeSizeInBytes);
			}
			break;
		case 16:
			{
				imgInfo.format = asset::EF_A1R5G5B5_UNORM_PACK16;

				if (flip)
					convertColorFlip<uint8_t, EF_A1R5G5B5_UNORM_PACK16, EF_A1R5G5B5_UNORM_PACK16>(image, data, false);
				else
					convertColorFlip<uint8_t, EF_A1R5G5B5_UNORM_PACK16, EF_A1R5G5B5_UNORM_PACK16>(image, data, true);
			}
			break;
		case 24:
			{
				imgInfo.format = asset::EF_R8G8B8_SRGB;
				
				if (flip)
					convertColorFlip<uint8_t, EF_B8G8R8_SRGB, EF_R8G8B8_SRGB>(image, data, false);
				else
					convertColorFlip<uint8_t, EF_B8G8R8_SRGB, EF_R8G8B8_SRGB>(image, data, true);
			}
			break;
		case 32:
			{
				imgInfo.format = asset::EF_R8G8B8A8_SRGB;

				if (flip)
					convertColorFlip<uint8_t, EF_B8G8R8A8_SRGB, EF_R8G8B8A8_SRGB>(image, data, false);
				else
					convertColorFlip<uint8_t, EF_B8G8R8A8_SRGB, EF_R8G8B8A8_SRGB>(image, data, true);
			}
			break;
		default:
			os::Printer::log("Unsupported TGA format", _file->getFileName().c_str(), ELL_ERROR);
			break;
	}

	memcpy(texelBuffer->getPointer(), data, endBufferSize);
	image->setBufferAndRegions(std::move(texelBuffer), regions);

	delete [] data;
	delete [] palette;

    return SAssetBundle({std::move(image)});
}

} // end namespace video
} // end namespace irr

#endif

