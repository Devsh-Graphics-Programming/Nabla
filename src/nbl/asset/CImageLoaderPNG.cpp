// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CImageLoaderPNG.h"

#ifdef _NBL_COMPILE_WITH_PNG_LOADER_

#ifdef _NBL_COMPILE_WITH_LIBPNG_
#   include "libpng/png.h"
#endif // _NBL_COMPILE_WITH_LIBPNG_

#include "nbl/asset/ICPUImageView.h"

#include "nbl/asset/IImageAssetHandlerBase.h"

#include "CReadFile.h"
#include "os.h"

namespace nbl
{
namespace asset
{

#ifdef _NBL_COMPILE_WITH_LIBPNG_
// PNG function for error handling
static void png_cpexcept_error(png_structp png_ptr, png_const_charp msg)
{
	os::Printer::log("PNG fatal error", msg, ELL_ERROR);
	longjmp(png_jmpbuf(png_ptr), 1);
}

// PNG function for warning handling
static void png_cpexcept_warn(png_structp png_ptr, png_const_charp msg)
{
	os::Printer::log("PNG warning", msg, ELL_WARNING);
}

// PNG function for file reading
void PNGAPI user_read_data_fcn(png_structp png_ptr, png_bytep data, png_size_t length)
{
	png_size_t check;

	// changed by zola {
	io::IReadFile* file=(io::IReadFile*)png_get_io_ptr(png_ptr);
	check=(png_size_t) file->read((void*)data,(uint32_t)length);
	// }

	if (check != length)
		png_error(png_ptr, "Read Error");
}
#endif // _NBL_COMPILE_WITH_LIBPNG_


//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderPng::isALoadableFileFormat(io::IReadFile* _file) const
{
#ifdef _NBL_COMPILE_WITH_LIBPNG_
	if (!_file)
		return false;

    const size_t prevPos = _file->getPos();

	png_byte buffer[8];
	// Read the first few bytes of the PNG _file
    if (_file->read(buffer, 8) != 8)
    {
        _file->seek(prevPos);
        return false;
    }

    _file->seek(prevPos);
	// Check if it really is a PNG _file
	return !png_sig_cmp(buffer, 0, 8);
#else
	return false;
#endif // _NBL_COMPILE_WITH_LIBPNG_
}


// load in the image data
asset::SAssetBundle CImageLoaderPng::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
#ifdef _NBL_COMPILE_WITH_LIBPNG_
	if (!_file)
        return {};

	uint32_t imageSize[3] = { 1,1,1 };
	uint32_t& Width = imageSize[0];
	uint32_t& Height = imageSize[1];
	//Used to point to image rows
	uint8_t** RowPointers = 0;

	png_byte buffer[8];
	// Read the first few bytes of the PNG _file
	if( _file->read(buffer, 8) != 8 )
	{
		os::Printer::log("LOAD PNG: can't read _file\n", _file->getFileName().c_str(), ELL_ERROR);
        return {};
	}

	// Check if it really is a PNG _file
	if( png_sig_cmp(buffer, 0, 8) )
	{
		os::Printer::log("LOAD PNG: not really a png\n", _file->getFileName().c_str(), ELL_ERROR);
        return {};
	}

	// Allocate the png read struct
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
		nullptr, (png_error_ptr)png_cpexcept_error, (png_error_ptr)png_cpexcept_warn);
	if (!png_ptr)
	{
		os::Printer::log("LOAD PNG: Internal PNG create read struct failure\n", _file->getFileName().c_str(), ELL_ERROR);
        return {};
	}

	// Allocate the png info struct
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		os::Printer::log("LOAD PNG: Internal PNG create info struct failure\n", _file->getFileName().c_str(), ELL_ERROR);
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return {};
	}

	// for proper error handling
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
		if (RowPointers)
			_NBL_DELETE_ARRAY(RowPointers, Height);
        return {};
	}

	// changed by zola so we don't need to have public FILE pointers
	png_set_read_fn(png_ptr, _file, user_read_data_fcn);

	png_set_sig_bytes(png_ptr, 8); // Tell png that we read the signature

	png_read_info(png_ptr, info_ptr); // Read the info section of the png _file

	int32_t BitDepth;
	int32_t ColorType;
	{
		// Use temporary variables to avoid passing casted pointers
		png_uint_32 w,h;
		// Extract info
		png_get_IHDR(png_ptr, info_ptr,
			&w, &h,
			&BitDepth, &ColorType, nullptr, nullptr, nullptr);
		Width=w;
		Height=h;
	}
	
	if (ColorType == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);

	// Convert low bit colors to 8 bit colors
	if (BitDepth < 8)
	{
		switch (ColorType) {
			case PNG_COLOR_TYPE_GRAY:
			case PNG_COLOR_TYPE_GRAY_ALPHA:
				png_set_expand_gray_1_2_4_to_8(png_ptr);
				break;
			default:
				png_set_packing(png_ptr);
		}
	}
	
	// Add an alpha channel if transparency information is found in tRNS chunk
	if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png_ptr);

	// Convert high bit colors to 8 bit colors
	if (BitDepth == 16)
		png_set_strip_16(png_ptr);

	int intent;
	const double screen_gamma = 2.2;

	if (png_get_sRGB(png_ptr, info_ptr, &intent))
		png_set_gamma(png_ptr, screen_gamma, 0.45455);
	else
	{
		double image_gamma;
		if (png_get_gAMA(png_ptr, info_ptr, &image_gamma))
			png_set_gamma(png_ptr, screen_gamma, image_gamma);
		else
			png_set_gamma(png_ptr, screen_gamma, 0.45455);
	}

	// Update the changes in between, as we need to get the new color type
	// for proper processing of the RGBA type
	png_read_update_info(png_ptr, info_ptr);
	{
		// Use temporary variables to avoid passing casted pointers
		png_uint_32 w,h;
		// Extract info
		png_get_IHDR(png_ptr, info_ptr, &w, &h, &BitDepth, &ColorType, nullptr, nullptr, nullptr);
		Width = w;
		Height = h;
	}

	// Create the image structure to be filled by png data
    ICPUImage::SCreationParams imgInfo;
    imgInfo.type = ICPUImage::ET_2D;
    imgInfo.extent.width = Width;
    imgInfo.extent.height = Height;
    imgInfo.extent.depth = 1u;
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = 1u;
    imgInfo.samples = ICPUImage::ESCF_1_BIT;
    imgInfo.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
    core::smart_refctd_ptr<ICPUImage> image = nullptr;

	bool lumaAlphaType = false;
	switch (ColorType) {
		case PNG_COLOR_TYPE_RGB_ALPHA:
            imgInfo.format = EF_R8G8B8A8_SRGB;
			break;
		case PNG_COLOR_TYPE_RGB:
            imgInfo.format = EF_R8G8B8_SRGB;
			break;
		case PNG_COLOR_TYPE_GRAY:
            imgInfo.format = EF_R8_SRGB;
			break;
		case PNG_COLOR_TYPE_GRAY_ALPHA:
            imgInfo.format = EF_R8G8B8A8_SRGB;
			lumaAlphaType = true;
			break;
		default:
			{
				os::Printer::log("Unsupported PNG colorspace (only RGB/RGBA/8-bit grayscale), operation aborted.", ELL_ERROR);
                return {};
			}
	}

	// Create array of pointers to rows in image data
    RowPointers = _NBL_NEW_ARRAY(png_bytep, Height);
	if (!RowPointers)
	{
		os::Printer::log("LOAD PNG: Internal PNG create row pointers failure\n", _file->getFileName().c_str(), ELL_ERROR);
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return {};
	}

	auto dimension = asset::getBlockDimensions(imgInfo.format);
	assert(dimension.X == 1 && dimension.Y == 1 && dimension.Z == 1);

    const uint32_t texelFormatBytesize = getTexelOrBlockBytesize(imgInfo.format);

    auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
    ICPUImage::SBufferCopy& region = regions->front();
    //region.imageSubresource.aspectMask = ...; //waits for Vulkan
    region.imageSubresource.mipLevel = 0u;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = 1u;
    region.bufferOffset = 0u;
    region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(Width, texelFormatBytesize);
    region.bufferImageHeight = 0u; //tightly packed
    region.imageOffset = { 0u, 0u, 0u };
    region.imageExtent = imgInfo.extent;

	auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(region.bufferRowLength * region.imageExtent.height * texelFormatBytesize);

	// Fill array of pointers to rows in image data
	const uint32_t pitch = region.bufferRowLength*texelFormatBytesize;
	uint8_t* data = reinterpret_cast<uint8_t*>(texelBuffer->getPointer());
	for (uint32_t i=0; i<Height; ++i)
	{
		RowPointers[i] = (png_bytep)data;
		data += pitch;
	}

	// for proper error handling
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        _NBL_DELETE_ARRAY(RowPointers, Height);
        return {};
	}

	// Read data using the library function that handles all transformations including interlacing
	png_read_image(png_ptr, RowPointers);

	png_read_end(png_ptr, nullptr);
	if (lumaAlphaType)
	{
		assert(imgInfo.format==asset::EF_R8G8B8A8_SRGB);
		for (uint32_t i=0u; i<Height; ++i)
		for (uint32_t j=0u; j<Width;)
		{
			uint32_t in = reinterpret_cast<uint16_t*>(RowPointers[i])[j];
			j++;
			auto& out = reinterpret_cast<uint32_t*>(RowPointers[i])[Width-j];
			out = in|(in << 16u); // LXLA
			out &= 0xffff00ffu;
			out |= (in&0xffu) << 8u;
		}
	}
    _NBL_DELETE_ARRAY(RowPointers, Height);
	png_destroy_read_struct(&png_ptr,&info_ptr, 0); // Clean up memory
#else
    return {};
#endif // _NBL_COMPILE_WITH_LIBPNG_

	image = ICPUImage::create(std::move(imgInfo));

	if (!image)
	{
		os::Printer::log("LOAD PNG: Internal PNG create image struct failure\n", _file->getFileName().c_str(), ELL_ERROR);
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		return {};
	}

	image->setBufferAndRegions(std::move(texelBuffer), regions);

	if (imgInfo.format == asset::EF_R8_SRGB)
		image = asset::IImageAssetHandlerBase::convertR8ToR8G8B8Image(image);

    return SAssetBundle({image});
}


}// end namespace nbl
}//end namespace video

#endif