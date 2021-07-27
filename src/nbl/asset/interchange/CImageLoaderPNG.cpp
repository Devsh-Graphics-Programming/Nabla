// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors


#include "os.h"

#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"
#include "nbl/system/ISystem.h"
using namespace nbl;
using namespace system;
using namespace core;

#ifdef _NBL_COMPILE_WITH_PNG_LOADER_

#include "CImageLoaderPNG.h"

#ifdef _NBL_COMPILE_WITH_LIBPNG_
#   include "libpng/png.h"
#endif // _NBL_COMPILE_WITH_LIBPNG_

#include "nbl/system/IFile.h"

namespace nbl
{
namespace asset
{


#ifdef _NBL_COMPILE_WITH_LIBPNG_
// PNG function for error handling

void updateFilePos(png_structp png_pt, size_t new_file_pos)
{
	auto ptr = (CImageLoaderPng::SContext*)png_get_user_chunk_ptr(png_pt);
	ptr->file_pos = new_file_pos;
	png_set_read_user_chunk_fn(png_pt, ptr, nullptr);
}
static void png_cpexcept_error(png_structp png_ptr, png_const_charp msg)
{
	auto ctx = (CImageLoaderPng::SContext*)png_get_user_chunk_ptr(png_ptr);
	ctx->logger.log("PNG fatal error", system::ILogger::ELL_ERROR, msg);
	longjmp(png_jmpbuf(png_ptr), 1);
}

// PNG function for warning handling
static void png_cpexcept_warn(png_structp png_ptr, png_const_charp msg)
{
	auto ctx = (CImageLoaderPng::SContext*)png_get_user_chunk_ptr(png_ptr);
	ctx->logger.log("PNG warning", system::ILogger::ELL_WARNING, msg);
}

// PNG function for file reading
void PNGAPI user_read_data_fcn(png_structp png_pt, png_bytep data, png_size_t length)
{
	png_size_t check;

	auto* userData = (CImageLoaderPng::SContext*)png_get_user_chunk_ptr(png_pt);
	size_t file_pos = userData->file_pos;

	system::IFile* file=(system::IFile*)png_get_io_ptr(png_pt);

	system::ISystem::future_t<uint32_t> future;
	userData->system->readFile(future, file, data, file_pos, length);
	
	file_pos += length;
	updateFilePos(png_pt, file_pos);

	if (check != length)
		png_error(png_pt, "Read Error");
}
#endif // _NBL_COMPILE_WITH_LIBPNG_


//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderPng::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr& logger) const
{
#ifdef _NBL_COMPILE_WITH_LIBPNG_
	if (!_file)
		return false;


	png_byte buffer[8];

	system::ISystem::future_t<uint32_t> future;
	m_system->readFile(future, _file, buffer, 0, 8);
	// Read the first few bytes of the PNG _file
    if (future.get() != 8)
    {
        return false;
    }
	// Check if it really is a PNG _file
	return !png_sig_cmp(buffer, 0, 8);
#else
	return false;
#endif // _NBL_COMPILE_WITH_LIBPNG_
}


// load in the image data
asset::SAssetBundle CImageLoaderPng::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
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

	system::ISystem::future_t<uint32_t> future;
	m_system->readFile(future, _file, buffer, 0, sizeof buffer);
	if(future.get() != 8 )
	{
		_params.logger.log("LOAD PNG: can't read _file\n", system::ILogger::ELL_ERROR, _file->getFileName().string());
        return {};
	}

	// Check if it really is a PNG _file
	if( png_sig_cmp(buffer, 0, 8) )
	{
		_params.logger.log("LOAD PNG: not really a png\n", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
        return {};
	}

	// Allocate the png read struct
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
		nullptr, (png_error_ptr)png_cpexcept_error, (png_error_ptr)png_cpexcept_warn);
	if (!png_ptr)
	{
		_params.logger.log("LOAD PNG: Internal PNG create read struct failure\n", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str(), ELL_ERROR);
        return {};
	}

	// Allocate the png info struct
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		_params.logger.log("LOAD PNG: Internal PNG create info struct failure\n", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
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
	SContext usrData(m_system.get(), _params.logger);
	png_set_read_user_chunk_fn(png_ptr, &usrData, nullptr);

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
			_params.logger.log("Unsupported PNG colorspace (only RGB/RGBA/8-bit grayscale), operation aborted.", system::ILogger::ELL_ERROR);
                return {};
			}
	}

	// Create array of pointers to rows in image data
    RowPointers = _NBL_NEW_ARRAY(png_bytep, Height);
	if (!RowPointers)
	{
		_params.logger.log("LOAD PNG: Internal PNG create row pointers failure\n", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
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
		_params.logger.log("LOAD PNG: Internal PNG create image struct failure\n", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		return {};
	}

	image->setBufferAndRegions(std::move(texelBuffer), regions);

	if (imgInfo.format == asset::EF_R8_SRGB)
		image = asset::IImageAssetHandlerBase::convertR8ToR8G8B8Image(image);

    return SAssetBundle(nullptr,{image});
}


}// end namespace nbl
}//end namespace video

#endif