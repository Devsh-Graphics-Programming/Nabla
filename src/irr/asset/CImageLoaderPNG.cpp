// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderPNG.h"

#ifdef _IRR_COMPILE_WITH_PNG_LOADER_

#ifdef _IRR_COMPILE_WITH_LIBPNG_
#   include "libpng/png.h"
#endif // _IRR_COMPILE_WITH_LIBPNG_

#include "irr/asset/ICPUTexture.h"
#include "irr/asset/CImageData.h"
#include "CReadFile.h"
#include "os.h"

namespace irr
{
namespace asset
{

#ifdef _IRR_COMPILE_WITH_LIBPNG_
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
#endif // _IRR_COMPILE_WITH_LIBPNG_


//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderPng::isALoadableFileFormat(io::IReadFile* _file) const
{
#ifdef _IRR_COMPILE_WITH_LIBPNG_
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
#endif // _IRR_COMPILE_WITH_LIBPNG_
}


// load in the image data
asset::IAsset* CImageLoaderPng::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    core::vector<asset::CImageData*> images;
#ifdef _IRR_COMPILE_WITH_LIBPNG_
	if (!_file)
		return nullptr;
	
	asset::CImageData* image = 0;
	//Used to point to image rows
	uint8_t** RowPointers = 0;

	png_byte buffer[8];
	// Read the first few bytes of the PNG _file
	if( _file->read(buffer, 8) != 8 )
	{
		os::Printer::log("LOAD PNG: can't read _file\n", _file->getFileName().c_str(), ELL_ERROR);
		return nullptr;
	}

	// Check if it really is a PNG _file
	if( png_sig_cmp(buffer, 0, 8) )
	{
		os::Printer::log("LOAD PNG: not really a png\n", _file->getFileName().c_str(), ELL_ERROR);
		return nullptr;
	}

	// Allocate the png read struct
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
		nullptr, (png_error_ptr)png_cpexcept_error, (png_error_ptr)png_cpexcept_warn);
	if (!png_ptr)
	{
		os::Printer::log("LOAD PNG: Internal PNG create read struct failure\n", _file->getFileName().c_str(), ELL_ERROR);
		return nullptr;
	}

	// Allocate the png info struct
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		os::Printer::log("LOAD PNG: Internal PNG create info struct failure\n", _file->getFileName().c_str(), ELL_ERROR);
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		return nullptr;
	}

	// for proper error handling
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
		if (RowPointers)
			delete [] RowPointers;
		return nullptr;
	}

	// changed by zola so we don't need to have public FILE pointers
	png_set_read_fn(png_ptr, _file, user_read_data_fcn);

	png_set_sig_bytes(png_ptr, 8); // Tell png that we read the signature

	png_read_info(png_ptr, info_ptr); // Read the info section of the png _file

	uint32_t imageSize[3] = {1,1,1};
	uint32_t& Width = imageSize[0];
	uint32_t& Height = imageSize[1];
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
	uint32_t nullOffset[3] = {0,0,0};
	
	switch (ColorType) {
		case PNG_COLOR_TYPE_RGB_ALPHA:
			image = new asset::CImageData(nullptr, nullOffset, imageSize, 0, asset::EF_R8G8B8A8_SRGB);
			break;
		case PNG_COLOR_TYPE_RGB:
			image = new asset::CImageData(nullptr, nullOffset, imageSize, 0, asset::EF_R8G8B8_SRGB);
			break;
		case PNG_COLOR_TYPE_GRAY:
			image = new asset::CImageData(nullptr, nullOffset, imageSize, 0, asset::EF_R8_SRGB);
			break;
		default:
			{
				os::Printer::log("Unsupported PNG colorspace (only RGB/RGBA/8-bit grayscale), operation aborted.", ELL_ERROR);
				return nullptr;
			}
	}
	
	if (!image)
	{
		os::Printer::log("LOAD PNG: Internal PNG create image struct failure\n", _file->getFileName().c_str(), ELL_ERROR);
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		return nullptr;
	}

	// Create array of pointers to rows in image data
	RowPointers = new png_bytep[Height];
	if (!RowPointers)
	{
		os::Printer::log("LOAD PNG: Internal PNG create row pointers failure\n", _file->getFileName().c_str(), ELL_ERROR);
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		image->drop();
		return nullptr;
	}

	// Fill array of pointers to rows in image data
	const uint32_t pitch = image->getPitchIncludingAlignment();
	uint8_t* data = reinterpret_cast<uint8_t*>(image->getData()) + (image->getSize().X * image->getSize().Y * (image->getBitsPerPixel() / 8)) - pitch;
	for (uint32_t i=0; i<Height; ++i)
	{
		RowPointers[i] = (png_bytep)data;
		data -= pitch;
	}

	// for proper error handling
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
		delete [] RowPointers;
		image->drop();
		return nullptr;
	}

	// Read data using the library function that handles all transformations including interlacing
	png_read_image(png_ptr, RowPointers);

	png_read_end(png_ptr, nullptr);
	delete [] RowPointers;
	png_destroy_read_struct(&png_ptr,&info_ptr, 0); // Clean up memory

	images.push_back(image);
#endif // _IRR_COMPILE_WITH_LIBPNG_

    asset::ICPUTexture* tex = asset::ICPUTexture::create(images);
    for (auto& img : images)
        img->drop();
    return tex;
}


}// end namespace irr
}//end namespace video

#endif

