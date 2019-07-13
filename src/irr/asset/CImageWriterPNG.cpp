// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageWriterPNG.h"

#ifdef _IRR_COMPILE_WITH_PNG_WRITER_

#include "CImageLoaderPNG.h"
#include "IWriteFile.h"
#include "os.h" // for logging
#include "irr/asset/ICPUTexture.h"

#ifdef _IRR_COMPILE_WITH_LIBPNG_
#   include "libpng/png.h"
#endif // _IRR_COMPILE_WITH_LIBPNG_

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
static void png_cpexcept_warning(png_structp png_ptr, png_const_charp msg)
{
	os::Printer::log("PNG warning", msg, ELL_WARNING);
}

// PNG function for file writing
void PNGAPI user_write_data_fcn(png_structp png_ptr, png_bytep data, png_size_t length)
{
	png_size_t check;

	io::IWriteFile* file=(io::IWriteFile*)png_get_io_ptr(png_ptr);
	check=(png_size_t) file->write((const void*)data,(uint32_t)length);

	if (check != length)
		png_error(png_ptr, "Write Error");
}
#endif // _IRR_COMPILE_WITH_LIBPNG_

CImageWriterPNG::CImageWriterPNG()
{
#ifdef _IRR_DEBUG
	setDebugName("CImageWriterPNG");
#endif
}

bool CImageWriterPNG::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);
#ifdef _IRR_COMPILE_WITH_LIBPNG_
    SAssetWriteContext ctx{_params, _file};

    const asset::CImageData* image =
#   ifndef _IRR_DEBUG
        static_cast<const asset::CImageData*>(_params.rootAsset);
#   else
        dynamic_cast<const asset::CImageData*>(_params.rootAsset);
#   endif
    assert(image);

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, {image, 0u});

	if (!file || !image)
		return false;

	// Allocate the png write struct
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
		nullptr, (png_error_ptr)png_cpexcept_error, (png_error_ptr)png_cpexcept_warning);
	if (!png_ptr)
	{
		os::Printer::log("PNGWriter: Internal PNG create write struct failure\n", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}

	// Allocate the png info struct
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		os::Printer::log("PNGWriter: Internal PNG create info struct failure\n", file->getFileName().c_str(), ELL_ERROR);
		png_destroy_write_struct(&png_ptr, nullptr);
		return false;
	}

	// for proper error handling
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return false;
	}
	
	auto format = image->getColorFormat();
	
	png_set_write_fn(png_ptr, file, user_write_data_fcn, nullptr);
	
	// Set info
	switch (format)
	{
		case asset::EF_R8G8B8_SRGB:
			png_set_IHDR(png_ptr, info_ptr,
				image->getSize().X, image->getSize().Y,
				8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
				PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
			break;
		case asset::EF_R8G8B8A8_SRGB:
			png_set_IHDR(png_ptr, info_ptr,
				image->getSize().X, image->getSize().Y,
				8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
				PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
		break;
		case asset::EF_R8_SRGB:
		case asset::EF_R8_UNORM:
			png_set_IHDR(png_ptr, info_ptr,
				image->getSize().X, image->getSize().Y,
				8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
				PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
		break;
		default:
			{
				os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
				return false;
			}
	}

	int32_t lineWidth = image->getSize().X;
	switch (format)
	{
		case asset::EF_R8_SRGB:
		case asset::EF_R8_UNORM:
			lineWidth *= 1;
			break;
		case asset::EF_R8G8B8_SRGB:
			lineWidth *= 3;
			break;
		case asset::EF_R8G8B8A8_SRGB:
			lineWidth *= 4;
			break;
		default:
			{
				os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
				return false;
			}
	}
	
	uint8_t* data = (uint8_t*)image->getData();
	core::vector3d<uint32_t> dim = image->getSize();
	
	// Write in reverse order because the texture is flipped to match OpenGL coords
	data += (dim.X * dim.Y * (image->getBitsPerPixel() / 8)) - lineWidth;
	
	// Create array of pointers to rows in image data
	uint8_t** RowPointers = new png_bytep[dim.Y];
	if (!RowPointers)
	{
		os::Printer::log("PNGWriter: Internal PNG create row pointers failure\n", file->getFileName().c_str(), ELL_ERROR);
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return false;
	}

	// Fill array of pointers to rows in image data
	for (uint32_t i = 0; i < dim.Y; ++i)
	{
		switch (format) {
			case asset::EF_R8_UNORM:
				{
					const void *src_container[4] = {data, nullptr, nullptr, nullptr};
					video::convertColor<EF_R8_UNORM, EF_R8_SRGB>(src_container, data, dim.X, dim);
				}
				break;
			case asset::EF_R8_SRGB:
				_IRR_FALLTHROUGH;
			case asset::EF_R8G8B8_SRGB:
				_IRR_FALLTHROUGH;
			case asset::EF_R8G8B8A8_SRGB:
				RowPointers[i] = data;
				break;
			default:
			{
				os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
				if (RowPointers) delete [] RowPointers;
				return false;
			}
		}
		
		data -= lineWidth;
	}
	
	// for proper error handling
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_write_struct(&png_ptr, &info_ptr);
		delete [] RowPointers;
		return false;
	}

	png_set_rows(png_ptr, info_ptr, RowPointers);
	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, nullptr);

	delete [] RowPointers;
	png_destroy_write_struct(&png_ptr, &info_ptr);
	return true;
#else
	return false;
#endif
}

} // namespace video
} // namespace irr

#endif

