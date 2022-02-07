// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/core/core.h"
#include "nbl/asset/compile_config.h"
#include "CImageWriterPNG.h"

#ifdef _NBL_COMPILE_WITH_PNG_WRITER_

#include "IWriteFile.h"

#include "os.h"  // for logging

#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

#include "CImageLoaderPNG.h"

#ifdef _NBL_COMPILE_WITH_LIBPNG_
#include "libpng/png.h"
#endif  // _NBL_COMPILE_WITH_LIBPNG_

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
static void png_cpexcept_warning(png_structp png_ptr, png_const_charp msg)
{
    os::Printer::log("PNG warning", msg, ELL_WARNING);
}

// PNG function for file writing
void PNGAPI user_write_data_fcn(png_structp png_ptr, png_bytep data, png_size_t length)
{
    png_size_t check;

    io::IWriteFile* file = (io::IWriteFile*)png_get_io_ptr(png_ptr);
    check = (png_size_t)file->write((const void*)data, (uint32_t)length);

    if(check != length)
        png_error(png_ptr, "Write Error");
}
#endif  // _NBL_COMPILE_WITH_LIBPNG_

CImageWriterPNG::CImageWriterPNG()
{
#ifdef _NBL_DEBUG
    setDebugName("CImageWriterPNG");
#endif
}

bool CImageWriterPNG::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if(!_override)
        getDefaultOverride(_override);

#if defined(_NBL_COMPILE_WITH_LIBPNG_)

    SAssetWriteContext ctx{_params, _file};

    auto imageView = IAsset::castDown<const ICPUImageView>(_params.rootAsset);

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, {imageView, 0u});

    if(!file || !imageView)
        return false;

    // Allocate the png write struct
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
        nullptr, (png_error_ptr)png_cpexcept_error, (png_error_ptr)png_cpexcept_warning);
    if(!png_ptr)
    {
        os::Printer::log("PNGWriter: Internal PNG create write struct failure\n", file->getFileName().c_str(), ELL_ERROR);
        return false;
    }

    // Allocate the png info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if(!info_ptr)
    {
        os::Printer::log("PNGWriter: Internal PNG create info struct failure\n", file->getFileName().c_str(), ELL_ERROR);
        png_destroy_write_struct(&png_ptr, nullptr);
        return false;
    }

    // for proper error handling
    if(setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    core::smart_refctd_ptr<ICPUImage> convertedImage;
    {
        const auto channelCount = asset::getFormatChannelCount(imageView->getCreationParameters().format);
        if(channelCount == 1)
            convertedImage = IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8_SRGB>(imageView);
        else if(channelCount == 2 || channelCount == 3)
            convertedImage = IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8G8B8_SRGB>(imageView);
        else
            convertedImage = IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8G8B8A8_SRGB>(imageView);
    }

    const auto& convertedImageParams = convertedImage->getCreationParameters();
    const auto& convertedRegion = convertedImage->getRegions().begin();
    auto convertedFormat = convertedImageParams.format;

    assert(convertedRegion->bufferRowLength && convertedRegion->bufferImageHeight, "Detected changes in createImageDataForCommonWriting!");
    auto trueExtent = core::vector3du32_SIMD(convertedRegion->bufferRowLength, convertedRegion->bufferImageHeight, convertedRegion->imageExtent.depth);

    png_set_write_fn(png_ptr, file, user_write_data_fcn, nullptr);

    // Set info
    switch(convertedFormat)
    {
        case asset::EF_R8G8B8_SRGB:
            png_set_IHDR(png_ptr, info_ptr,
                trueExtent.X, trueExtent.Y,
                8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            break;
        case asset::EF_R8G8B8A8_SRGB:
            png_set_IHDR(png_ptr, info_ptr,
                trueExtent.X, trueExtent.Y,
                8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            break;
        case asset::EF_R8_SRGB:
            png_set_IHDR(png_ptr, info_ptr,
                trueExtent.X, trueExtent.Y,
                8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            break;
        default: {
            os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
            return false;
        }
    }

    int32_t lineWidth = trueExtent.X;
    switch(convertedFormat)
    {
        case asset::EF_R8_SRGB:
            lineWidth *= 1;
            break;
        case asset::EF_R8G8B8_SRGB:
            lineWidth *= 3;
            break;
        case asset::EF_R8G8B8A8_SRGB:
            lineWidth *= 4;
            break;
        default: {
            os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
            return false;
        }
    }

    uint8_t* data = (uint8_t*)convertedImage->getBuffer()->getPointer();

    constexpr uint32_t maxPNGFileHeight = 16u * 1024u;  // arbitrary limit
    if(trueExtent.Y > maxPNGFileHeight)
    {
        os::Printer::log("PNGWriter: Image dimensions too big!\n", file->getFileName().c_str(), ELL_ERROR);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    // Create array of pointers to rows in image data
    png_bytep RowPointers[maxPNGFileHeight];

    // Fill array of pointers to rows in image data
    for(uint32_t i = 0; i < trueExtent.Y; ++i)
    {
        RowPointers[i] = reinterpret_cast<png_bytep>(data);
        data += lineWidth;
    }

    // for proper error handling
    if(setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    png_set_rows(png_ptr, info_ptr, RowPointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, nullptr);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
#else
    _NBL_DEBUG_BREAK_IF(true);
    return false;
#endif  //defined(_NBL_COMPILE_WITH_LIBPNG_)
}

}  // namespace video
}  // namespace nbl

#endif