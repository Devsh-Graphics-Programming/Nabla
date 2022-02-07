// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

#include "nbl/asset/compile_config.h"
#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

#ifdef _NBL_COMPILE_WITH_JPG_WRITER_

#include "CImageWriterJPG.h"

#ifdef _NBL_COMPILE_WITH_LIBJPEG_
#include <stdio.h>  // required for jpeglib.h
extern "C" {
#include "jpeglib.h"
#include "jerror.h"
}

// The writer uses a 4k buffer and flushes to disk each time it's filled
#define OUTPUT_BUF_SIZE 4096

using namespace nbl;
using namespace asset;

struct mem_destination_mgr
{
    struct jpeg_destination_mgr pub; /* public fields */
    system::ISystem* system;
    system::IFile* file; /* target file */
    size_t filePos = 0;
    JOCTET buffer[OUTPUT_BUF_SIZE]; /* image buffer */
};
using mem_dest_ptr = mem_destination_mgr*;

// init
static void jpeg_init_destination(j_compress_ptr cinfo)
{
    mem_dest_ptr dest = (mem_dest_ptr)cinfo->dest;
    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
}

// flush to disk and reset buffer
static boolean jpeg_empty_output_buffer(j_compress_ptr cinfo)
{
    mem_dest_ptr dest = (mem_dest_ptr)cinfo->dest;

    // for now just exit upon file error
    system::future<size_t> future;
    dest->file->write(future, dest->buffer, dest->filePos, OUTPUT_BUF_SIZE);
    if(future.get() != OUTPUT_BUF_SIZE)
    {
        ERREXIT(cinfo, JERR_FILE_WRITE);
    }

    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
    dest->filePos += OUTPUT_BUF_SIZE;
    return TRUE;
}

static void jpeg_term_destination(j_compress_ptr cinfo)
{
    mem_dest_ptr dest = (mem_dest_ptr)cinfo->dest;
    const int32_t datacount = (int32_t)(OUTPUT_BUF_SIZE - dest->pub.free_in_buffer);
    // for now just exit upon file error
    system::future<size_t> future;
    dest->file->write(future, dest->buffer, dest->filePos, datacount);
    if(future.get() != datacount)
    {
        ERREXIT(cinfo, JERR_FILE_WRITE);
    }

    dest->filePos += datacount;
}

// set up buffer data
static void jpeg_file_dest(j_compress_ptr cinfo, system::IFile* file, system::ISystem* sys)
{
    if(cinfo->dest == nullptr)
    { /* first time for this JPEG object? */
        cinfo->dest = (struct jpeg_destination_mgr*)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo,
            JPOOL_PERMANENT,
            sizeof(mem_destination_mgr));
    }

    mem_dest_ptr dest = (mem_dest_ptr)cinfo->dest; /* for casting */

    /* Initialize method pointers */
    dest->pub.init_destination = jpeg_init_destination;
    dest->pub.empty_output_buffer = jpeg_empty_output_buffer;
    dest->pub.term_destination = jpeg_term_destination;

    /* Initialize private member */
    dest->file = file;
    dest->system = sys;
    dest->filePos = 0;
}

/* write_JPEG_memory: store JPEG compressed image into memory.
*/
static bool writeJPEGFile(system::IFile* file, system::ISystem* sys, const asset::ICPUImageView* imageView, uint32_t quality, const system::logger_opt_ptr& logger)
{
    core::smart_refctd_ptr<ICPUImage> convertedImage;
    {
        const auto channelCount = asset::getFormatChannelCount(imageView->getCreationParameters().format);
        if(channelCount == 1)
            convertedImage = asset::IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8_SRGB>(imageView, logger);
        else
            convertedImage = asset::IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8G8B8_SRGB>(imageView, logger);
    }

    const auto& convertedImageParams = convertedImage->getCreationParameters();
    auto convertedFormat = convertedImageParams.format;

    bool grayscale = (convertedFormat == asset::EF_R8_SRGB);

    core::vector3d<uint32_t> dim = {convertedImageParams.extent.width, convertedImageParams.extent.height, convertedImageParams.extent.depth};
    const auto rowByteSize = asset::getTexelOrBlockBytesize(convertedFormat) * convertedImage->getRegions().begin()->bufferRowLength;

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_compress(&cinfo);
    jpeg_file_dest(&cinfo, file, sys);
    cinfo.image_width = dim.X;
    cinfo.image_height = dim.Y;
    cinfo.input_components = grayscale ? 1 : 3;
    cinfo.in_color_space = grayscale ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_set_defaults(&cinfo);

    if(0 == quality)
        quality = 85;

    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const auto JPG_BYTE_PITCH = rowByteSize;
    auto destBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(JPG_BYTE_PITCH);
    auto dest = reinterpret_cast<uint8_t*>(destBuffer->getPointer());

    if(dest)
    {
        const uint32_t pitch = JPG_BYTE_PITCH;
        JSAMPROW row_pointer[1]; /* pointer to JSAMPLE row[s] */
        row_pointer[0] = dest;

        uint8_t* src = (uint8_t*)convertedImage->getBuffer()->getPointer();

        /* Switch up, write from bottom -> top because the texture is flipped from OpenGL side */
        uint32_t eof = cinfo.image_height * cinfo.image_width * cinfo.input_components;

        while(cinfo.next_scanline < cinfo.image_height)
        {
            memcpy(dest, src, pitch);

            src += pitch;
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }

        /* Step 6: Finish compression */
        jpeg_finish_compress(&cinfo);
    }

    /* Step 7: Destroy */
    jpeg_destroy_compress(&cinfo);

    return (dest != 0);
}
#endif  // _NBL_COMPILE_WITH_LIBJPEG_

CImageWriterJPG::CImageWriterJPG(core::smart_refctd_ptr<system::ISystem>&& sys)
    : m_system(std::move(sys))
{
#ifdef _NBL_DEBUG
    setDebugName("CImageWriterJPG");
#endif
}

bool CImageWriterJPG::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
#if !defined(_NBL_COMPILE_WITH_LIBJPEG_)
    return false;
#else
    SAssetWriteContext ctx{_params, _file};

    auto imageView = IAsset::castDown<const ICPUImageView>(_params.rootAsset);

    system::IFile* file = _override->getOutputFile(_file, ctx, {imageView, 0u});
    const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(ctx, imageView, 0u);
    const float comprLvl = _override->getAssetCompressionLevel(ctx, imageView, 0u);

    return writeJPEGFile(file, m_system.get(), imageView, (!!(flags & asset::EWF_COMPRESSED)) * static_cast<uint32_t>((1.f - comprLvl) * 100.f), _params.logger);  // if quality==0, then it defaults to 75

#endif  //!defined(_NBL_COMPILE_WITH_LIBJPEG_ )
}

#undef OUTPUT_BUF_SIZE
#endif