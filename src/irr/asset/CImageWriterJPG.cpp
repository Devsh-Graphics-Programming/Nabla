// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageWriterJPG.h"

#ifdef _IRR_COMPILE_WITH_JPG_WRITER_

#include "IWriteFile.h"
#include "irr/asset/format/convertColor.h"
#include "irr/asset/ICPUImageView.h"
#include "os.h"

#ifdef _IRR_COMPILE_WITH_LIBJPEG_
#include <stdio.h> // required for jpeglib.h
extern "C"
{
	#include "libjpeg/jpeglib.h"
	#include "libjpeg/jerror.h"
}

// The writer uses a 4k buffer and flushes to disk each time it's filled
#define OUTPUT_BUF_SIZE 4096

using namespace irr;
using namespace asset;

namespace 
{
typedef struct
{
	struct jpeg_destination_mgr pub;/* public fields */

	io::IWriteFile* file;		/* target file */
	JOCTET buffer[OUTPUT_BUF_SIZE];	/* image buffer */
} mem_destination_mgr;


typedef mem_destination_mgr * mem_dest_ptr;
}

// init
static void jpeg_init_destination(j_compress_ptr cinfo)
{
	mem_dest_ptr dest = (mem_dest_ptr) cinfo->dest;
	dest->pub.next_output_byte = dest->buffer;
	dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
}


// flush to disk and reset buffer
static boolean jpeg_empty_output_buffer(j_compress_ptr cinfo)
{
	mem_dest_ptr dest = (mem_dest_ptr) cinfo->dest;

	// for now just exit upon file error
	if (dest->file->write(dest->buffer, OUTPUT_BUF_SIZE) != OUTPUT_BUF_SIZE)
		ERREXIT (cinfo, JERR_FILE_WRITE);

	dest->pub.next_output_byte = dest->buffer;
	dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;

	return TRUE;
}


static void jpeg_term_destination(j_compress_ptr cinfo)
{
	mem_dest_ptr dest = (mem_dest_ptr) cinfo->dest;
	const int32_t datacount = (int32_t)(OUTPUT_BUF_SIZE - dest->pub.free_in_buffer);
	// for now just exit upon file error
	if (dest->file->write(dest->buffer, datacount) != datacount)
		ERREXIT (cinfo, JERR_FILE_WRITE);
}


// set up buffer data
static void jpeg_file_dest(j_compress_ptr cinfo, io::IWriteFile* file)
{
	if (cinfo->dest == nullptr)
	{ /* first time for this JPEG object? */
		cinfo->dest = (struct jpeg_destination_mgr *)
			(*cinfo->mem->alloc_small) ((j_common_ptr) cinfo,
						JPOOL_PERMANENT,
						sizeof(mem_destination_mgr));
	}

	mem_dest_ptr dest = (mem_dest_ptr) cinfo->dest;  /* for casting */

	/* Initialize method pointers */
	dest->pub.init_destination = jpeg_init_destination;
	dest->pub.empty_output_buffer = jpeg_empty_output_buffer;
	dest->pub.term_destination = jpeg_term_destination;

	/* Initialize private member */
	dest->file = file;
}

template<asset::E_FORMAT outFormat>
core::smart_refctd_ptr<asset::ICPUImage> getJPGConvertedOutput(const asset::ICPUImage* image)
{
	static_assert(!asset::isBlockCompressionFormat<outFormat>(), "Only non BC formats supported!");

	using CONVERSION_FILTER = asset::CConvertFormatImageFilter<asset::EF_UNKNOWN, outFormat>;

	core::smart_refctd_ptr<asset::ICPUImage> newConvertedImage;
	{
		auto referenceImageParams = image->getCreationParameters();
		auto referenceBuffer = image->getBuffer();
		auto referenceRegions = image->getRegions();
		auto referenceRegion = referenceRegions.begin();
		const auto newTexelOrBlockByteSize = asset::getTexelOrBlockBytesize(outFormat);

		asset::TexelBlockInfo referenceBlockInfo(referenceImageParams.format);
		core::vector3du32_SIMD referenceTrueExtent = referenceBlockInfo.convertTexelsToBlocks(referenceRegion->getTexelStrides());

		auto newImageParams = referenceImageParams;
		auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(referenceTrueExtent.X * referenceTrueExtent.Y * referenceTrueExtent.Z * newTexelOrBlockByteSize);
		auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1);
		newRegions->front() = *referenceRegion;

		newImageParams.format = outFormat;
		newConvertedImage = ICPUImage::create(std::move(newImageParams));
		newConvertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

		CONVERSION_FILTER convertFilter;
		CONVERSION_FILTER::state_type state;

		auto attachedRegion = newConvertedImage->getRegions().begin();

		state.inImage = image;
		state.outImage = newConvertedImage.get();
		state.inOffset = { 0, 0, 0 };
		state.inBaseLayer = 0;
		state.outOffset = { 0, 0, 0 };
		state.outBaseLayer = 0;
		state.extent = attachedRegion->getExtent();
		state.layerCount = attachedRegion->imageSubresource.layerCount;
		state.inMipLevel = 0;
		state.outMipLevel = 0;

		if (!convertFilter.execute(&state))
			os::Printer::log("Something went wrong while converting!", ELL_WARNING);

		return newConvertedImage;
	}
}

/* write_JPEG_memory: store JPEG compressed image into memory.
*/
static bool writeJPEGFile(io::IWriteFile* file, const asset::ICPUImage* image, uint32_t quality)
{
	core::smart_refctd_ptr<ICPUImage> convertedImage;
	{
		const auto channelCount = asset::getFormatChannelCount(image->getCreationParameters().format);
		if (channelCount == 1)
			convertedImage = getJPGConvertedOutput<asset::EF_R8_SRGB>(image);
		else
			convertedImage = getJPGConvertedOutput<asset::EF_R8G8B8_SRGB>(image);
	}

	const auto& convertedImageParams = convertedImage->getCreationParameters();
	auto convertedFormat = convertedImageParams.format;

	bool grayscale = (convertedFormat == asset::EF_R8_SRGB);
	
	core::vector3d<uint32_t> dim = { convertedImageParams.extent.width, convertedImageParams.extent.height, convertedImageParams.extent.depth };
	const auto rowByteSize = asset::getTexelOrBlockBytesize(convertedFormat) * convertedImage->getRegions().begin()->bufferRowLength;

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_compress(&cinfo);
	jpeg_file_dest(&cinfo, file);
	cinfo.image_width = dim.X;
	cinfo.image_height = dim.Y;
	cinfo.input_components = grayscale ? 1 : 3;
	cinfo.in_color_space = grayscale ? JCS_GRAYSCALE : JCS_RGB;

	jpeg_set_defaults(&cinfo);

	if ( 0 == quality )
		quality = 85;

	jpeg_set_quality(&cinfo, quality, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	const auto JPG_BYTE_PITCH = rowByteSize;
	uint8_t* dest = new uint8_t[JPG_BYTE_PITCH];

	if (dest)
	{
		const uint32_t pitch = JPG_BYTE_PITCH;
		JSAMPROW row_pointer[1];      /* pointer to JSAMPLE row[s] */
		row_pointer[0] = dest;
		
		uint8_t* src = (uint8_t*)convertedImage->getBuffer()->getPointer();
		
		/* Switch up, write from bottom -> top because the texture is flipped from OpenGL side */
		uint32_t eof = cinfo.image_height * cinfo.image_width * cinfo.input_components;
		
		while (cinfo.next_scanline < cinfo.image_height)
		{
			switch (convertedFormat)
			{
				case asset::EF_R8_SRGB: _IRR_FALLTHROUGH;
				case asset::EF_R8G8B8_SRGB:
					memcpy(dest, src, pitch);
					break;
				default:
				{
					os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
					delete [] dest;
					return false;
				}
			}
			
			src += pitch;
			jpeg_write_scanlines(&cinfo, row_pointer, 1);
		}

		delete [] dest;

		/* Step 6: Finish compression */
		jpeg_finish_compress(&cinfo);
	}

	/* Step 7: Destroy */
	jpeg_destroy_compress(&cinfo);

	return (dest != 0);
}
#endif // _IRR_COMPILE_WITH_LIBJPEG_

CImageWriterJPG::CImageWriterJPG()
{
#ifdef _IRR_DEBUG
	setDebugName("CImageWriterJPG");
#endif
}

bool CImageWriterJPG::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
#if !defined(_IRR_COMPILE_WITH_LIBJPEG_ )
	return false;
#else
	SAssetWriteContext ctx{ _params, _file };

	const asset::ICPUImageView* imageView = IAsset::castDown<ICPUImageView>(_params.rootAsset);
	const auto smartImage = IImageAssetHandlerBase::getTopImageDataForCommonWriting(imageView);
	const auto image = smartImage.get();

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, {image, 0u});
    const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(ctx, image, 0u);
    const float comprLvl = _override->getAssetCompressionLevel(ctx, image, 0u);

	return writeJPEGFile(file, image, (!!(flags & asset::EWF_COMPRESSED)) * static_cast<uint32_t>((1.f-comprLvl)*100.f)); // if quality==0, then it defaults to 75

#endif//!defined(_IRR_COMPILE_WITH_LIBJPEG_ )
}

#undef OUTPUT_BUF_SIZE
#endif