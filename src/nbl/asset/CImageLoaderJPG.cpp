// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/asset/compile_config.h"

#include "CImageLoaderJPG.h"

#ifdef _NBL_COMPILE_WITH_JPG_LOADER_

#include "IReadFile.h"
#include "os.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/ICPUImageView.h"

#include "nbl/asset/IImageAssetHandlerBase.h"

#include <string>

#include <stdio.h> // required for jpeglib.h
#ifdef _NBL_COMPILE_WITH_LIBJPEG_
extern "C" {
#include "libjpeg/jpeglib.h" // use irrlicht jpeglib
#include <setjmp.h>
}
#endif // _NBL_COMPILE_WITH_LIBJPEG_

namespace nbl
{
namespace asset
{

//! constructor
CImageLoaderJPG::CImageLoaderJPG()
{
	#ifdef _NBL_DEBUG
	setDebugName("CImageLoaderJPG");
	#endif
}



//! destructor
CImageLoaderJPG::~CImageLoaderJPG()
{
}


#ifdef _NBL_COMPILE_WITH_LIBJPEG_
namespace jpeg
{
	// struct for handling jpeg errors
	struct irr_jpeg_error_mgr
	{
		// public jpeg error fields
		struct jpeg_error_mgr pub;

		// for longjmp, to return to caller on a fatal error
		jmp_buf setjmp_buffer;
	};

	/* 	Receives control for a fatal error.  Information sufficient to
	generate the error message has been stored in cinfo->err; call
	output_message to display it.  Control must NOT return to the caller;
	generally this routine will exit() or longjmp() somewhere.
	Typically you would override this routine to get rid of the exit()
	default behavior.  Note that if you continue processing, you should
	clean up the JPEG object with jpeg_abort() or jpeg_destroy().
	*/
	void error_exit(j_common_ptr cinfo)
	{
		// unfortunately we need to use a goto rather than throwing an exception
		// as gcc crashes under linux crashes when using throw from within
		// extern c code

		// Always display the message
		(*cinfo->err->output_message) (cinfo);

		// cinfo->err really points to a irr_error_mgr struct
		irr_jpeg_error_mgr* myerr = (irr_jpeg_error_mgr*)cinfo->err;

		longjmp(myerr->setjmp_buffer, 1);
	}

	/* output error messages via Irrlicht logger. */
	void output_message(j_common_ptr cinfo)
	{
		// display the error message.
		char temp1[JMSG_LENGTH_MAX];
		(*cinfo->err->format_message)(cinfo, temp1);
		std::string errMsg("JPEG FATAL ERROR in ");
		errMsg += reinterpret_cast<char*>(cinfo->client_data);
		os::Printer::log(errMsg, temp1, ELL_ERROR);
	}

	/*	Initialize source.  This is called by jpeg_read_header() before any
	data is actually read.  Unlike init_destination(), it may leave
	bytes_in_buffer set to 0 (in which case a fill_input_buffer() call
	will occur immediately). */
	void init_source(j_decompress_ptr cinfo)
	{
		// DO NOTHING
	}

	/*	This is called whenever bytes_in_buffer has reached zero and more
	data is wanted.  In typical applications, it should read fresh data
	into the buffer (ignoring the current state of next_input_byte and
	bytes_in_buffer), reset the pointer & count to the start of the
	buffer, and return TRUE indicating that the buffer has been reloaded.
	It is not necessary to fill the buffer entirely, only to obtain at
	least one more byte.  bytes_in_buffer MUST be set to a positive value
	if TRUE is returned.  A FALSE return should only be used when I/O
	suspension is desired (this mode is discussed in the next section). */
	boolean fill_input_buffer(j_decompress_ptr cinfo)
	{
		// DO NOTHING
		return TRUE;
	}

	/* Skip num_bytes worth of data.  The buffer pointer and count should
	be advanced over num_bytes input bytes, refilling the buffer as
	needed.  This is used to skip over a potentially large amount of
	uninteresting data (such as an APPn marker).  In some applications
	it may be possible to optimize away the reading of the skipped data,
	but it's not clear that being smart is worth much trouble; large
	skips are uncommon.  bytes_in_buffer may be zero on return.
	A zero or negative skip count should be treated as a no-op. */
	void skip_input_data(j_decompress_ptr cinfo, long num_bytes)
	{
		jpeg_source_mgr* src = cinfo->src;
		if (num_bytes > 0)
		{
			src->bytes_in_buffer -= num_bytes;
			src->next_input_byte += num_bytes;
		}
	}

	/* Terminate source --- called by jpeg_finish_decompress() after all
	data has been read.  Often a no-op. */
	void term_source(j_decompress_ptr cinfo)
	{
		// DO NOTHING
	}

}
#endif // _NBL_COMPILE_WITH_LIBJPEG_

//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderJPG::isALoadableFileFormat(io::IReadFile* _file) const
{
#ifndef _NBL_COMPILE_WITH_LIBJPEG_
	return false;
#else
	if (!_file)
		return false;

    const size_t prevPos = _file->getPos();

	int32_t jfif = 0;
	_file->seek(6);
	_file->read(&jfif, sizeof(int32_t));
    _file->seek(prevPos);
	return (jfif == 0x4a464946 || jfif == 0x4649464a || jfif == 0x66697845u || jfif == 0x70747468u); // maybe 0x4a464946 can go
#endif
}

//! creates a surface from the file
asset::SAssetBundle CImageLoaderJPG::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
#ifndef _NBL_COMPILE_WITH_LIBJPEG_
	os::Printer::log("Can't load as not compiled with _NBL_COMPILE_WITH_LIBJPEG_:", _file->getFileName().c_str(), ELL_DEBUG);
	return nullptr
#else
	if (!_file || _file->getSize()>0xffffffffull)
        return {};

	const io::path& Filename = _file->getFileName();

	uint8_t* input = new uint8_t[_file->getSize()];
	_file->read(input, static_cast<uint32_t>(_file->getSize()));

	// allocate and initialize JPEG decompression object
	struct jpeg_decompress_struct cinfo;
	struct jpeg::irr_jpeg_error_mgr jerr;

	//We have to set up the error handler first, in case the initialization
	//step fails.  (Unlikely, but it could happen if you are out of memory.)
	//This routine fills in the contents of struct jerr, and returns jerr's
	//address which we place into the link field in cinfo.

	cinfo.err = jpeg_std_error(&jerr.pub);
	cinfo.err->error_exit = jpeg::error_exit;
	cinfo.err->output_message = jpeg::output_message;
    cinfo.client_data = const_cast<char*>(Filename.c_str());

	auto exitRoutine = [&] {
		jpeg_destroy_decompress(&cinfo);
		delete[] input;
	};
	auto exiter = core::makeRAIIExiter(exitRoutine);
	// compatibility fudge:
	// we need to use setjmp/longjmp for error handling as gcc-linux
	// crashes when throwing within external c code
	if (setjmp(jerr.setjmp_buffer))
	{
		os::Printer::log("Can't load libjpeg threw an error:", _file->getFileName().c_str(), ELL_ERROR);
		// RAIIExiter takes care of cleanup
        return {};
	}

	// Now we can initialize the JPEG decompression object.
	jpeg_create_decompress(&cinfo);

	// specify data source
	jpeg_source_mgr jsrc;

	// Set up data pointer
	jsrc.bytes_in_buffer = _file->getSize();
	jsrc.next_input_byte = (JOCTET*)input;
	cinfo.src = &jsrc;

	jsrc.init_source = jpeg::init_source;
	jsrc.fill_input_buffer = jpeg::fill_input_buffer;
	jsrc.skip_input_data = jpeg::skip_input_data;
	jsrc.resync_to_restart = jpeg_resync_to_restart;
	jsrc.term_source = jpeg::term_source;

	// Decodes JPG input from whatever source
	// Does everything AFTER jpeg_create_decompress
	// and BEFORE jpeg_destroy_decompress
	// Caller is responsible for arranging these + setting up cinfo

	// read _file parameters with jpeg_read_header()
	jpeg_read_header(&cinfo, TRUE);

    uint32_t imageSize[3] = { cinfo.image_width,cinfo.image_height,1 };
    const uint32_t& width = imageSize[0];
    const uint32_t& height = imageSize[1];

    ICPUImage::SCreationParams imgInfo;
    imgInfo.type = ICPUImage::ET_2D;
    imgInfo.extent.width = width;
    imgInfo.extent.height = height;
    imgInfo.extent.depth = 1u;
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = 1u;
    imgInfo.samples = ICPUImage::ESCF_1_BIT;
    imgInfo.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);

	switch (cinfo.jpeg_color_space)
	{
		case JCS_GRAYSCALE:
			cinfo.out_color_components = 1;
			cinfo.output_gamma = 1.0; // output_gamma is a dead variable in libjpegturbo and jpeglib
            imgInfo.format = EF_R8_SRGB;
			break;
		case JCS_RGB:
			cinfo.out_color_components = 3;
			cinfo.output_gamma = 2.2333333; // output_gamma is a dead variable in libjpegturbo and jpeglib
            imgInfo.format = EF_R8G8B8_SRGB;
			break;
		case JCS_YCbCr:
			cinfo.out_color_components = 3;
			cinfo.output_gamma = 2.2333333; // output_gamma is a dead variable in libjpegturbo and jpeglib
            imgInfo.format = EF_R8G8B8_SRGB;
			// it seems that libjpeg does Y'UV to R'G'B'conversion automagically
			// however be prepared that the colors might be a bit "off"
			// https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
			break;
		case JCS_CMYK:
			os::Printer::log("CMYK color space is unsupported:", _file->getFileName().c_str(), ELL_ERROR);
			return {};
			break;
		case JCS_YCCK: // this I have no resources on
			os::Printer::log("YCCK color space is unsupported:", _file->getFileName().c_str(), ELL_ERROR);
			return {};
			break;
		case JCS_BG_RGB: // interesting
			os::Printer::log("Loading JPEG Big Gamut RGB is not implemented yet:", _file->getFileName().c_str(), ELL_ERROR);
			return {};
			break;
		case JCS_BG_YCC: // interesting
			os::Printer::log("Loading JPEG Big Gamut YCbCr is not implemented yet:", _file->getFileName().c_str(), ELL_ERROR);
			return {};
			break;
		default:
			os::Printer::log("Can't load as color space is unknown:", _file->getFileName().c_str(), ELL_ERROR);
			return {};
			break;
	}
	cinfo.do_fancy_upsampling = TRUE;
	
	// Start decompressor
	jpeg_start_decompress(&cinfo);

	auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
	ICPUImage::SBufferCopy& region = regions->front();
	//region.imageSubresource.aspectMask = ...; //waits for Vulkan
	region.imageSubresource.mipLevel = 0u;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.bufferOffset = 0u;
	region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(width, getTexelOrBlockBytesize(imgInfo.format));
	region.bufferImageHeight = 0u; //tightly packed
	region.imageOffset = { 0u, 0u, 0u };
	region.imageExtent = imgInfo.extent;
	
	// Get image data
	uint32_t rowspan = region.bufferRowLength * cinfo.out_color_components;

	// Allocate memory for buffer
	auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(rowspan*height);

	// Here we use the library's state variable cinfo.output_scanline as the
	// loop counter, so that we don't have to keep track ourselves.
	// Create array of row pointers for lib
	constexpr uint32_t MaxJPEGResolution = 65535u;
	uint8_t* rowPtr[MaxJPEGResolution];
	for (uint32_t i = 0; i < height; ++i)
		rowPtr[i] = &reinterpret_cast<uint8_t*>(buffer->getPointer())[i*rowspan];

	// Read rows from bottom order to match OpenGL coords
	uint32_t rowsRead = 0;
	while (cinfo.output_scanline < cinfo.output_height)
		rowsRead += jpeg_read_scanlines(&cinfo, &rowPtr[rowsRead], 1);
	
	// Finish decompression
	jpeg_finish_decompress(&cinfo);

	core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imgInfo));
	image->setBufferAndRegions(std::move(buffer), regions);

	if (image->getCreationParameters().format == asset::EF_R8_SRGB)
		image = asset::IImageAssetHandlerBase::convertR8ToR8G8B8Image(image);
	
    return SAssetBundle({image});

#endif
}

} // end namespace video
} // end namespace nbl

#endif