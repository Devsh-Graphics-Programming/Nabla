// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderJPG.h"

#ifdef _IRR_COMPILE_WITH_JPG_LOADER_

#include "IReadFile.h"
#include "CImage.h"
#include "os.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/ICPUTexture.h"
#include <string>

#include <stdio.h> // required for jpeglib.h
#ifdef _IRR_COMPILE_WITH_LIBJPEG_
extern "C" {
#include "libjpeg/jpeglib.h" // use irrlicht jpeglib
#include <setjmp.h>
}
#endif // _IRR_COMPILE_WITH_LIBJPEG_

namespace irr
{
namespace asset
{

//! constructor
CImageLoaderJPG::CImageLoaderJPG()
{
	#ifdef _IRR_DEBUG
	setDebugName("CImageLoaderJPG");
	#endif
}



//! destructor
CImageLoaderJPG::~CImageLoaderJPG()
{
}


#ifdef _IRR_COMPILE_WITH_LIBJPEG_
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
#endif // _IRR_COMPILE_WITH_LIBJPEG_

//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderJPG::isALoadableFileFormat(io::IReadFile* _file) const
{
#ifndef _IRR_COMPILE_WITH_LIBJPEG_
	return false;
#else
	if (!_file)
		return false;

    const size_t prevPos = _file->getPos();

	int32_t jfif = 0;
	_file->seek(6);
	_file->read(&jfif, sizeof(int32_t));
    _file->seek(prevPos);
	return (jfif == 0x4a464946 || jfif == 0x4649464a);
#endif
}

//! creates a surface from the file
asset::IAsset* CImageLoaderJPG::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
#ifndef _IRR_COMPILE_WITH_LIBJPEG_
	os::Printer::log("Can't load as not compiled with _IRR_COMPILE_WITH_LIBJPEG_:", _file->getFileName().c_str(), ELL_DEBUG);
	return nullptr
#else
	if (!_file || _file->getSize()>0xffffffffull)
		return nullptr;

	const io::path& Filename = _file->getFileName();

	uint8_t** rowPtr = nullptr;
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

	asset::ICPUBuffer* output = nullptr;

	auto exitRoutine = [&] {
		if (output)
			output->drop();
		if (rowPtr)
			delete[] rowPtr;
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
		return nullptr;
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

	switch (cinfo.jpeg_color_space)
	{
		case JCS_GRAYSCALE:
			cinfo.out_color_components = 1;
			cinfo.output_gamma = 1.0; // output_gamma is a dead variable in libjpegturbo and jpeglib
			break;
		case JCS_RGB:
			cinfo.out_color_components = 3;
			cinfo.output_gamma = 2.2333333f; // output_gamma is a dead variable in libjpegturbo and jpeglib
			break;
		case JCS_YCbCr:
			cinfo.out_color_components = 3;
			cinfo.output_gamma = 2.2333333f; // output_gamma is a dead variable in libjpegturbo and jpeglib
			// it seems that libjpeg does Y'UV to R'G'B'conversion automagically
			// however be prepared that the colors might be a bit "off"
			// https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
			break;
		case JCS_CMYK:
			os::Printer::log("CMYK color space is unsupported:", _file->getFileName().c_str(), ELL_ERROR);
			return nullptr;
			break;
		case JCS_YCCK: // this I have no resources on
			os::Printer::log("YCCK color space is unsupported:", _file->getFileName().c_str(), ELL_ERROR);
			return nullptr;
			break;
		case JCS_BG_RGB: // interesting
			os::Printer::log("Loading JPEG Big Gamut RGB is not implemented yet:", _file->getFileName().c_str(), ELL_ERROR);
			return nullptr;
			break;
		case JCS_BG_YCC: // interesting
			os::Printer::log("Loading JPEG Big Gamut YCbCr is not implemented yet:", _file->getFileName().c_str(), ELL_ERROR);
			return nullptr;
			break;
		default:
			os::Printer::log("Can't load as color space is unknown:", _file->getFileName().c_str(), ELL_ERROR);
			return nullptr;
			break;
	}
	cinfo.do_fancy_upsampling = TRUE;
	
	// Start decompressor
	jpeg_start_decompress(&cinfo);
	
	// Get image data
	uint32_t rowspan = cinfo.image_width * cinfo.out_color_components;
	uint32_t imageSize[3] = {cinfo.image_width,cinfo.image_height,1};
	uint32_t& width = imageSize[0];
	uint32_t& height = imageSize[1];

	// Allocate memory for buffer
	output = new asset::ICPUBuffer(rowspan*height);

	// Here we use the library's state variable cinfo.output_scanline as the
	// loop counter, so that we don't have to keep track ourselves.
	// Create array of row pointers for lib
	rowPtr = new uint8_t* [height];
	for (uint32_t i = 0; i < height; ++i)
		rowPtr[i] = &reinterpret_cast<uint8_t*>(output->getPointer())[i*rowspan];

	// Read rows from bottom order to match OpenGL coords
	uint32_t rowsRead = 0;
	while (cinfo.output_scanline < cinfo.output_height)
		rowsRead += jpeg_read_scanlines(&cinfo, &rowPtr[cinfo.output_height - 1 - rowsRead], 1);
	
	// Finish decompression
	jpeg_finish_decompress(&cinfo);
	
	asset::CImageData* image = nullptr;
	uint32_t nullOffset[3] = {0,0,0};
	switch (cinfo.jpeg_color_space)
	{
		case JCS_GRAYSCALE:
			// https://github.com/buildaworldnet/IrrlichtBAW/pull/273#issuecomment-491492010
			image = new asset::CImageData(output->getPointer(),nullOffset,imageSize,0u,asset::EF_R8_SRGB,1);
			break;
		case JCS_RGB:
			image = new asset::CImageData(output->getPointer(),nullOffset,imageSize,0u,asset::EF_R8G8B8_SRGB,1);
			break;
		case JCS_YCbCr:
			// libjpeg does implicit conversion to R'G'B'
			image = new asset::CImageData(output->getPointer(),nullOffset,imageSize,0u,asset::EF_R8G8B8_SRGB,1);
			break;
		default: // should never get here
			os::Printer::log("Unsupported color space, operation aborted.", ELL_ERROR);
			return nullptr;
			break;
	}

	asset::ICPUTexture* tex = asset::ICPUTexture::create({image});
	image->drop();
	return tex;
#endif
}

} // end namespace video
} // end namespace irr

#endif

