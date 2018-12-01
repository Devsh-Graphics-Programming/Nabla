// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageLoaderJPG.h"

#ifdef _IRR_COMPILE_WITH_JPG_LOADER_

#include "IReadFile.h"
#include "CImage.h"
#include "os.h"
#include "ICPUTexture.h"
#include <string>

namespace irr
{
namespace video
{

//! constructor
CImageLoaderJPG::CImageLoaderJPG()
{
	#ifdef _DEBUG
	setDebugName("CImageLoaderJPG");
	#endif
}



//! destructor
CImageLoaderJPG::~CImageLoaderJPG()
{
}


#ifdef _IRR_COMPILE_WITH_LIBJPEG_

    // struct for handling jpeg errors
    struct irr_jpeg_error_mgr
    {
        // public jpeg error fields
        struct jpeg_error_mgr pub;

        // for longjmp, to return to caller on a fatal error
        jmp_buf setjmp_buffer;
    };

void CImageLoaderJPG::init_source (j_decompress_ptr cinfo)
{
	// DO NOTHING
}



boolean CImageLoaderJPG::fill_input_buffer (j_decompress_ptr cinfo)
{
	// DO NOTHING
	return 1;
}



void CImageLoaderJPG::skip_input_data (j_decompress_ptr cinfo, long count)
{
	jpeg_source_mgr * src = cinfo->src;
	if(count > 0)
	{
		src->bytes_in_buffer -= count;
		src->next_input_byte += count;
	}
}



void CImageLoaderJPG::term_source (j_decompress_ptr cinfo)
{
	// DO NOTHING
}


void CImageLoaderJPG::error_exit (j_common_ptr cinfo)
{
	// unfortunately we need to use a goto rather than throwing an exception
	// as gcc crashes under linux crashes when using throw from within
	// extern c code

	// Always display the message
	(*cinfo->err->output_message) (cinfo);

	// cinfo->err really points to a irr_error_mgr struct
	irr_jpeg_error_mgr *myerr = (irr_jpeg_error_mgr*) cinfo->err;

	longjmp(myerr->setjmp_buffer, 1);
}


void CImageLoaderJPG::output_message(j_common_ptr cinfo)
{
	// display the error message.
	char temp1[JMSG_LENGTH_MAX];
	(*cinfo->err->format_message)(cinfo, temp1);
	std::string errMsg("JPEG FATAL ERROR in ");
    errMsg += reinterpret_cast<char*>(cinfo->client_data);
	os::Printer::log(errMsg,temp1, ELL_ERROR);
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
    core::vector<CImageData*> images;

#ifndef _IRR_COMPILE_WITH_LIBJPEG_
	os::Printer::log("Can't load as not compiled with _IRR_COMPILE_WITH_LIBJPEG_:", _file->getFileName(), ELL_DEBUG);
#else
	if (!_file)
		return nullptr;

	const io::path& Filename = _file->getFileName();

	uint8_t **rowPtr=0;
	uint8_t* input = new uint8_t[_file->getSize()];
	_file->read(input, _file->getSize());

	// allocate and initialize JPEG decompression object
	struct jpeg_decompress_struct cinfo;
	struct irr_jpeg_error_mgr jerr;

	//We have to set up the error handler first, in case the initialization
	//step fails.  (Unlikely, but it could happen if you are out of memory.)
	//This routine fills in the contents of struct jerr, and returns jerr's
	//address which we place into the link field in cinfo.

	cinfo.err = jpeg_std_error(&jerr.pub);
	cinfo.err->error_exit = error_exit;
	cinfo.err->output_message = output_message;
    cinfo.client_data = const_cast<char*>(Filename.c_str());

	// compatibility fudge:
	// we need to use setjmp/longjmp for error handling as gcc-linux
	// crashes when throwing within external c code
	if (setjmp(jerr.setjmp_buffer))
	{
		// If we get here, the JPEG code has signaled an error.
		// We need to clean up the JPEG object and return.

		jpeg_destroy_decompress(&cinfo);

		delete [] input;
		// if the row pointer was created, we delete it.
		if (rowPtr)
			delete [] rowPtr;

		// return null pointer
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

	jsrc.init_source = init_source;
	jsrc.fill_input_buffer = fill_input_buffer;
	jsrc.skip_input_data = skip_input_data;
	jsrc.resync_to_restart = jpeg_resync_to_restart;
	jsrc.term_source = term_source;

	// Decodes JPG input from whatever source
	// Does everything AFTER jpeg_create_decompress
	// and BEFORE jpeg_destroy_decompress
	// Caller is responsible for arranging these + setting up cinfo

	// read _file parameters with jpeg_read_header()
	jpeg_read_header(&cinfo, TRUE);

	bool useCMYK=false;
	if (cinfo.jpeg_color_space==JCS_CMYK)
	{
		cinfo.out_color_space=JCS_CMYK;
		cinfo.out_color_components=4;
		useCMYK=true;
	}
	else
	{
		cinfo.out_color_space=JCS_RGB;
		cinfo.out_color_components=3;
	}
	cinfo.output_gamma=2.2;
	cinfo.do_fancy_upsampling=FALSE;

	// Start decompressor
	jpeg_start_decompress(&cinfo);

	// Get image data
	uint16_t rowspan = cinfo.image_width * cinfo.out_color_components;
	uint32_t imageSize[3] = {cinfo.image_width,cinfo.image_height,1};
	uint32_t& width = imageSize[0];
	uint32_t& height = imageSize[1];

	// Allocate memory for buffer
	uint8_t* output = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(rowspan * height,_IRR_SIMD_ALIGNMENT));

	// Here we use the library's state variable cinfo.output_scanline as the
	// loop counter, so that we don't have to keep track ourselves.
	// Create array of row pointers for lib
	rowPtr = new uint8_t* [height];

	for( uint32_t i = 0; i < height; i++ )
		rowPtr[i] = &output[ i * rowspan ];

	uint32_t rowsRead = 0;

	while( cinfo.output_scanline < cinfo.output_height )
		rowsRead += jpeg_read_scanlines( &cinfo, &rowPtr[rowsRead], cinfo.output_height - rowsRead );

	delete [] rowPtr;
	// Finish decompression

	jpeg_finish_decompress(&cinfo);

	// Release JPEG decompression object
	// This is an important step since it will release a good deal of memory.
	jpeg_destroy_decompress(&cinfo);

	uint32_t nullOffset[3] = {0,0,0};
	// convert image
	CImageData* image = 0;
	if (useCMYK)
	{
		image = new CImageData(NULL,nullOffset,imageSize,0,EF_R8G8B8_UNORM);
		const uint32_t size = 3*width*height;
		uint8_t* data = (uint8_t*)image->getData();
		if (data)
		{
			for (uint32_t i=0,j=0; i<size; i+=3, j+=4)
			{
				// Also works without K, but has more contrast with K multiplied in
//				data[i+0] = output[j+2];
//				data[i+1] = output[j+1];
//				data[i+2] = output[j+0];
				data[i+0] = (char)(output[j+2]*(output[j+3]/255.f));
				data[i+1] = (char)(output[j+1]*(output[j+3]/255.f));
				data[i+2] = (char)(output[j+0]*(output[j+3]/255.f));
			}
		}
		_IRR_ALIGNED_FREE(output);
	}
	else
		image = new CImageData(output,nullOffset,imageSize,0,EF_R8G8B8_UNORM,1,true);

	delete [] input;

    images.push_back(image);
#endif

    asset::ICPUTexture* tex = asset::ICPUTexture::create(images);
    for (auto img : images)
        img->drop();
    return tex;
}

} // end namespace video
} // end namespace irr

#endif

