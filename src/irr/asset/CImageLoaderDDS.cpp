// Copyright (C) 2002-2012 Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

/*
	Based on Code from Copyright (c) 2003 Randy Reddig
	Based on code from Nvidia's DDS example:
	http://www.nvidia.com/object/dxtc_decompression_code.html

	mainly c to cpp
*/

#include "CImageLoaderDDS.h"

#ifdef _IRR_COMPILE_WITH_DDS_LOADER_

#include <utility>
#include "IReadFile.h"
#include "os.h"
#include "CColorConverter.h"
#include "irr/asset/CImageData.h"
#include "irr/asset/ICPUTexture.h"


namespace irr
{

namespace asset
{

static int32_t DDSLittleLong( int32_t src ) { return src; }

static bool DDSGetInfo(CImageLoaderDDS::ddsBuffer *dds, int32_t *width, int32_t *height, int32_t *depth, CImageLoaderDDS::eDDSPixelFormat *pf )
{
	if(	dds == NULL || pf == NULL )
		return false;

    if (strncmp(dds->magic, "DDS ", 4u) != 0)	
		return false;
	
	if(DDSLittleLong( dds->size ) != 124 )
		return false;

	/* extract width and height */
	if ( width != NULL )
		*width = DDSLittleLong( dds->width );
	if ( height != NULL )
		*height = DDSLittleLong( dds->height );
    if ( depth != NULL && (dds->flags & 0x800000u) )//DDSD_DEPTH)
        *depth = DDSLittleLong( dds->depth );
    else
        *depth = 1;

	/* get pixel format */
	const uint32_t fourCC = dds->pixelFormat.fourCC;

	if( fourCC == 0 )
	{
	    bool hasAlpha = false;
	    bool hasRGB = false;
	    bool hasLuma = false;
	    uint32_t bitDepth = dds->pixelFormat.privateFormatBitCount;

	    if (dds->pixelFormat.flags & 0x3)
			hasAlpha = true;
	    if (dds->pixelFormat.flags & 0x40)
			hasRGB = true;
	    if (dds->pixelFormat.flags & 0x20000)
			hasLuma = true;

        if (bitDepth==32&&(dds->pixelFormat.rBitMask&0x00ff0000)&&hasRGB&&hasAlpha)
            *pf = CImageLoaderDDS::DDS_PF_ARGB8888;
        else if (bitDepth==32&&(dds->pixelFormat.rBitMask&0xff)&&hasRGB&&hasAlpha)
            *pf = CImageLoaderDDS::DDS_PF_ABGR8888;
        else if (bitDepth==24&&(dds->pixelFormat.rBitMask&0x00ff0000)&&hasRGB)
            *pf = CImageLoaderDDS::DDS_PF_RGB888;
        else if (bitDepth==16&&(dds->pixelFormat.rBitMask&0x7c00)&&hasRGB&&hasAlpha)
            *pf = CImageLoaderDDS::DDS_PF_ARGB1555;
        else if (bitDepth==16&&(dds->pixelFormat.rBitMask&0xf800)&&hasRGB)
            *pf = CImageLoaderDDS::DDS_PF_RGB565;
        else if (bitDepth==16&&(dds->pixelFormat.rBitMask&0xff)&&hasLuma&&hasAlpha)
            *pf = CImageLoaderDDS::DDS_PF_LA88;
        else if (bitDepth==8&&(dds->pixelFormat.rBitMask&0xff)&&hasLuma)
            *pf = CImageLoaderDDS::DDS_PF_L8;
        else if (bitDepth==8&&(dds->pixelFormat.rBitMask==0)&&hasAlpha)
            *pf = CImageLoaderDDS::DDS_PF_A8;
        else
            return false;
	}
	else if( fourCC == *((uint32_t*) "DXT1") )
		*pf = CImageLoaderDDS::DDS_PF_DXT1;
	else if( fourCC == *((uint32_t*) "DXT2") )
		*pf = CImageLoaderDDS::DDS_PF_DXT2;
	else if( fourCC == *((uint32_t*) "DXT3") )
		*pf = CImageLoaderDDS::DDS_PF_DXT3;
	else if( fourCC == *((uint32_t*) "DXT4") )
		*pf = CImageLoaderDDS::DDS_PF_DXT4;
	else if( fourCC == *((uint32_t*) "DXT5") )
		*pf = CImageLoaderDDS::DDS_PF_DXT5;
	else
		return false;
	
	return true;
}

//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderDDS::isALoadableFileFormat(io::IReadFile* _file) const
{
	if (!_file)
		return false;

    const size_t prevPos = _file->getPos();

	ddsBuffer header;
	_file->read(&header, sizeof(header));
    _file->seek(prevPos);

	int32_t width, height, depth;
	eDDSPixelFormat pixelFormat;

	return DDSGetInfo(&header, &width, &height, &depth, &pixelFormat);
}

//! creates a surface from the file
asset::IAsset* CImageLoaderDDS::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	core::vector<asset::CImageData*> images;

    CImageLoaderDDS::eDDSPixelFormat pixelFormat;
    int32_t width, height, depth, mipmapCnt;

	ddsBuffer header;
	_file->read(&header, sizeof(header)-4);

	if (DDSGetInfo(&header, &width, &height, &depth, &pixelFormat))
	{
	    if (header.flags & 0x20000)//DDSD_MIPMAPCOUNT)
            mipmapCnt = header.mipMapCount;
	    else
            mipmapCnt = 1;


        for (int32_t i=0; i<mipmapCnt; i++)
        {
            uint32_t zeroDummy[3] = {0,0,0};
            uint32_t mipSize[3] = {0,height,depth};
            uint32_t& tmpWidth = mipSize[0];
            switch( pixelFormat )
            {
                case DDS_PF_DXT1:
                case DDS_PF_DXT2:
                case DDS_PF_DXT3:
                case DDS_PF_DXT4:
                case DDS_PF_DXT5:
                    tmpWidth = width;
                    break;
                default:
                    tmpWidth = header.pitch;
                    break;
            }
            uint32_t& tmpHeight = mipSize[1];
            uint32_t& tmpDepth = mipSize[2];
            tmpWidth += (uint32_t(1)<<i)-1;
            tmpHeight += (uint32_t(1)<<i)-1;
            if (false)
                tmpDepth += (uint32_t(1)<<i)-1; //! CHANGE AGAIN FOR 2D ARRAY AND CUBEMAP TEXTURES
            tmpWidth /= uint32_t(1)<<i;
            tmpHeight /= uint32_t(1)<<i;
            if (false)
                tmpDepth /= uint32_t(1)<<i; //! CHANGE AGAIN FOR 2D ARRAY AND CUBEMAP TEXTURES

            /* decompress */
            asset::E_FORMAT colorFormat = asset::EF_UNKNOWN;
            switch( pixelFormat )
            {
                case DDS_PF_ARGB8888:
                case DDS_PF_ABGR8888:
                    /* fixme: support other [a]rgb formats */
                    {
                        colorFormat = pixelFormat==DDS_PF_ABGR8888 ? asset::EF_R8G8B8A8_UNORM:asset::EF_B8G8R8A8_UNORM;
                        asset::CImageData* data = new asset::CImageData(NULL,zeroDummy,mipSize,i,colorFormat,4);
                        _file->read(data->getData(),data->getImageDataSizeInBytes());
                        images.push_back(data);
                    }
                    break;
                case DDS_PF_RGB888:
                    /* fixme: support other [a]rgb formats */
                    {
                        colorFormat = asset::EF_R8G8B8_UNORM;
                        asset::CImageData* data = new asset::CImageData(NULL,zeroDummy,mipSize,i,colorFormat,1);
                        _file->read(data->getData(),data->getImageDataSizeInBytes());
                        images.push_back(data);

                        uint8_t* dataToManipulate = reinterpret_cast<uint8_t*>(data->getData());
                        for (uint32_t j=0; j<data->getImageDataSizeInBytes(); j+=3)
                        {
                            uint8_t byte1 = dataToManipulate[j+0];
                            uint8_t byte2 = dataToManipulate[j+1];
                            uint8_t byte3 = dataToManipulate[j+2];

                            dataToManipulate[j+0] = byte3;
                            dataToManipulate[j+1] = byte2;
                            dataToManipulate[j+2] = byte1;
                        }
                    }
                    break;
                case DDS_PF_ARGB1555:
                    /* fixme: support other [a]rgb formats */
                    {
                        colorFormat = asset::EF_A1R5G5B5_UNORM_PACK16;
                        asset::CImageData* data = new asset::CImageData(NULL,zeroDummy,mipSize,i,colorFormat,2);
                        _file->read(data->getData(),data->getImageDataSizeInBytes());
                        images.push_back(data);
                    }
                    break;
                case DDS_PF_RGB565:
                    break;
                case DDS_PF_LA88:
                    /* fixme: support other [a]rgb formats */
                    {
                        colorFormat = asset::EF_R8G8_UNORM;
                        asset::CImageData* data = new asset::CImageData(NULL,zeroDummy,mipSize,i,colorFormat,2);
                        _file->read(data->getData(),data->getImageDataSizeInBytes());
                        images.push_back(data);
                    }
                    break;
                case DDS_PF_L8:
                case DDS_PF_A8:
                    /* fixme: support other [a]rgb formats */
                    {
                        colorFormat = asset::EF_R8_UNORM;
                        asset::CImageData* data = new asset::CImageData(NULL,zeroDummy,mipSize,i,colorFormat,1);
                        _file->read(data->getData(),data->getImageDataSizeInBytes());
                        images.push_back(data);
                    }
                    break;

                case DDS_PF_DXT1:
                case DDS_PF_DXT2:
                case DDS_PF_DXT3:
                case DDS_PF_DXT4:
                case DDS_PF_DXT5:
                    {
                        if (pixelFormat==CImageLoaderDDS::DDS_PF_DXT2||pixelFormat==CImageLoaderDDS::DDS_PF_DXT3)
                            colorFormat = asset::EF_BC2_UNORM_BLOCK;
                        else if (pixelFormat==CImageLoaderDDS::DDS_PF_DXT4||pixelFormat==CImageLoaderDDS::DDS_PF_DXT5)
                            colorFormat = asset::EF_BC3_UNORM_BLOCK;
                        else if (pixelFormat==CImageLoaderDDS::DDS_PF_DXT1_ALPHA)
                            colorFormat = asset::EF_BC1_RGBA_UNORM_BLOCK;
                        else if (pixelFormat==CImageLoaderDDS::DDS_PF_DXT1)
                            colorFormat = asset::EF_BC1_RGB_UNORM_BLOCK;

                        asset::CImageData* data = new asset::CImageData(NULL,zeroDummy,mipSize,i,colorFormat,1);
                        _file->read(data->getData(),data->getImageDataSizeInBytes());
                        images.push_back(data);
                    }
                    break;

                default:
					{
						os::Printer::log("Unsupported DDS texture format", ELL_ERROR);
						return nullptr;
					}
					break;
            }
        }
	}

	asset::ICPUTexture* tex = asset::ICPUTexture::create(images);
    for (auto& img : images)
        img->drop();
    return tex;
}


} // end namespace video
} // end namespace irr

#endif

