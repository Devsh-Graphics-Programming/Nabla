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

#include "IReadFile.h"
#include "os.h"
#include "CColorConverter.h"
#include "CImage.h"
#include "irrString.h"


namespace irr
{

namespace video
{


#ifndef __BIG_ENDIAN__
#ifdef _SGI_SOURCE
#define	__BIG_ENDIAN__
#endif
#endif


#ifdef __BIG_ENDIAN__

	s32   DDSBigLong( s32 src ) { return src; }
	s16 DDSBigShort( s16 src ) { return src; }
	f32 DDSBigFloat( f32 src ) { return src; }

	s32 DDSLittleLong( s32 src )
	{
		return ((src & 0xFF000000) >> 24) |
			((src & 0x00FF0000) >> 8) |
			((src & 0x0000FF00) << 8) |
			((src & 0x000000FF) << 24);
	}

	s16 DDSLittleShort( s16 src )
	{
		return ((src & 0xFF00) >> 8) |
			((src & 0x00FF) << 8);
	}

	f32 DDSLittleFloat( f32 src )
	{
		floatSwapUnion in,out;
		in.f = src;
		out.c[ 0 ] = in.c[ 3 ];
		out.c[ 1 ] = in.c[ 2 ];
		out.c[ 2 ] = in.c[ 1 ];
		out.c[ 3 ] = in.c[ 0 ];
		return out.f;
	}

#else /*__BIG_ENDIAN__*/

	s32   DDSLittleLong( s32 src ) { return src; }
	s16 DDSLittleShort( s16 src ) { return src; }
	f32 DDSLittleFloat( f32 src ) { return src; }

	s32 DDSBigLong( s32 src )
	{
		return ((src & 0xFF000000) >> 24) |
			((src & 0x00FF0000) >> 8) |
			((src & 0x0000FF00) << 8) |
			((src & 0x000000FF) << 24);
	}

	s16 DDSBigShort( s16 src )
	{
		return ((src & 0xFF00) >> 8) |
			((src & 0x00FF) << 8);
	}

	f32 DDSBigFloat( f32 src )
	{
		floatSwapUnion in,out;
		in.f = src;
		out.c[ 0 ] = in.c[ 3 ];
		out.c[ 1 ] = in.c[ 2 ];
		out.c[ 2 ] = in.c[ 1 ];
		out.c[ 3 ] = in.c[ 0 ];
		return out.f;
	}

#endif /*__BIG_ENDIAN__*/

namespace
{

/*!
	DDSDecodePixelFormat()
	determines which pixel format the dds texture is in
*/
void DDSDecodePixelFormat( ddsBuffer *dds, eDDSPixelFormat *pf )
{
	/* dummy check */
	if(	dds == NULL || pf == NULL )
		return;

	/* extract fourCC */
	const u32 fourCC = dds->pixelFormat.fourCC;

	/* test it */
	if( fourCC == 0 )
	{
	    bool hasAlpha = false;
	    bool hasRGB = false;
	    bool hasLuma = false;
	    u32 bitDepth = dds->pixelFormat.privateFormatBitCount;

	    if (dds->pixelFormat.flags&0x3)
	    {
            hasAlpha = true;
	    }
	    if (dds->pixelFormat.flags&0x40)
	    {
            hasRGB = true;
	    }
	    if (dds->pixelFormat.flags&0x20000)
	    {
            hasLuma = true;
	    }

        if (bitDepth==32&&(dds->pixelFormat.rBitMask&0x00ff0000)&&hasRGB&&hasAlpha)
            *pf = DDS_PF_ARGB8888;
        else if (bitDepth==32&&(dds->pixelFormat.rBitMask&0xff)&&hasRGB&&hasAlpha)
            *pf = DDS_PF_ABGR8888;
        else if (bitDepth==24&&(dds->pixelFormat.rBitMask&0x00ff0000)&&hasRGB)
            *pf = DDS_PF_RGB888;
        else if (bitDepth==16&&(dds->pixelFormat.rBitMask&0x7c00)&&hasRGB&&hasAlpha)
            *pf = DDS_PF_ARGB1555;
        else if (bitDepth==16&&(dds->pixelFormat.rBitMask&0xf800)&&hasRGB)
            *pf = DDS_PF_RGB565;
        else if (bitDepth==16&&(dds->pixelFormat.rBitMask&0xff)&&hasLuma&&hasAlpha)
            *pf = DDS_PF_LA88;
        else if (bitDepth==8&&(dds->pixelFormat.rBitMask&0xff)&&hasLuma)
            *pf = DDS_PF_L8;
        else if (bitDepth==8&&(dds->pixelFormat.rBitMask==0)&&hasAlpha)
            *pf = DDS_PF_A8;
        else
            *pf = DDS_PF_UNKNOWN;
	}
	else if( fourCC == *((u32*) "DXT1") )
	{ // sodan was here
//	    if (dds->pixelFormat.privateFormatBitCount==24)
            *pf = DDS_PF_DXT1;
/*        else if (dds->pixelFormat.privateFormatBitCount==32)
            *pf = DDS_PF_DXT1_ALPHA;
        else
            printf("IRRLICHT BUUUUUUUUGGGGG!!!!!!!!!!\n SHOOT SOMEONE!\n");*/
	}
	else if( fourCC == *((u32*) "DXT2") )
		*pf = DDS_PF_DXT2;
	else if( fourCC == *((u32*) "DXT3") )
		*pf = DDS_PF_DXT3;
	else if( fourCC == *((u32*) "DXT4") )
		*pf = DDS_PF_DXT4;
	else if( fourCC == *((u32*) "DXT5") )
		*pf = DDS_PF_DXT5;
	else
		*pf = DDS_PF_UNKNOWN;
}


/*!
DDSGetInfo()
extracts relevant info from a dds texture, returns 0 on success
*/
s32 DDSGetInfo( ddsBuffer *dds, s32 *width, s32 *height, eDDSPixelFormat *pf )
{
	/* dummy test */
	if( dds == NULL )
		return -1;

	/* test dds header */
	if( *((s32*) dds->magic) != *((s32*) "DDS ") )
		return -1;
	if( DDSLittleLong( dds->size ) != 124 )
		return -1;

	/* extract width and height */
	if( width != NULL )
		*width = DDSLittleLong( dds->width );
	if( height != NULL )
		*height = DDSLittleLong( dds->height );

	/* get pixel format */
	DDSDecodePixelFormat( dds, pf );

	/* return ok */
	return 0;
}


} // end anonymous namespace


//! returns true if the file maybe is able to be loaded by this class
//! based on the file extension (e.g. ".tga")
bool CImageLoaderDDS::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension ( filename, "dds" );
}


//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderDDS::isALoadableFileFormat(io::IReadFile* file) const
{
	if (!file)
		return false;

	ddsBuffer header;
	file->read(&header, sizeof(header));

	s32 width, height;
	eDDSPixelFormat pixelFormat;

	return (0 == DDSGetInfo( &header, &width, &height, &pixelFormat));
}


//! proper load, returns allocated image data (mip maps inclusif)
u8* CImageLoaderDDS::loadDataBuffer(io::IReadFile* file, eDDSPixelFormat *pixelFormat, s32 *width, s32 *height, s32 *mipmapCnt) const
{
	u8 *memFile = new u8 [ file->getSize() ];
	file->read ( memFile, file->getSize() );

	ddsBuffer *header = (ddsBuffer*) memFile;
	u8* data = 0;

	if ( 0 == DDSGetInfo( header, width, height, pixelFormat) )
	{
	    size_t dataSize = 0;

	    if (header->flags & 0x20000)
            *mipmapCnt = header->mipMapCount;
	    else
            *mipmapCnt = 1;

        u32 tmpWidth = *width;
        u32 tmpHeight = *height;

        s32 r = 1;
        /* decompress */
        switch( *pixelFormat )
        {
        case DDS_PF_ARGB8888:
        case DDS_PF_ABGR8888:
            /* fixme: support other [a]rgb formats */
            tmpWidth = header->pitch;
            for (s32 i=0; i<*mipmapCnt; i++)
            {
                dataSize += tmpWidth*tmpHeight;
                tmpWidth /= 2;
                tmpHeight /= 2;
            }
            break;
        case DDS_PF_RGB888:
            /* fixme: support other [a]rgb formats */
            tmpWidth = header->pitch;
            for (s32 i=0; i<*mipmapCnt; i++)
            {
                size_t bytes = tmpWidth*tmpHeight;
                for (u32 j=0; j<bytes; j+=3)
                {
                    u8 byte1 = header->data[dataSize+j];
                    u8 byte2 = header->data[dataSize+j+1];
                    u8 byte3 = header->data[dataSize+j+2];

                    header->data[dataSize+j] = byte3;
                    header->data[dataSize+j+1] = byte2;
                    header->data[dataSize+j+2] = byte1;
                }
                dataSize += bytes;
                tmpWidth /= 2;
                tmpHeight /= 2;
            }
            break;
        case DDS_PF_ARGB1555:
        case DDS_PF_RGB565:
        case DDS_PF_LA88:
        case DDS_PF_L8:
        case DDS_PF_A8:
            /* fixme: support other [a]rgb formats */
            tmpWidth = header->pitch;
            for (s32 i=0; i<*mipmapCnt; i++)
            {
                dataSize += tmpWidth*tmpHeight;
                tmpWidth /= 2;
                tmpHeight /= 2;
            }
            break;

        case DDS_PF_DXT1:
            for (s32 i=0; i<*mipmapCnt; i++)
            {
                dataSize += core::max_(u32(1), u32((tmpWidth)+3) / 4) * core::max_(u32(1), u32(tmpHeight+3) / 4)*8;
                tmpWidth /= 2;
                tmpHeight /= 2;
            }
            break;

        case DDS_PF_DXT2:
        case DDS_PF_DXT3:
        case DDS_PF_DXT4:
        case DDS_PF_DXT5:
            for (s32 i=0; i<*mipmapCnt; i++)
            {
                dataSize += core::max_(u32(1), u32((tmpWidth)+3) / 4) * core::max_(u32(1), u32(tmpHeight+3) / 4)*16;
                tmpWidth /= 2;
                tmpHeight /= 2;
            }
            break;

        default:
        case DDS_PF_UNKNOWN:
            r = -1;
            break;
        }


		if ( r != -1 && dataSize>0)
		{
            data = new u8[dataSize];
		    memcpy(data,header->data,dataSize);
		}
	}

	if (!data)
	{
	    *pixelFormat = DDS_PF_UNKNOWN;
	    *width = -1;
	    *height = -1;
	    *mipmapCnt = -1;
	}

	delete [] memFile;

	return data;
}

//! creates a surface from the file
IImage* CImageLoaderDDS::loadImage(io::IReadFile* file) const
{
	u8 *memFile = new u8 [ file->getSize() ];
	file->read ( memFile, file->getSize() );

	ddsBuffer *header = (ddsBuffer*) memFile;
	IImage* image = 0;
	s32 width, height;
	eDDSPixelFormat pixelFormat;

	if ( 0 == DDSGetInfo( header, &width, &height, &pixelFormat) )
	{/*
		image = new CImage(ECF_A8R8G8B8, core::dimension2d<u32>(width, height));

		if ( DDSDecompress( header, (u8*) image->lock() ) == -1)
		{
			image->unlock();
			image->drop();
			image = 0;
		}*/
	}

	delete [] memFile;
	if ( image )
		image->unlock();

	return image;
}


//! creates a loader which is able to load dds images
IImageLoader* createImageLoaderDDS()
{
	return new CImageLoaderDDS();
}


} // end namespace video
} // end namespace irr

#endif

