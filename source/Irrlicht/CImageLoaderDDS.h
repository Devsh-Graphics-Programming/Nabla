// Copyright (C) 2002-2012 Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_LOADER_DDS_H_INCLUDED__
#define __C_IMAGE_LOADER_DDS_H_INCLUDED__

#include "IrrCompileConfig.h"

#if defined(_IRR_COMPILE_WITH_DDS_LOADER_)

#include "IImageLoader.h"

namespace irr
{
namespace video
{

/* dependencies */
/* dds definition */
enum eDDSPixelFormat
{
	DDS_PF_ARGB8888,
	DDS_PF_ABGR8888,
	DDS_PF_RGB888,
	DDS_PF_ARGB1555,
	DDS_PF_RGB565,
	DDS_PF_LA88,
	DDS_PF_L8,
	DDS_PF_A8,
	DDS_PF_DXT1,
	DDS_PF_DXT1_ALPHA,
	DDS_PF_DXT2,
	DDS_PF_DXT3,
	DDS_PF_DXT4,
	DDS_PF_DXT5,
	DDS_PF_UNKNOWN
};

/* 16bpp stuff */
#define DDS_LOW_5		0x001F;
#define DDS_MID_6		0x07E0;
#define DDS_HIGH_5		0xF800;
#define DDS_MID_555		0x03E0;
#define DDS_HI_555		0x7C00;


// byte-align structures
#include "irrpack.h"

/* structures */
struct ddsColorKey
{
	u32		colorSpaceLowValue;
	u32		colorSpaceHighValue;
} PACK_STRUCT;

struct ddsCaps
{
	u32		caps1;
	u32		caps2;
	u32		caps3;
	u32		caps4;
} PACK_STRUCT;

struct ddsMultiSampleCaps
{
	u16		flipMSTypes;
	u16		bltMSTypes;
} PACK_STRUCT;


struct ddsPixelFormat
{
	u32		size;
	u32		flags;
	u32		fourCC;
	union
	{
		u32	rgbBitCount;
		u32	yuvBitCount;
		u32	zBufferBitDepth;
		u32	alphaBitDepth;
		u32	luminanceBitCount;
		u32	bumpBitCount;
		u32	privateFormatBitCount;
	};
	union
	{
		u32	rBitMask;
		u32	yBitMask;
		u32	stencilBitDepth;
		u32	luminanceBitMask;
		u32	bumpDuBitMask;
		u32	operations;
	};
	union
	{
		u32	gBitMask;
		u32	uBitMask;
		u32	zBitMask;
		u32	bumpDvBitMask;
		ddsMultiSampleCaps	multiSampleCaps;
	};
	union
	{
		u32	bBitMask;
		u32	vBitMask;
		u32	stencilBitMask;
		u32	bumpLuminanceBitMask;
	};
	union
	{
		u32	rgbAlphaBitMask;
		u32	yuvAlphaBitMask;
		u32	luminanceAlphaBitMask;
		u32	rgbZBitMask;
		u32	yuvZBitMask;
	};
} PACK_STRUCT;


struct ddsBuffer
{
	/* magic: 'dds ' */
	c8				magic[ 4 ];

	/* directdraw surface */
	u32		size;//4
	u32		flags;//8
	u32		height;//12
	u32		width;//16
	union
	{
		s32				pitch;
		u32	linearSize;
	};//20
	u32		backBufferCount;//24
	union
	{
		u32	mipMapCount;
		u32	refreshRate;
		u32	srcVBHandle;
	};//28
	u32		alphaBitDepth;//32
	u32		reserved;//36
	/// I think its time someone reminded the irrlicht folk that void* is non portable 32-64bit sizewise in structs
	u32		surface_UNUSED;//40
	union
	{
		ddsColorKey	ckDestOverlay;
		u32	emptyFaceColor;
	};//44
	ddsColorKey		ckDestBlt;//52
	ddsColorKey		ckSrcOverlay;//60
	ddsColorKey		ckSrcBlt;//68
	union
	{
		ddsPixelFormat	pixelFormat;
		u32	fvf;
	};//76
	ddsCaps			caps;//
	u32		textureStage;

	/* data (Varying size) */
	u8		data[ 4 ];
} PACK_STRUCT;

/*
struct ddsColorBlock
{
	u16		colors[ 2 ];
	u8		row[ 4 ];
} PACK_STRUCT;


struct ddsAlphaBlockExplicit
{
	u16		row[ 4 ];
} PACK_STRUCT;


struct ddsAlphaBlock3BitLinear
{
	u8		alpha0;
	u8		alpha1;
	u8		stuff[ 6 ];
} PACK_STRUCT;


struct ddsColor
{
	u8		r, g, b, a;
} PACK_STRUCT;*/

// Default alignment
#include "irrunpack.h"


/* endian tomfoolery */
typedef union
{
	f32	f;
	c8	c[ 4 ];
}
floatSwapUnion;


	s32   DDSLittleLong( s32 src );
	s16 DDSLittleShort( s16 src );
	f32 DDSLittleFloat( f32 src );

	s32 DDSBigLong( s32 src );

	s16 DDSBigShort( s16 src );

	f32 DDSBigFloat( f32 src );


/*!
	Surface Loader for DDS images
*/
class CImageLoaderDDS : public IImageLoader
{
public:

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".tga")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! returns true if the file maybe is able to be loaded by this class
	virtual bool isALoadableFileFormat(io::IReadFile* file) const;

	//! creates a surface from the file
	virtual IImage* loadImage(io::IReadFile* file) const;

	//! proper load, returns allocated image data (mip maps inclusif)
	u8* loadDataBuffer(io::IReadFile* file, eDDSPixelFormat *pixelFormat, s32 *width, s32 *height, s32 *mipmapCnt) const;
};


} // end namespace video
} // end namespace irr

#endif // compiled with DDS loader
#endif

