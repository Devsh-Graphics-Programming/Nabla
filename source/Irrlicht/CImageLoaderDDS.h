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
	uint32_t		colorSpaceLowValue;
	uint32_t		colorSpaceHighValue;
} PACK_STRUCT;

struct ddsCaps
{
	uint32_t		caps1;
	uint32_t		caps2;
	uint32_t		caps3;
	uint32_t		caps4;
} PACK_STRUCT;

struct ddsMultiSampleCaps
{
	uint16_t		flipMSTypes;
	uint16_t		bltMSTypes;
} PACK_STRUCT;


struct ddsPixelFormat
{
	uint32_t		size;
	uint32_t		flags;
	uint32_t		fourCC;
	union
	{
		uint32_t	rgbBitCount;
		uint32_t	yuvBitCount;
		uint32_t	zBufferBitDepth;
		uint32_t	alphaBitDepth;
		uint32_t	luminanceBitCount;
		uint32_t	bumpBitCount;
		uint32_t	privateFormatBitCount;
	} PACK_STRUCT;
	union
	{
		uint32_t	rBitMask;
		uint32_t	yBitMask;
		uint32_t	stencilBitDepth;
		uint32_t	luminanceBitMask;
		uint32_t	bumpDuBitMask;
		uint32_t	operations;
	} PACK_STRUCT;
	union
	{
		uint32_t	gBitMask;
		uint32_t	uBitMask;
		uint32_t	zBitMask;
		uint32_t	bumpDvBitMask;
		ddsMultiSampleCaps	multiSampleCaps;
	} PACK_STRUCT;
	union
	{
		uint32_t	bBitMask;
		uint32_t	vBitMask;
		uint32_t	stencilBitMask;
		uint32_t	bumpLuminanceBitMask;
	} PACK_STRUCT;
	union
	{
		uint32_t	rgbAlphaBitMask;
		uint32_t	yuvAlphaBitMask;
		uint32_t	luminanceAlphaBitMask;
		uint32_t	rgbZBitMask;
		uint32_t	yuvZBitMask;
	} PACK_STRUCT;
} PACK_STRUCT;


struct ddsBuffer
{
	/* magic: 'dds ' */
	char				magic[ 4 ];

	/* directdraw surface */
	uint32_t		size;//4
	uint32_t		flags;//8
	uint32_t		height;//12
	uint32_t		width;//16
	union
	{
		int32_t				pitch;
		uint32_t	linearSize;
	} PACK_STRUCT;//20
	uint32_t		depth;//24
	union
	{
		uint32_t	mipMapCount;
		uint32_t	refreshRate;
		uint32_t	srcVBHandle;
	} PACK_STRUCT;//28
	uint32_t		alphaBitDepth;//32
	uint32_t		reserved;//36
	/// I think its time someone reminded the irrlicht folk that void* is non portable 32-64bit sizewise in structs
	uint32_t		surface_UNUSED;//40
	union
	{
		ddsColorKey	ckDestOverlay;
		uint32_t	emptyFaceColor;
	} PACK_STRUCT;//44
	ddsColorKey		ckDestBlt;//52
	ddsColorKey		ckSrcOverlay;//60
	ddsColorKey		ckSrcBlt;//68
	union
	{
		ddsPixelFormat	pixelFormat;
		uint32_t	fvf;
	} PACK_STRUCT;//76
	ddsCaps			caps;//
	uint32_t		textureStage;

	/* data (Varying size) */
	uint8_t		data[ 4 ];
} PACK_STRUCT;

/*
struct ddsColorBlock
{
	uint16_t		colors[ 2 ];
	uint8_t		row[ 4 ];
} PACK_STRUCT;


struct ddsAlphaBlockExplicit
{
	uint16_t		row[ 4 ];
} PACK_STRUCT;


struct ddsAlphaBlock3BitLinear
{
	uint8_t		alpha0;
	uint8_t		alpha1;
	uint8_t		stuff[ 6 ];
} PACK_STRUCT;


struct ddsColor
{
	uint8_t		r, g, b, a;
} PACK_STRUCT;*/


/* endian tomfoolery */
typedef union
{
	float	f;
	char	c[ 4 ];
}
floatSwapUnion PACK_STRUCT;

// Default alignment
#include "irrunpack.h"


	int32_t   DDSLittleLong( int32_t src );
	int16_t DDSLittleShort( int16_t src );
	float DDSLittleFloat( float src );

	int32_t DDSBigLong( int32_t src );

	int16_t DDSBigShort( int16_t src );

	float DDSBigFloat( float src );


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
	virtual std::vector<CImageData*> loadImage(io::IReadFile* file) const;
};


} // end namespace video
} // end namespace irr

#endif // compiled with DDS loader
#endif

