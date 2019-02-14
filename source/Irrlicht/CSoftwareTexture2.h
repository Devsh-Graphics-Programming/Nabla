// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_SOFTWARE_2_TEXTURE_H_INCLUDED__
#define __C_SOFTWARE_2_TEXTURE_H_INCLUDED__

#include "SoftwareDriver2_compile_config.h"

#include "ITexture.h"
#include "CImage.h"

namespace irr
{
namespace video
{

/*!
	interface for a Video Driver dependent Texture.
*/
class CSoftwareTexture2 : public ITexture, public IDriverMemoryAllocation
{
protected:
	//! destructor
	virtual ~CSoftwareTexture2();

public:
	//! constructor
	enum eTex2Flags
	{
		GEN_MIPMAP	= 1,
		///IS_RENDERTARGET	= 2,
		NP2_SIZE	= 4,
		HAS_ALPHA	= 8
	};
	CSoftwareTexture2(asset::CImageData* surface, const io::path& name, uint32_t flags);

    virtual IVirtualTexture::E_DIMENSION_COUNT getDimensionality() const {return IVirtualTexture::EDC_TWO;}
    virtual bool updateSubRegion(const asset::E_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap=0, const uint32_t& unpackRowByteAlignment=0) {return false;}
    virtual bool resize(const uint32_t* size, const uint32_t& mipLevels=0) {return false;}
    virtual IVirtualTexture::E_VIRTUAL_TEXTURE_TYPE getVirtualTextureType() const {return IVirtualTexture::EVTT_OPAQUE_FILTERABLE;}
    virtual E_TEXTURE_TYPE getTextureType() const {return ETT_2D;}
	virtual uint32_t getMipMapLevelCount() const {return 1;}

	//!
    virtual size_t getAllocationSize() const {return 0xdeadbeefu;}
    virtual E_SOURCE_MEMORY_TYPE getType() const {return ESMT_NOT_DEVICE_LOCAL;}
    virtual void unmapMemory() {}
	virtual bool isDedicated() const {return true;}

	//!
	virtual IDriverMemoryAllocation* getBoundMemory() {return this;}
	virtual const IDriverMemoryAllocation* getBoundMemory() const {return this;}
	virtual size_t getBoundMemoryOffset() const {return 0ull;}

	//! lock function
	virtual void* lock(uint32_t mipmapLevel=0)
	{
		if (Flags & GEN_MIPMAP)
			MipMapLOD=mipmapLevel;
		return MipMap[MipMapLOD]->getData();
	}

	//! unlock function
	virtual void unlock()
	{
	}

	//! Returns original size of the texture.
	virtual const core::dimension2d<uint32_t>& getOriginalSize() const
	{
		//return MipMap[0]->getDimension();
		return OrigSize;
	}

	//! Returns the size of the largest mipmap.
	float getLODFactor( const float texArea ) const
	{
		return OrigImageDataSizeInPixels * texArea;
		//return MipMap[0]->getImageDataSizeInPixels () * texArea;
	}

	//! Returns (=size) of the texture.
	virtual const uint32_t* getSize() const
	{
		return &MipMap[MipMapLOD]->getDimension().Width;
	}
	virtual core::dimension2du getRenderableSize() const
	{
		return MipMap[MipMapLOD]->getDimension();
	}

	//! returns unoptimized surface
	virtual CImage* getImage() const
	{
		return MipMap[0];
	}

	//! returns texture surface
	virtual CImage* getTexture() const
	{
		return MipMap[MipMapLOD];
	}


	//! returns driver type of texture (=the driver, who created the texture)
	virtual E_DRIVER_TYPE getDriverType() const
	{
		return EDT_BURNINGSVIDEO;
	}

	//! returns color format of texture
	virtual asset::E_FORMAT getColorFormat() const
	{
		return BURNINGSHADER_COLOR_FORMAT;
	}

	//! returns pitch of texture (in bytes)
	virtual uint32_t getPitch() const
	{
		return MipMap[MipMapLOD]->getPitch();
	}

	//! Regenerates the mip map levels of the texture. Useful after locking and
	//! modifying the texture
	virtual void regenerateMipMapLevels();

	//! support mipmaps
	virtual bool hasMipMaps() const
	{
		return (Flags & GEN_MIPMAP ) != 0;
	}

	//! Returns if the texture has an alpha channel
	virtual bool hasAlpha() const
	{
		return (Flags & HAS_ALPHA ) != 0;
	}

private:
	float OrigImageDataSizeInPixels;
	core::dimension2d<uint32_t> OrigSize;

	CImage * MipMap[SOFTWARE_DRIVER_2_MIPMAPPING_MAX];

	uint32_t MipMapLOD;
	uint32_t Flags;
	asset::E_FORMAT OriginalFormat;
};


} // end namespace video
} // end namespace irr

#endif

