// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_TEXTURE_H_INCLUDED__
#define __I_TEXTURE_H_INCLUDED__

#include "path.h"
#include "dimension2d.h"
#include "irr/asset/CImageData.h"
#include "IFrameBuffer.h"
#include "IVirtualTexture.h"
#include "IDriverMemoryBacked.h"

namespace irr
{
namespace video
{


//! Enumeration flags telling the video driver in which format textures should be created.
enum E_TEXTURE_CREATION_FLAG //depr
{
	/** Forces the driver to create 16 bit textures always, independent of
	which format the file on disk has. When choosing this you may lose
	some color detail, but gain much speed and memory. 16 bit textures can
	be transferred twice as fast as 32 bit textures and only use half of
	the space in memory.
	When using this flag, it does not make sense to use the flag
	ETCF_ALWAYS_32_BIT at the same time. */
	ETCF_ALWAYS_16_BIT = 0x00000001,

	/** Forces the driver to create 32 bit textures always, independent of
	which format the file on disk has. Please note that some drivers (like
	the software device) will ignore this, because they are only able to
	create and use 16 bit textures.
	When using this flag, it does not make sense to use the flag
	ETCF_ALWAYS_16_BIT at the same time. */
	ETCF_ALWAYS_32_BIT = 0x00000002,

	/** Automatically creates mip map levels for the textures. */
	ETCF_CREATE_MIP_MAPS = 0x00000004,

	/** Discard any alpha layer and use non-alpha color format. */
	ETCF_NO_ALPHA_CHANNEL = 0x00000008,

	/** This flag is never used, it only forces the compiler to compile
	these enumeration values to 32 bit. */
	ETCF_FORCE_32_BIT_DO_NOT_USE = 0x7fffffff
};

//! Interface of a Video Driver dependent Texture.
/** An ITexture is created by an IVideoDriver by using IVideoDriver::addTexture
or IVideoDriver::getTexture. After that, the texture may only be used by this
VideoDriver. As you can imagine, textures of the DirectX and the OpenGL device
will, e.g., not be compatible. An exception is the Software device and the
NULL device, their textures are compatible. If you try to use a texture
created by one device with an other device, the device will refuse to do that
and write a warning or an error message to the output buffer.
*/
class ITexture : public core::impl::ResolveAlignment<IDriverMemoryBacked,IRenderableVirtualTexture>
{
    public:
        enum E_TEXTURE_TYPE
        {
            ETT_1D=0,
            ETT_2D,
            ETT_3D,
            ETT_1D_ARRAY,
            ETT_2D_ARRAY,
            ETT_CUBE_MAP,
            ETT_CUBE_MAP_ARRAY,
            ETT_COUNT
        };
        enum E_CUBE_MAP_FACE
        {
            ECMF_POSITIVE_X=0,
            ECMF_NEGATIVE_X,
            ECMF_POSITIVE_Y,
            ECMF_NEGATIVE_Y,
            ECMF_POSITIVE_Z,
            ECMF_NEGATIVE_Z,
            ECMF_COUNT
        };

        virtual E_TEXTURE_TYPE getTextureType() const = 0;

        //! Get dimension (=size) of the texture.
        /** \return The size of the texture. */
        virtual uint32_t getMipMapLevelCount() const = 0;

        //!
        virtual bool updateSubRegion(const asset::E_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap=0, const uint32_t& unpackRowByteAlignment=0) = 0;
        virtual bool resize(const uint32_t* size, const uint32_t& mipLevels=0) = 0;

        //! Get pitch of the main texture (in bytes).
        /** The pitch is the amount of bytes used for a row of pixels in a
        texture.
        \return Pitch of texture in bytes. */
        virtual uint32_t getPitch() const = 0;

        //! Check whether the texture has MipMaps
        /** \return True if texture has MipMaps, else false. */
        virtual bool hasMipMaps() const { return false; }

        //! Regenerates the mip map levels of the texture.
        /** Required after modifying the texture, usually after calling unlock().
        \param mipmapData Optional parameter to pass in image data which will be
        used instead of the previously stored or automatically generated mipmap
        data. The data has to be a continuous pixel data for all mipmaps until
        1x1 pixel. Each mipmap has to be half the width and height of the previous
        level. At least one pixel will be always kept.*/
        virtual void regenerateMipMapLevels() = 0;

        //! Get name of texture (in most cases this is the filename)
        const io::SNamedPath& getName() const { return NamedPath; }


        _IRR_RESOLVE_NEW_DELETE_AMBIGUITY(IRenderableVirtualTexture,IDriverMemoryBacked)
    protected:
        _IRR_INTERFACE_CHILD(ITexture) {}

        //! constructor
        ITexture(const IDriverMemoryBacked::SDriverMemoryRequirements& reqs, const io::path& name) : NamedPath(name)
        {
            IDriverMemoryBacked::cachedMemoryReqs = reqs;
        }

        io::SNamedPath NamedPath;
};


} // end namespace video
} // end namespace irr

#endif

