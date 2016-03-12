#ifndef __C_OPEN_GL_3D_TEXTURE_H_INCLUDED__
#define __C_OPEN_GL_3D_TEXTURE_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLTexture.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_



namespace irr
{
namespace video
{

//! OpenGL texture.
class COpenGL3DTexture : public COpenGLTexture
{
public:

	//! constructor
	COpenGL3DTexture(core::vector3d<u32> size, GLenum format, GLenum inDataFmt, GLenum inDataTpe, const io::path& name, void* data, void* mipmapData=0, COpenGLDriver* driver=0, u32 mipmapLevels=0);

	//! destructor
	virtual ~COpenGL3DTexture();

	//! Returns size of the texture.
	virtual const core::vector3d<u32>& getSize3D() const;

	//! returns pitch of texture (in bytes)
	virtual u32 getPitch() const;



	//! return whether this texture has mipmaps
	virtual void* lock(E_TEXTURE_LOCK_MODE mode=ETLM_READ_WRITE, u32 mipmapLevel=0) {return NULL;}
	virtual void unlock() {}
	virtual const core::dimension2d<u32>& getSize() const {return TextureSize;}
    virtual const core::dimension2du& getRenderableSize() const {return *reinterpret_cast<const core::dimension2du*>(&TextureSize3D);}


	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const;

    virtual void updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, void* data, u32 minX, u32 minY, u32 minZ, u32 maxX, u32 maxY, u32 maxZ, s32 mipmap=0);
    virtual void resize(core::vector3d<u32> size, u32 mipLevels=0);


protected:

	core::vector3d<u32> OrigSize3D;
	core::vector3d<u32> TextureSize3D;
};

//! OpenGL texture.
class COpenGL2DTextureArray : public COpenGLTexture
{
public:

//! needs my attention
	//! constructor
	COpenGL2DTextureArray(core::vector3d<u32> size, ECOLOR_FORMAT format, const io::path& name, void* mipmapData, COpenGLDriver* driver=0, u32 mipmapLevels=0);

	//! destructor
	virtual ~COpenGL2DTextureArray();

	//! Returns size of the texture.
	virtual const core::vector3d<u32>& getSize3D() const;

	//! returns pitch of texture (in bytes)
	virtual u32 getPitch() const;



	//! return whether this texture has mipmaps
	virtual void* lock(E_TEXTURE_LOCK_MODE mode=ETLM_READ_WRITE, u32 mipmapLevel=0) {return NULL;}
	virtual void unlock() {}
	virtual const core::dimension2d<u32>& getSize() const {return TextureSize;}
    virtual const core::dimension2du& getRenderableSize() const {return *reinterpret_cast<const core::dimension2du*>(&TextureSize3D);}



	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const;

	//virtual bool is3D() const { return true; }

//! needs my attention
    virtual void updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, void* data, u32 minX, u32 minY, u32 minZ, u32 maxX, u32 maxY, u32 maxZ, s32 mipmap=0);
    virtual void resize(core::vector3d<u32> size, u32 mipLevels=0);


protected:

	core::vector3d<u32> OrigSize3D;
	core::vector3d<u32> TextureSize3D;
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_

