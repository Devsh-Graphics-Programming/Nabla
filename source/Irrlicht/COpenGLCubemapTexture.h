#ifndef __C_OPEN_GL_CUBE_MAP_TEXTURE_H_INCLUDED__
#define __C_OPEN_GL_CUBE_MAP_TEXTURE_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLTexture.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_



namespace irr
{
namespace video
{


//! OpenGL texture.
class COpenGLCubemapTexture : public COpenGLFilterableTexture
{
public:
	//! constructor
	COpenGLCubemapTexture(GLenum internalFormat, const uint32_t* size, uint32_t mipmapLevels, const io::path& name="");


	virtual IVirtualTexture::E_DIMENSION_COUNT getDimensionality() const {return IVirtualTexture::EDC_THREE;} //! or maybe two?

    virtual E_TEXTURE_TYPE getTextureType() const {return ETT_CUBE_MAP;}


	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const {return GL_TEXTURE_CUBE_MAP;}


    virtual bool updateSubRegion(const asset::E_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap=0, const uint32_t& unpackRowByteAlignment=0);
    virtual bool resize(const uint32_t* size, const uint32_t& mipLevels=0);

    //!
     static inline GLenum faceEnumToGLenum(const E_CUBE_MAP_FACE& face)
    {
        switch (face)
        {
            case ECMF_NEGATIVE_X:
                return GL_TEXTURE_CUBE_MAP_NEGATIVE_X;
                break;
            case ECMF_POSITIVE_X:
                return GL_TEXTURE_CUBE_MAP_POSITIVE_X;
                break;
            case ECMF_NEGATIVE_Y:
                return GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;
                break;
            case ECMF_POSITIVE_Y:
                return GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
                break;
            case ECMF_NEGATIVE_Z:
                return GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
                break;
            case ECMF_POSITIVE_Z:
                return GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
                break;
            case ECMF_COUNT:
                assert(0);
                break;
        }
        return GL_INVALID_ENUM;
    }
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_



