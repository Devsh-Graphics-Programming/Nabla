
#include "IrrCompileConfig.h"

#include "SMaterialLayer.h"
#include "COpenGLTexture.h"

using namespace irr;
using namespace video;

uint64_t STextureSamplingParams::calculateHash(const IRenderableVirtualTexture* tex) const
{
#ifdef _IRR_COMPILE_WITH_OPENGL_
    if (!tex||tex->getDriverType()!=EDT_OPENGL)
        return 0;

    uint64_t zero64 = 0;
    STextureSamplingParams tmp = *reinterpret_cast<STextureSamplingParams*>(&zero64);

    bool couldWantToUseMipmaps = false;
    if (tex->getVirtualTextureType()==IRenderableVirtualTexture::EVTT_OPAQUE_FILTERABLE)
    {
        couldWantToUseMipmaps = UseMipmaps&&static_cast<const ITexture*>(tex)->hasMipMaps();

        switch (static_cast<const COpenGLFilterableTexture*>(tex)->getOpenGLTextureType())
        {
            case GL_TEXTURE_1D:
            case GL_TEXTURE_1D_ARRAY:
                tmp.TextureWrapU = 0xfu;
                tmp.TextureWrapV = 0;
                tmp.TextureWrapW = 0;
                tmp.AnisotropicFilter = 0xffu;
                tmp.SeamlessCubeMap = 0;
                tmp.MinFilter = couldWantToUseMipmaps ? 0xfu:0x1u;
                tmp.MaxFilter = 0x1u;
                break;
            case GL_TEXTURE_2D:
            case GL_TEXTURE_2D_ARRAY:
            case GL_TEXTURE_RECTANGLE: //?
                tmp.TextureWrapU = 0xfu;
                tmp.TextureWrapV = 0xfu;
                tmp.TextureWrapW = 0;
                tmp.AnisotropicFilter = 0xffu;
                tmp.SeamlessCubeMap = 0;
                tmp.MinFilter = couldWantToUseMipmaps ? 0xfu:0x1u;
                tmp.MaxFilter = 0x1u;
                break;
            case GL_TEXTURE_CUBE_MAP:
            case GL_TEXTURE_CUBE_MAP_ARRAY:
                tmp.TextureWrapU = 0xfu;
                tmp.TextureWrapV = 0xfu;
                tmp.TextureWrapW = 0;
                tmp.AnisotropicFilter = 0xffu;
                tmp.SeamlessCubeMap = 1;
                tmp.MinFilter = couldWantToUseMipmaps ? 0xfu:0x1u;
                tmp.MaxFilter = 0x1u;
                break;
            case GL_TEXTURE_3D:
                tmp.TextureWrapU = 0xfu;
                tmp.TextureWrapV = 0xfu;
                tmp.TextureWrapW = 0xfu;
                tmp.AnisotropicFilter = 0xffu;
                tmp.SeamlessCubeMap = 0;
                tmp.MinFilter = couldWantToUseMipmaps ? 0xfu:0x1u;
                tmp.MaxFilter = 0x1u;
                break;
            default:
                tmp.TextureWrapU = 0;
                tmp.TextureWrapV = 0;
                tmp.TextureWrapW = 0;
                tmp.AnisotropicFilter = 0;
                tmp.SeamlessCubeMap = 0;
                tmp.MinFilter = 0;
                tmp.MaxFilter = 0;
                break;
        }
    }
    else
    {
        tmp.TextureWrapU = 0;
        tmp.TextureWrapV = 0;
        tmp.TextureWrapW = 0;
        tmp.AnisotropicFilter = 0;
        tmp.SeamlessCubeMap = 0;
        tmp.MinFilter = 0;
        tmp.MaxFilter = 0;
    }

    tmp.UseMipmaps = 0;
    *((uint32_t*)&tmp.LODBias) = couldWantToUseMipmaps ? 0xffffffffu:0;

    uint64_t retval = *reinterpret_cast<const uint64_t*>(this);
    retval &= *reinterpret_cast<const uint64_t*>(&tmp);
    return retval;
#else
    return 0;
#endif // _IRR_COMPILE_WITH_OPENGL_
}
