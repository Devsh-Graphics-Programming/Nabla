#ifndef _IRR_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_
#define _IRR_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_

//#include <irrlicht.h>
#include <cstdint>
#include <IReferenceCounted.h>
#include <IVideoDriver.h>

namespace irr {
    namespace video {
        class IVideoDriver;
        class IGPUBuffer;
        class ITexture;
    }
}

namespace irr
{
namespace ext
{
namespace Blur
{

class CBlurPerformer : public IReferenceCounted
{
    enum 
    {
        E_SSBO_BINDING = 0,
        E_IN_OFFSET_LOC = 2,
        E_OUT_OFFSET_LOC = 3,
        E_IN_MLT_LOC = 4,
        E_OUT_MLT_LOC = 5
    };

public:
    static CBlurPerformer* instantiate(video::IVideoDriver* _driver);

    video::ITexture* createBlurredTexture(const video::ITexture* _inputTex) const;

protected:
    ~CBlurPerformer();

private:
    CBlurPerformer(video::IVideoDriver* _driver, unsigned _sample, unsigned _gblur, unsigned _fblur) :
        m_driver(_driver),
        m_dsampleCs(_sample),
        m_blurGeneralCs(_gblur),
        m_blurFinalCs(_fblur)
    {
        m_ssbo = m_driver->createGPUBuffer(2 * 4 * 512 * 512 * sizeof(float), nullptr);
    }
private:
    static bool genBlurPassCs(char* _out, size_t _outSize, int _finalPass);

private:
    video::IVideoDriver* m_driver;
    unsigned m_dsampleCs, m_blurGeneralCs, m_blurFinalCs;
    video::IGPUBuffer* m_ssbo;

    static uint32_t s_texturesEverCreatedCount;
    static core::vector2d<size_t> s_outTexSize;
};

}
}
}

#endif