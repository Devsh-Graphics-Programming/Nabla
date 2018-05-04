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
        E_SSBO0_BINDING = 0,
        E_SSBO1_BINDING = 1
    };

public:
    static CBlurPerformer* instantiate(video::IVideoDriver* _driver);

    video::ITexture* createBlurredTexture(const video::ITexture* _inputTex) const;

protected:
    ~CBlurPerformer();

private:
    CBlurPerformer(video::IVideoDriver* _driver, unsigned _sample, unsigned _hblur1, unsigned _hblur2, unsigned _hblur3, unsigned _vblur1, unsigned _vblur2, unsigned _vblur3) :
        m_driver(_driver),
        m_dsampleCs(_sample),
        m_blurCs {_hblur1, _hblur2, _hblur3, _vblur1, _vblur2, _vblur3}
    {
        m_ssbo0 = m_driver->createGPUBuffer(4 * 512 * 512 * sizeof(float), nullptr);
        m_ssbo1 = m_driver->createGPUBuffer(4 * 512 * 512 * sizeof(float), nullptr);
    }
private:
    static bool genBlurPassCs(char* _out, size_t _outSize, const char* _inBufName, const char* _outBufName, const char* _inIdxName, const char* _outIdxName, int _finalPass);

private:
    video::IVideoDriver* m_driver;
    unsigned m_dsampleCs, m_blurCs[6];
    video::IGPUBuffer* m_ssbo0, *m_ssbo1;

    static uint32_t s_texturesEverCreatedCount;
};

}
}
}

#endif