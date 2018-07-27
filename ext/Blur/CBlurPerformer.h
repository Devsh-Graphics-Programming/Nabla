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
    struct ImageBindingData
    {
        unsigned name;
        int level;
        unsigned char layered;
        int layer;
        unsigned format, access;
    };

    enum 
    {
        E_SAMPLES_SSBO_BINDING = 0,
        E_PSUM_SSBO_BINDING = 1,
        E_IN_OFFSET_LOC = 2,
        E_OUT_OFFSET_LOC = 3,
        E_ITER_NUM_LOC = 4,
        E_OUT_MLT_LOC = 5
    };

public:
    static CBlurPerformer* instantiate(video::IVideoDriver* _driver, uint32_t _radius, core::vector2d<uint32_t> _outSize);

    video::ITexture* createBlurredTexture(video::ITexture* _inputTex) const;

protected:
    ~CBlurPerformer();

private:
    CBlurPerformer(video::IVideoDriver* _driver, unsigned _sample, unsigned _sat, unsigned _gblur, unsigned _fblur, uint32_t _radius, core::vector2d<uint32_t> _outSize) :
        m_driver(_driver),
        m_dsampleCs(_sample),
        m_psumCs(_sat),
        m_blurGeneralCs(_gblur),
        m_blurFinalCs(_fblur),
        m_radius(_radius),
        m_outSize(_outSize)
    {
        assert(m_outSize.X == m_outSize.Y);

        m_samplesSsbo = m_driver->createGPUBuffer(2 * 2 * m_outSize.X * m_outSize.Y * sizeof(uint32_t), nullptr);
        m_psumSsbo = m_driver->createGPUBuffer(4 * nearestPowOf2(m_outSize.X) * m_outSize.Y * sizeof(float), nullptr);
    }
private:
    static bool genDsampleCs(char* _out, size_t _bufSize, uint32_t _outTexSize);
    static bool genBlurPassCs(char* _out, size_t _bufSize, uint32_t _outTexSize, uint32_t _radius, int _finalPass);
    static bool genPsumCs(char* _out, size_t _bufSize, uint32_t _outTexSize);

    void bindSSBuffers() const;
    static ImageBindingData getCurrentImageBinding(unsigned _imgUnit);
    static void bindImage(unsigned _imgUnit, const ImageBindingData& _data);

    inline static uint32_t nearestPowOf2(uint32_t _x)
    {
        --_x;
        for (unsigned i = 1u; i <= 16u; i <<= 1)
            _x |= (_x >> i);
        return ++_x;
    };

private:
    video::IVideoDriver* m_driver;
    unsigned m_dsampleCs, m_psumCs, m_blurGeneralCs, m_blurFinalCs;
    video::IGPUBuffer* m_samplesSsbo, *m_psumSsbo;

    const uint32_t m_radius;
    const core::vector2d<uint32_t> m_outSize;

    static uint32_t s_texturesEverCreatedCount;
};

}
}
}

#endif