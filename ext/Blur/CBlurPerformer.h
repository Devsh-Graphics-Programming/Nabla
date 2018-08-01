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
        uint32_t name;
        int level;
        uint8_t layered;
        int layer;
        uint32_t format, access;
    };

    enum
    {
        E_SAMPLES_SSBO_BINDING = 0,
        E_PSUM_SSBO_BINDING = 1,
        E_UBO_BINDING = 0
    };

public:
    struct BlurPassUBO
    {
        uint32_t iterNum;
        uint32_t padding;
        uint32_t inOffset;
        uint32_t outOffset;
        uint32_t outMlt[2];
    };

    static inline size_t getRequiredUBOSize(video::IVideoDriver* driver)
    {
        return getSinglePaddedUBOSize(driver)*6u;
    }

    static CBlurPerformer* instantiate(video::IVideoDriver* _driver, uint32_t _radius, core::vector2d<uint32_t> _outSize,
                                       video::IGPUBuffer* uboBuffer=nullptr, const size_t& uboDataStaticOffset=0);

    video::ITexture* createBlurredTexture(video::ITexture* _inputTex) const;

protected:
    static inline size_t getSinglePaddedUBOSize(video::IVideoDriver* driver)
    {
        size_t paddedSize = sizeof(BlurPassUBO)+driver->getRequiredUBOAlignment()-1u;
        paddedSize /= driver->getRequiredUBOAlignment();
        paddedSize *= driver->getRequiredUBOAlignment();
        return paddedSize;
    }

    ~CBlurPerformer();

private:
    CBlurPerformer(video::IVideoDriver* _driver, uint32_t _sample, uint32_t _sat, uint32_t _gblur, uint32_t _fblur, uint32_t _radius,
                   core::vector2d<uint32_t> _outSize, video::IGPUBuffer* uboBuffer, const size_t& uboDataStaticOffset) :
        m_driver(_driver),
        m_dsampleCs(_sample),
        m_psumCs(_sat),
        m_blurGeneralCs(_gblur),
        m_blurFinalCs(_fblur),
        m_radius(_radius),
        m_paddedUBOSize(getSinglePaddedUBOSize(_driver)),
        m_outSize(_outSize),
        m_ubo(uboBuffer),
        m_uboStaticOffset(uboDataStaticOffset)
    {
        assert(m_outSize.X == m_outSize.Y);

        video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.alignment = 4;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CANNOT_MAP;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;

        reqs.vulkanReqs.size = 2 * 2 * m_outSize.X * m_outSize.Y * sizeof(uint32_t);
        m_samplesSsbo = m_driver->createGPUBufferOnDedMem(reqs);

        reqs.vulkanReqs.size = 4 * padToPoT(m_outSize.X) * m_outSize.Y * sizeof(float);
        m_psumSsbo =  m_driver->createGPUBufferOnDedMem(reqs);

        if (!m_ubo)
        {
            reqs.vulkanReqs.size = getRequiredUBOSize(m_driver)+m_uboStaticOffset;
            m_ubo =  m_driver->createGPUBufferOnDedMem(reqs);
        }
        else
            m_ubo->grab();

        writeUBOData();
    }

    //! TODO: reduce this just to a write AND allow for dynamic `m_uboStaticOffset` (obvs. rename to m_uboOffset)
    /**
    Write should take an already offset (don't add m_uboStaticOffset) BlurPassUBO* pointer and iterate through the 6 things
    writeUBOData in the constructor, in case of missing m_ubo should handle the staging buffer and upload.

    Add a function to set `m_uboOffset` at will.
    **/
    void writeUBOData();
private:
    static bool genDsampleCs(char* _out, size_t _bufSize, uint32_t _outTexSize);
    static bool genBlurPassCs(char* _out, size_t _bufSize, uint32_t _outTexSize, uint32_t _radius, int _finalPass);
    static bool genPsumCs(char* _out, size_t _bufSize, uint32_t _outTexSize);

    void bindSSBuffers() const;
    static ImageBindingData getCurrentImageBinding(uint32_t _imgUnit);
    static void bindImage(uint32_t _imgUnit, const ImageBindingData& _data);

    void bindUbo(uint32_t _bnd, uint32_t _part) const;

    inline static uint32_t padToPoT(uint32_t _x)
    {
        --_x;
        for (uint32_t i = 1u; i <= 16u; i <<= 1)
            _x |= (_x >> i);
        return ++_x;
    };

private:
    video::IVideoDriver* m_driver;
    uint32_t m_dsampleCs, m_psumCs, m_blurGeneralCs, m_blurFinalCs;
    video::IGPUBuffer* m_samplesSsbo, *m_psumSsbo;
    video::IGPUBuffer* m_ubo;

    const uint32_t m_radius;
    const uint32_t m_paddedUBOSize;
    const uint32_t m_uboStaticOffset;
    const core::vector2d<uint32_t> m_outSize;

    static uint32_t s_texturesEverCreatedCount;
};

}
}
}

#endif
