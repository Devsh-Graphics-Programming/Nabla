#ifndef _IRR_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_
#define _IRR_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_

#include <cstdint>
#include <tuple>
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

    using tuple4xu32 = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

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
        float radius;
        uint32_t inOffset;
        uint32_t outOffset;
        uint32_t outMlt[2];
    };

    static inline size_t getRequiredUBOSize(video::IVideoDriver* driver)
    {
        return getSinglePaddedUBOSize(driver)*6u;
    }

    static CBlurPerformer* instantiate(video::IVideoDriver* _driver, float _radius, core::vector2d<uint32_t> _dsFactor,
                                       video::IGPUBuffer* uboBuffer=nullptr, const size_t& uboDataStaticOffset=0);

    video::ITexture* createOutputTexture(video::ITexture* _inputTex, const std::string& _name);

    //! _inputTexture and _outputTexture can be the same texture.
    //! Output texture's color format must be video::ECF_A16B16G16R16F
    void blurTexture(video::ITexture* _inputTex, video::ITexture* _outputTex);

    //! Establishes output size. Also creates shaders and SSBO and fills UBO with proper data if _createGpuStuff is true
    void prepareForBlur(const uint32_t* _inputSize, bool _createGpuStuff);
    // WARNING: does NOT change values of shader handles
    void deleteShaders() const;

    void writeUBOData(void* _dst) const;

    inline bool setUniformBuffer(video::IGPUBuffer* _ubo)
    {
        if (!_ubo || _ubo->getSize() < getRequiredUBOSize(m_driver))
            return false;

        m_isCustomUbo = true;

        video::IGPUBuffer* oldUbo = m_ubo;
        m_ubo = _ubo;
        if (m_ubo)
            m_ubo->grab();
        if (oldUbo)
            oldUbo->drop();

        return true;
    }

    inline bool setUBOOffset(uint32_t _offset)
    {
        if ((m_ubo && m_ubo->getSize() < _offset) || (m_ubo && m_ubo->getSize() - _offset < getRequiredUBOSize(m_driver)))
            return false;

        m_uboOffset = _offset;
        return true;
    }

    //! _radius must be a value from range [0.f, 1.f], otherwise gets clamped.
    //! Radius in this case indicates % of X and Y dimensions of output size.
    inline void setRadius(float _radius)
    { 
        _radius = std::max(0.f, std::min(_radius, 1.f));
        if (m_radius == _radius)
            return;
        m_radius = _radius;


        if (m_dsampleCs && !m_isCustomUbo) // no need to update UBO if no shaders are present yet
            updateUBO();
    }
    inline float getRadius() const { return m_radius; }

    inline core::vector2d<uint32_t> getDownsampleFactor() const { return m_dsFactor; }

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
    inline CBlurPerformer(video::IVideoDriver* _driver, float _radius,
                   core::vector2d<uint32_t> _dsFactor, video::IGPUBuffer* _uboBuffer, const size_t& _uboOffset) :
        m_driver(_driver),
        m_dsampleCs{0u},
        m_blurGeneralCs{0u, 0u},
        m_blurFinalCs{0u},
        m_samplesSsbo{nullptr},
        m_ubo(_uboBuffer),
        m_radius{std::max(0.f, std::min(_radius, 1.f))},
        m_paddedUBOSize(getSinglePaddedUBOSize(_driver)),
        m_uboOffset(_uboOffset),
        m_dsFactor(clampDsFactor(_dsFactor)),
        m_isCustomUbo(_uboBuffer)
    {
        if (!m_ubo)
        {
            video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.alignment = 4;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
            reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CANNOT_MAP;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            reqs.vulkanReqs.size = getRequiredUBOSize(m_driver)+m_uboOffset;

            m_ubo =  m_driver->createGPUBufferOnDedMem(reqs);
            // ubo gets filled with actual data when output size is established
        }
        else
            m_ubo->grab();
    }

    void updateUBO(const void* _contents);
    void updateUBO();

    static tuple4xu32 makeShaders(const core::vector2d<uint32_t>& _outSize);

    static bool genDsampleCs(char* _out, size_t _bufSize, const core::vector2d<uint32_t>& _outTexSize);
    static bool genBlurPassCs(char* _out, size_t _bufSize, uint32_t _outTexSize, int _finalPass);

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
    }

    inline static core::vector2d<uint32_t> clampDsFactor(core::vector2d<uint32_t> _dsf)
    {
        _dsf.X = std::max(s_MIN_DS_FACTOR, std::min(_dsf.X, s_MAX_DS_FACTOR));
        _dsf.Y = std::max(s_MIN_DS_FACTOR, std::min(_dsf.Y, s_MAX_DS_FACTOR));
        return _dsf;
    }

private:
    video::IVideoDriver* m_driver;
    uint32_t m_dsampleCs, m_blurGeneralCs[2], m_blurFinalCs;
    video::IGPUBuffer* m_samplesSsbo;
    video::IGPUBuffer* m_ubo;

    float m_radius;
    const uint32_t m_paddedUBOSize;
    uint32_t m_uboOffset;
    const core::vector2d<uint32_t> m_dsFactor;
    core::vector2d<uint32_t> m_outSize;
    bool m_isCustomUbo;

    static uint32_t s_MAX_WORK_GROUP_SIZE;
    static constexpr uint32_t s_ABSOLUTELY_MAX_WORK_GROUP_SIZE = 1024u;
    static constexpr uint32_t s_MAX_OUTPUT_SIZE_XY = 1024u;
    static constexpr uint32_t s_MIN_DS_FACTOR = 1u;
    static constexpr uint32_t s_MAX_DS_FACTOR = 16u;
};

}
}
}

#endif
