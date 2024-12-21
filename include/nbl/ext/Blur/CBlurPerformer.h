// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_
#define _NBL_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_

#include <cstdint>
#include <tuple>
#include "nbl/core/IReferenceCounted.h"
#include <nabla.h>


namespace nbl::ext::Blur
{

#if 0 // TODO: redo
class CBlurPerformer : public core::IReferenceCounted
{
    struct ImageBindingData
    {
        uint32_t name;
        int level;
        uint8_t layered;
        int layer;
        uint32_t format, access;
    };

    using tuple2xu32 = std::tuple<uint32_t, uint32_t>;

    enum
    {
        E_SAMPLES_SSBO_BINDING = 0,
        E_PSUM_SSBO_BINDING = 1,
        E_UBO_BINDING = 0
    };

public:
    struct BlurPassUBO // i'll leave it as a struct in case of changes in the future
    {
        float radius;
    };

    static inline size_t getRequiredUBOSize(video::IVideoDriver* _driver)
    {
        return nbl::core::alignUp(sizeof(BlurPassUBO), 16u);
    }

    //! Instantiates blur performer.
    /**
    @param _radius Radius of blur in both axes. Must be in range [0; 1] since it indicates % of output texture size in X and Y axes.
    @param _dsFactor Downscale factor of output texture relatively to input texture.
    @param _passesPerAxis Number of box blur passes that will be executed in both axes.
    @param _outputColorFmt Format of output texture. Must be video::ECT_A16B16G16R16F or video::ECF_RGB9_E5.
    @param uboBuffer Optional custom uniform buffer. If nullptr, then Blur performer will create its own. UBO is not updated internally by blur performer if custom uniform buffer is set.
    @param uboOffset Offset in uniform buffer. Irrelevant if `uboBuffer` is nullptr.
    */
    static CBlurPerformer* instantiate(video::IVideoDriver* _driver, const asset::IIncludeHandler* _inclhandler, float _radius, core::vector2d<uint32_t> _dsFactor, uint32_t _passesPerAxis = 2u,
                                       asset::E_FORMAT _outputColorFmt = asset::EF_R16G16B16A16_SFLOAT, video::IGPUBuffer* uboBuffer=nullptr, const size_t& uboOffset=0);

    core::smart_refctd_ptr<video::ITexture> createOutputTexture(video::ITexture* _inputTex);

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


        if (m_blurGeneralCs[0] && !m_isCustomUbo) // no need to update UBO if no shaders are present yet
            updateUBO();
    }
    inline float getRadius() const { return m_radius; }

    inline core::vector2d<uint32_t> getDownsampleFactor() const { return m_dsFactor; }

protected:
    // Not used as for now
    static inline size_t getSinglePaddedUBOSize(video::IVideoDriver* driver)
    {
        size_t paddedSize = sizeof(BlurPassUBO)+driver->getRequiredUBOAlignment()-1u;
        paddedSize /= driver->getRequiredUBOAlignment();
        paddedSize *= driver->getRequiredUBOAlignment();
        return paddedSize;
    }
    inline size_t getUBOSizePerShaderPass() const
    {
        return getRequiredUBOSize(m_driver);
    }

    ~CBlurPerformer();

private:
    inline CBlurPerformer(video::IVideoDriver* _driver, const asset::IIncludeHandler* _inclhandler, float _radius,
                   core::vector2d<uint32_t> _dsFactor, uint32_t _passesPerAxis, asset::E_FORMAT _colorFmt, video::IGPUBuffer* _uboBuffer, const size_t& _uboOffset) :
        m_driver(_driver),
        m_inclHandler(_inclhandler),
        m_blurGeneralCs{0u, 0u},
        m_samplesSsbo{nullptr},
        m_ubo(_uboBuffer),
        m_radius{std::max(0.f, std::min(_radius, 1.f))},
        m_passesPerAxisNum{_passesPerAxis},
        //m_paddedUBOSize(getSinglePaddedUBOSize(_driver)),
        m_uboOffset(_uboOffset),
        m_dsFactor(validateDsFactor(_dsFactor)),
        m_isCustomUbo(_uboBuffer),
        m_outputColorFormat(_colorFmt)
    {
        if (!m_ubo)
        {
            video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs;
            reqs.vulkanReqs.alignment = 4;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = video::IDeviceMemoryAllocation::ESMT_DEVICE_LOCAL;
            reqs.mappingCapability = video::IDeviceMemoryAllocation::EMCF_CANNOT_MAP;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            //reqs.vulkanReqs.size = getRequiredUBOSize(m_driver)+m_uboOffset;
            reqs.vulkanReqs.size = getRequiredUBOSize(_driver);

            m_ubo =  m_driver->createGPUBufferOnDedMem(reqs);
            // ubo gets filled with actual data when output size is established
        }
        else
            m_ubo->grab();
    }

    void updateUBO(const void* _contents);
    void updateUBO();

    static tuple2xu32 makeShaders(video::IVideoDriver* _driver, const asset::IIncludeHandler* _inclhandler, const core::vector2d<uint32_t>& _outSize, const core::vector2d<uint32_t>& _dsf, uint32_t _passesPerAxis, asset::E_FORMAT _colorFmt);

    static bool genBlurPassCs(char* _out, video::IVideoDriver* _driver, const asset::IIncludeHandler* _inclhandler, size_t _bufSize, uint32_t _axisSize, const core::vector2d<uint32_t>& _outTexSize, uint32_t _passes, const core::vector2d<uint32_t>& _dsf, asset::E_FORMAT _colorFmt, int _finalPass);

    void bindSSBuffers() const;
    static ImageBindingData getCurrentImageBinding(uint32_t _imgUnit);
    static void bindImage(uint32_t _imgUnit, const ImageBindingData& _data);

    void bindUBO(uint32_t _bnd) const;

    inline static uint32_t padToPoT(uint32_t _x)
    {
        --_x;
        for (uint32_t i = 1u; i <= 16u; i <<= 1)
            _x |= (_x >> i);
        return ++_x;
    }

    //! Clamps ds factor and rounds to factor of two
    inline static core::vector2d<uint32_t> validateDsFactor(core::vector2d<uint32_t> _dsf)
    {
        _dsf.X += (_dsf.X & 1u);
        _dsf.Y += (_dsf.Y & 1u);
        _dsf.X = std::max(s_MIN_DS_FACTOR, std::min(_dsf.X, s_MAX_DS_FACTOR));
        _dsf.Y = std::max(s_MIN_DS_FACTOR, std::min(_dsf.Y, s_MAX_DS_FACTOR));
        return _dsf;
    }

private:
    video::IVideoDriver* m_driver;
    const asset::IIncludeHandler* m_inclHandler;
    uint32_t m_blurGeneralCs[2];
    video::IGPUBuffer* m_samplesSsbo;
    video::IGPUBuffer* m_ubo;

    float m_radius;
    uint32_t m_passesPerAxisNum;
    //const uint32_t m_paddedUBOSize;
    uint32_t m_uboOffset;
    const core::vector2d<uint32_t> m_dsFactor;
    core::vector2d<uint32_t> m_outSize;
    bool m_isCustomUbo;
    const asset::E_FORMAT m_outputColorFormat;

    static uint32_t s_MAX_WORK_GROUP_SIZE;
    static constexpr uint32_t s_ABSOLUTELY_MAX_WORK_GROUP_SIZE = 1024u;
    static constexpr uint32_t s_MAX_OUTPUT_SIZE_XY = 1024u;
    static constexpr uint32_t s_MIN_DS_FACTOR = 1u;
    static constexpr uint32_t s_MAX_DS_FACTOR = 16u;
};
#endif

}

#endif
