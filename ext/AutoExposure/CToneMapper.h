#ifndef _IRR_EXT_AUTO_EXPOSURE_C_TONE_MAPPER_INCLUDED_
#define _IRR_EXT_AUTO_EXPOSURE_C_TONE_MAPPER_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace AutoExposure
{

//! Tweakable defines
#define LOWER_LUMA_PERCENTILE 72
#define UPPER_LUMA_PERCENTILE 96

//do not touch this, affects compute occupancy
constexpr uint32_t SUBCELL_SIZE = 16;

class CToneMapper : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
        static CToneMapper* instantiateTonemapper(video::IVideoDriver* _driver,
                                                  const io::path& firstPassShaderFileName,
                                                  const io::path& secondPassShaderFileName,
                                                  const size_t& inputTexScaleOff, const size_t& percentilesOff, const size_t& outputOff);

        void setHistogramSamplingRate(float* outTexScale, uint32_t* percentileSearchVals,
                                      core::dimension2du viewportRes, const float inViewportScale[2])
        {
            viewportRes /= 2;
            viewportRes += core::dimension2du(SUBCELL_SIZE-1,SUBCELL_SIZE-1);
            m_workGroupCount[0] = viewportRes.Width/SUBCELL_SIZE;
            m_workGroupCount[1] = viewportRes.Height/SUBCELL_SIZE;
            m_totalThreadCount[0] = m_workGroupCount[0]*SUBCELL_SIZE;
            m_totalThreadCount[1] = m_workGroupCount[1]*SUBCELL_SIZE;

            outTexScale[0] = inViewportScale[0]/float(m_totalThreadCount[0]);
            outTexScale[1] = inViewportScale[1]/float(m_totalThreadCount[1]);

            uint32_t TOTAL_PIXEL_COUNT = m_totalThreadCount[0]*m_totalThreadCount[1];
            percentileSearchVals[0] = (LOWER_LUMA_PERCENTILE*TOTAL_PIXEL_COUNT)/100;
            percentileSearchVals[1] = (UPPER_LUMA_PERCENTILE*TOTAL_PIXEL_COUNT)/100;
        }

        bool CalculateFrameExposureFactors(video::IGPUBuffer* outBuffer, video::IGPUBuffer* uniformBuffer, video::ITexture* inputTexture);
    private:
        CToneMapper(video::IVideoDriver* _driver, const uint32_t& _histoProgram, const uint32_t& _autoExpProgram);
        ~CToneMapper();

        video::IVideoDriver* m_driver;
        video::IGPUBuffer* m_histogramBuffer;
        uint32_t m_histogramProgram,m_autoExpParamProgram;
        uint32_t m_workGroupCount[2];
        uint32_t m_totalThreadCount[2];
};

}
}
}

#endif
