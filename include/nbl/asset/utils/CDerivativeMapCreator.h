#ifndef __NBL_I_DERIVATIVE_MAP_CREATOR_H_INCLUDED__
#define __NBL_I_DERIVATIVE_MAP_CREATOR_H_INCLUDED__

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUImageView.h"

namespace nbl {
namespace asset
{

class CDerivativeMapCreator
{
public:
    CDerivativeMapCreator() = delete;
    ~CDerivativeMapCreator() = delete;

    static core::smart_refctd_ptr<asset::ICPUImage> createDerivativeMapFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor);
    static core::smart_refctd_ptr<asset::ICPUImageView> createDerivativeMapViewFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor);
};

}
}

#endif
