#ifndef __NBL_I_DERIVATIVE_MAP_CREATOR_H_INCLUDED__
#define __NBL_I_DERIVATIVE_MAP_CREATOR_H_INCLUDED__

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/filters/CNormalMapToDerivativeFilter.h"

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

    static core::smart_refctd_ptr<asset::ICPUImage> createDerivativeMapFromNormalMap(asset::ICPUImage* _inImg);
    static core::smart_refctd_ptr<asset::ICPUImageView> createDerivativeMapViewFromNormalMap(asset::ICPUImage* _inImg);

private:
       
	static inline asset::E_FORMAT getRGformat(asset::E_FORMAT f)
	{
		const uint32_t bytesPerChannel = (getBytesPerPixel(f) * core::rational(1, getFormatChannelCount(f))).getIntegerApprox();

		switch (bytesPerChannel)
		{
			case 1u:
				return asset::EF_R8G8_SNORM;
			case 2u:
				return asset::EF_R16G16_SNORM;
			case 4u:
				return asset::EF_R32G32_SFLOAT;
			case 8u:
				return asset::EF_R64G64_SFLOAT;
			default:
				return asset::EF_UNKNOWN;
		}
	};
};

}
}

#endif
