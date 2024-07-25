#ifndef _NBL_I_DERIVATIVE_MAP_CREATOR_H_INCLUDED_
#define _NBL_I_DERIVATIVE_MAP_CREATOR_H_INCLUDED_

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/filters/CNormalMapToDerivativeFilter.h"
#include "nbl/asset/metadata/CDerivativeMapMetadata.h"

namespace nbl::asset
{

//! `isotropicNormalization` makes filter to use max value of all channels for normalization instead of per-channel max
class CDerivativeMapCreator
{
	public:
		CDerivativeMapCreator() = delete;
		~CDerivativeMapCreator() = delete;

		template<bool isotropicNormalization>
		static core::smart_refctd_ptr<asset::ICPUImage> createDerivativeMapFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor);
		template<bool isotropicNormalization>
		static core::smart_refctd_ptr<asset::ICPUImageView> createDerivativeMapViewFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor);

		//! Normalization is always done per-layer
		template<bool isotropicNormalization>
		static core::smart_refctd_ptr<asset::ICPUImage> createDerivativeMapFromNormalMap(asset::ICPUImage* _inImg, float* out_normalizationFactor);
		template<bool isotropicNormalization>
		static core::smart_refctd_ptr<asset::ICPUImageView> createDerivativeMapViewFromNormalMap(asset::ICPUImage* _inImg, float* out_normalizationFactor);

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

#endif // __NBL_I_DERIVATIVE_MAP_CREATOR_H_INCLUDED__
