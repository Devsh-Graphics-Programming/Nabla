// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_SAMPLER_H_INCLUDED__
#define __NBL_ASSET_I_CPU_SAMPLER_H_INCLUDED__

#include "nbl/asset/IAsset.h"
#include "nbl/asset/ISampler.h"

namespace nbl
{
namespace asset
{

class ICPUSampler : public ISampler, public IAsset
{
	protected:
		virtual ~ICPUSampler() = default;

	public:
		ICPUSampler(const SParams& _params) : ISampler(_params), IAsset() {}
        
		//
		static inline core::vectorSIMDu32 wrapTextureCoordinate(core::vectorSIMDi32 texelCoord, const ISampler::E_TEXTURE_CLAMP wrapModes[3],
																const core::vector3du32_SIMD& mipExtent, const core::vector3du32_SIMD& mipLastCoord)
		{
			for (auto i=0; i<3; i++)
			{
				const int32_t originalWasNegative = texelCoord[i]<0 ? 1:0;
				auto repeat = [&texelCoord,i,&mipExtent,&mipLastCoord,originalWasNegative]()
				{
					texelCoord[i] %= int32_t(mipExtent[i]);
					if (originalWasNegative)
						texelCoord[i] = (texelCoord[i] ? mipExtent[i]:mipLastCoord[i]) + texelCoord[i];
				};
				switch (wrapModes[i])
				{
					case ISampler::ETC_REPEAT:
						repeat();
						break;
					case ISampler::ETC_CLAMP_TO_EDGE:
						texelCoord[i] = core::clamp<int32_t,int32_t>(texelCoord[i],0,mipLastCoord[i]);
						break;
					case ISampler::ETC_MIRROR_CLAMP_TO_EDGE:
						texelCoord[i] = core::clamp<int32_t,int32_t>(texelCoord[i],-int32_t(mipExtent[i]),mipExtent[i]+mipLastCoord[i]);
					case ISampler::ETC_MIRROR:
						{
							int32_t repeatID = (originalWasNegative+texelCoord[i])/int32_t(mipExtent[i]);
							repeat();
							if ((repeatID&0x1)!=originalWasNegative)
								texelCoord[i] = mipLastCoord[i]-texelCoord[i];
						}
						break;
					default:
						// TODO: Handle borders, would have to have a static globally initialized memory array, with blocks of each <E_FORMAT,E_TEXTURE_BORDER_COLOR> combo
						assert(false);
						break;
				}
			}
			return std::move(reinterpret_cast<core::vectorSIMDu32&>(texelCoord));
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<ICPUSampler>(m_params);
            clone_common(cp.get());

            return cp;
        }

		size_t conservativeSizeEstimate() const override { return sizeof(m_params); }

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_SAMPLER;
		inline E_TYPE getAssetType() const override { return AssetType; }

		bool compatible(const IAsset* _other) const override
		{
			return true;
		}
protected:
		nbl::core::vector<const IAsset*> getMembersToRecurse() const override { return {}; }
	
};

}
}

#endif