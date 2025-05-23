// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_IMAGE_VIEW_H_INCLUDED_
#define _NBL_ASSET_I_CPU_IMAGE_VIEW_H_INCLUDED_

#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/IImageView.h"

namespace nbl::asset
{

class ICPUImageView final : public IImageView<ICPUImage>, public IAsset
{
	public:
		static core::smart_refctd_ptr<ICPUImageView> create(SCreationParams&& params)
		{
			// default aspect masks if none supplied
			if (!params.subresourceRange.aspectMask)
			{
				if (isDepthOrStencilFormat(params.format))
				{
					if (!isDepthOnlyFormat(params.format))
						params.subresourceRange.aspectMask |= ICPUImage::EAF_STENCIL_BIT;
					if (!isStencilOnlyFormat(params.format))
						params.subresourceRange.aspectMask |= ICPUImage::EAF_DEPTH_BIT;
				}
				else
					params.subresourceRange.aspectMask |= ICPUImage::EAF_COLOR_BIT;
			}
			if (!validateCreationParameters(params))
				return nullptr;

			return core::make_smart_refctd_ptr<ICPUImageView>(std::move(params));
		}
		ICPUImageView(SCreationParams&& _params) : IImageView<ICPUImage>(std::move(_params)) {}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto par = params;
            if (_depth > 0u && par.image)
                par.image = core::smart_refctd_ptr_static_cast<ICPUImage>(par.image->clone(_depth-1u));

            return core::make_smart_refctd_ptr<ICPUImageView>(std::move(par));
        }

		//!
		constexpr static inline auto AssetType = ET_IMAGE_VIEW;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

    inline core::unordered_set<const IAsset*> computeDependants() const override
		{
			return computeDependantsImpl(this);
		}

    inline core::unordered_set<IAsset*> computeDependants() override
		{
			return computeDependantsImpl(this);
		}

		//!
		const SComponentMapping& getComponents() const { return params.components; }
		SComponentMapping&	getComponents() 
		{ 
			assert(isMutable());
			return params.components;
		}

		inline void setAspectFlags(core::bitflag<IImage::E_ASPECT_FLAGS> aspect)
		{
			params.subresourceRange.aspectMask = aspect.value;
		}

	protected:
		virtual ~ICPUImageView() = default;

  private:
    template <typename Self>
      requires(std::same_as<std::remove_cv_t<Self>, ICPUImageView>)
    static auto computeDependantsImpl(Self* self) {
        using asset_ptr_t = std::conditional_t<std::is_const_v<Self>, const IAsset*, IAsset*>;
        return core::unordered_set<asset_ptr_t>{ self->params.image.get() };
    }
};

}

#endif