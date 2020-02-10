#ifndef __IRR_I_GPU_TEXTURE_VIEW_H_INCLUDED__
#define __IRR_I_GPU_TEXTURE_VIEW_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

#include "irr/asset/IImageView.h"

#include "irr/video/IGPUImage.h"

namespace irr
{
namespace video
{

class IGPUImageView : public asset::IImageView<IGPUImage>
{
	public:
		//! Regenerates the mip map levels of the texture.
		virtual void regenerateMipMapLevels() = 0; // deprecated

        const SCreationParams& getCreationParameters() const { return params; }

	protected:
		IGPUImageView(SCreationParams&& _params) : IImageView<IGPUImage>(std::move(_params)) {}
		virtual ~IGPUImageView() = default;
};

}
}

#endif