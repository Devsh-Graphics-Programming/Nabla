// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_TEXTURE_VIEW_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_TEXTURE_VIEW_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IImageView.h"

#include "nbl/video/IGPUImage.h"

namespace nbl
{
namespace video
{
class IGPUImageView : public asset::IImageView<IGPUImage>
{
public:
    //! Regenerates the mip map levels of the texture.
    virtual void regenerateMipMapLevels() = 0;  // deprecated

    const SCreationParams& getCreationParameters() const { return params; }

protected:
    IGPUImageView(SCreationParams&& _params)
        : IImageView<IGPUImage>(std::move(_params)) {}
    virtual ~IGPUImageView() = default;
};

}
}

#endif