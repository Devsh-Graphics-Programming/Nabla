// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GPU_MESH_BUFFER_H_INCLUDED__
#define __I_GPU_MESH_BUFFER_H_INCLUDED__

#include <algorithm>

#include "irr/asset/asset.h"

#include "ITransformFeedback.h"
#include "IGPUBuffer.h"
#include "IGPUDescriptorSet.h"
#include "IGPURenderpassIndependentPipeline.h"

namespace irr
{
namespace video
{
	class IGPUMeshBuffer final : public asset::IMeshBuffer<IGPUBuffer,IGPUDescriptorSet,IGPURenderpassIndependentPipeline>
	{
    public:
        _IRR_STATIC_INLINE_CONSTEXPR size_t PUSH_CONSTANTS_BYTESIZE = 128u;

        uint8_t* getPushConstantsDataPtr() { return m_pushConstantsData; }
        const uint8_t* getPushConstantsDataPtr() const { return m_pushConstantsData; }

    private:
        uint8_t m_pushConstantsData[PUSH_CONSTANTS_BYTESIZE];
	};

} // end namespace video
} // end namespace irr



#endif


