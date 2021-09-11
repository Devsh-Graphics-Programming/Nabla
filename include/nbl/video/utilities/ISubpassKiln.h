// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_I_SUBPASS_KILN_H_INCLUDED_
#define _NBL_VIDEO_I_SUBPASS_KILN_H_INCLUDED_


#include "nbl/video/utilities/IDrawIndirectAllocator.h"

#include <functional>


namespace nbl::video
{

class ISubpassKiln : public core::IReferenceCounted
{
    public:
        struct DrawcallInfo
        {
            alignas(16) uint8_t pushConstantData[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE]; // could try to push it to 64, if we had containers capable of such allocations
            core::smart_refctd_ptr<IGPUGraphicsPipeline> pipeline;
            core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSets[4] = {};
            asset::SBufferBinding<IGPUBuffer> m_vertexBufferBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
            asset::SBufferBinding<IGPUBuffer> m_indexBufferBinding;
            uint32_t drawCountOffset = IDrawIndirectAllocator::invalid_draw_count_ix;
            uint32_t drawCallOffset;
            uint32_t drawMaxCount = 0u;
            uint16_t orderNumber = 0u; // i.e. subpass ID
            // 2 bytes of padding remain

            inline bool operator<(const DrawcallInfo& rhs) const
            {
                return chainComparator<false,std::less>(rhs);
            }

        private:
            template<bool equalRetval, template<class> class op>
            inline bool chainComparator(const DrawcallInfo& rhs) const // why isnt something like this in the STL?
            {
                if (orderNumber==rhs.orderNumber)
                {/*
                    auto layout = pipeline->;
                    if ()
                    {
                        return op<>(drawCallOffset,rhs.drawCallOffset);
                    }*/
                    return false;
                }
                return op<decltype(orderNumber)>()(orderNumber,rhs.orderNumber);
            }
        };
};

}

#endif