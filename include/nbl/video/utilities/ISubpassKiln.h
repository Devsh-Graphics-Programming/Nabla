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
            public:
                alignas(16) uint8_t pushConstantData[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE]; // could try to push it to 64, if we had containers capable of such allocations
                core::smart_refctd_ptr<IGPUGraphicsPipeline> pipeline;
                core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSets[4] = {};
                asset::SBufferBinding<IGPUBuffer> vertexBufferBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
                asset::SBufferBinding<IGPUBuffer> indexBufferBinding;
                uint32_t drawCountOffset = IDrawIndirectAllocator::invalid_draw_count_ix;
                uint32_t drawCallOffset;
                uint32_t drawMaxCount = 0u;
                // 4 bytes of padding remain

                inline bool operator<(const DrawcallInfo& rhs) const
                {
                    return chainComparator<false,std::less>(rhs);
                }
                inline bool operator==(const DrawcallInfo& rhs) const
                {
                    return chainComparator<true,std::equal_to>(rhs);
                }

            private:
                template<bool equalRetval, template<class> class Cmp>
                inline bool chainComparator(const DrawcallInfo& rhs) const // why isnt something like this in the STL?
                {
                    if (pipeline->getRenderpass()==rhs.pipeline->getRenderpass())
                    {
                        if (pipeline->getSubpassIndex()==rhs.pipeline->getSubpassIndex())
                        {
                            auto indep = pipeline->getRenderpassIndependentPipeline();
                            auto other_indep = rhs.pipeline->getRenderpassIndependentPipeline();
                            auto layout = indep->getLayout();
                            auto other_layout = other_indep->getLayout();
                            auto pcranges = layout->getPushConstantRanges();
                            auto other_pcranges = other_layout->getPushConstantRanges();
                            if (pcranges.size()==other_pcranges.size())
                            {
                                const uint32_t pcrange_count = pcranges.size();
                                for (auto i=0u; i<pcrange_count; i++)
                                {
                                    const auto& lhs = pcranges.begin()[i];
                                    const auto& rhs = other_pcranges.begin()[i];
                                    if (lhs==rhs)
                                        continue;
                                    return Cmp<asset::SPushConstantRange>()(lhs,rhs);
                                }
                                for (auto i=0u; i<4u; i++)
                                {
                                    auto dsLayout = layout->getDescriptorSetLayout(i);
                                    auto other_dsLayout = other_layout->getDescriptorSetLayout(i);
                                    if (dsLayout==other_dsLayout)
                                        continue;
                                    return Cmp<const void*>()(dsLayout,other_dsLayout);
                                }
                                if (pipeline==rhs.pipeline)
                                {
                                    // first dset
                                    for (auto i=0u; i<4u; i++)
                                    {
                                        if (descriptorSets[i]==rhs.descriptorSets[i])
                                            continue;
                                        return Cmp<const void*>()(descriptorSets[i].get(),rhs.descriptorSets[i].get());
                                    }
                                    // then vertex bindings
                                    for (auto i=0u; i<IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
                                    {
                                        if (vertexBufferBindings[i].buffer==rhs.vertexBufferBindings[i].buffer)
                                        {
                                            if (vertexBufferBindings[i].offset,rhs.vertexBufferBindings[i].offset)
                                                continue;
                                            return Cmp<uint64_t>()(vertexBufferBindings[i].offset,rhs.vertexBufferBindings[i].offset);
                                        }
                                        return Cmp<const void*>()(vertexBufferBindings[i].buffer.get(),rhs.vertexBufferBindings[i].buffer.get());
                                    }
                                    // then index binding
                                    if (indexBufferBinding.buffer==rhs.indexBufferBinding.buffer)
                                    {
                                        if (indexBufferBinding.offset==rhs.indexBufferBinding.offset)
                                        {
                                            // then drawcall stuff
                                            if (drawCountOffset==rhs.drawCountOffset)
                                            {
                                                if (drawCallOffset==rhs.drawCallOffset)
                                                {
                                                    if (drawMaxCount==rhs.drawMaxCount)
                                                        return equalRetval;
                                                    return Cmp<uint32_t>()(drawMaxCount,rhs.drawMaxCount);
                                                }
                                                return Cmp<uint32_t>()(drawCallOffset,rhs.drawCallOffset);
                                            }
                                            return Cmp<uint32_t>()(drawCountOffset,rhs.drawCountOffset);
                                        }
                                        return Cmp<uint64_t>()(indexBufferBinding.offset,rhs.indexBufferBinding.offset);
                                    }
                                    return Cmp<const void*>()(indexBufferBinding.buffer.get(),rhs.indexBufferBinding.buffer.get());
                                }
                                return Cmp<const void*>()(pipeline.get(),rhs.pipeline.get());
                            }
                            return Cmp<size_t>()(pcranges.size(),other_pcranges.size());
                        }
                        return Cmp<uint32_t>()(pipeline->getSubpassIndex(),rhs.pipeline->getSubpassIndex());
                    }
                    return Cmp<const void*>()(pipeline->getRenderpass(),rhs.pipeline->getRenderpass());
                }
        };
};

}

#endif