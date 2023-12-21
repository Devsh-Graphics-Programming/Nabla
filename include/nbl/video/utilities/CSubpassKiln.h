// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_SUBPASS_KILN_H_INCLUDED_
#define _NBL_VIDEO_C_SUBPASS_KILN_H_INCLUDED_


#include "nbl/video/SPhysicalDeviceLimits.h"
#include "nbl/video/utilities/IDrawIndirectAllocator.h"

#include <functional>


namespace nbl::video
{
    
class CSubpassKiln
{
    public:
        static void enableRequiredFeautres(SPhysicalDeviceFeatures& featuresToEnable)
        {
        }

        static void enablePreferredFeatures(const SPhysicalDeviceFeatures& availableFeatures, SPhysicalDeviceFeatures& featuresToEnable)
        {
        }

        // for finding upper and lower bounds of subpass drawcalls
        struct SearchObject
        {
            const IGPURenderpass* renderpass;
            uint32_t subpassIndex;
        };
        //
        struct DrawcallInfo
        {
            alignas(16) uint8_t pushConstantData[SPhysicalDeviceLimits::MaxMaxPushConstantsSize]; // could try to push alignment to 64, if we had containers capable of such allocations
            core::smart_refctd_ptr<const IGPUGraphicsPipeline> pipeline;
            core::smart_refctd_ptr<const IGPUDescriptorSet> descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = {};
            asset::SBufferBinding<IGPUBuffer> vertexBufferBindings[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT] = {};
            core::smart_refctd_ptr<const IGPUBuffer> indexBufferBinding;
            uint32_t drawCommandStride : 31;
            uint32_t indexType : 2;
            uint32_t drawCountOffset = IDrawIndirectAllocator::invalid_draw_count_ix;
            uint32_t drawCallOffset;
            uint32_t drawMaxCount = 0u;
        };
        // 
        struct DefaultOrder
        {
            public:
                static inline constexpr uint64_t invalidTypeID = 0ull;
                static inline constexpr uint64_t typeID = 1ull;

                struct less
                {
                    inline bool operator()(const DrawcallInfo& lhs, const DrawcallInfo& rhs) const
                    {
                        return chainComparator<false,std::less>(lhs,rhs);
                    }
                };
                struct equal_to
                {
                    inline bool operator()(const DrawcallInfo& lhs, const DrawcallInfo& rhs) const
                    {
                        return chainComparator<true,std::equal_to>(lhs,rhs);
                    }
                };
                struct renderpass_subpass_comp
                {
                    inline bool operator()(const SearchObject& lhs, const DrawcallInfo& rhs)
                    {
                        const auto renderpass = rhs.pipeline->getRenderpass();
                        if (lhs.renderpass==rhs.pipeline->getRenderpass())
                            return lhs.subpassIndex<rhs.pipeline->getCachedCreationParams().subpassIx;
                        return lhs.renderpass<renderpass;
                    }
                    inline bool operator()(const DrawcallInfo& lhs, const SearchObject& rhs)
                    {
                        const auto renderpass = lhs.pipeline->getRenderpass();
                        if (lhs.pipeline->getRenderpass()==rhs.renderpass)
                            return lhs.pipeline->getCachedCreationParams().subpassIx<rhs.subpassIndex;
                        return renderpass<rhs.renderpass;
                    }
                };

            private:
                template<bool equalRetval, template<class> class Cmp>
                static inline bool chainComparator(const DrawcallInfo& lhs, const DrawcallInfo& rhs) // why isnt something like this in the STL? (apparently std::tie could be used here?)
                {
                    if (lhs.pipeline->getRenderpass()==rhs.pipeline->getRenderpass())
                    {
                        if (lhs.pipeline->getCachedCreationParams().subpassIx==rhs.pipeline->getCachedCreationParams().subpassIx)
                        {
                            auto lhs_layout = lhs.pipeline->getLayout();
                            auto rhs_layout = rhs.pipeline->getLayout();
                            auto lhs_pcranges = lhs_layout->getPushConstantRanges();
                            auto rhs_pcranges = rhs_layout->getPushConstantRanges();
                            if (lhs_pcranges.size()==rhs_pcranges.size())
                            {
                                const uint32_t pcrange_count = lhs_pcranges.size();
                                for (auto i=0u; i<pcrange_count; i++)
                                {
                                    const auto& _lhs = lhs_pcranges.begin()[i];
                                    const auto& _rhs = rhs_pcranges.begin()[i];
                                    if (_lhs==_rhs)
                                        continue;
                                    return Cmp<asset::SPushConstantRange>()(_lhs,_rhs);
                                }
                                for (auto i=0u; i<IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
                                {
                                    auto lhs_dsLayout = lhs_layout->getDescriptorSetLayout(i);
                                    auto rhs_dsLayout = rhs_layout->getDescriptorSetLayout(i);
                                    if (lhs_dsLayout==rhs_dsLayout)
                                        continue;
                                    return Cmp<const void*>()(lhs_dsLayout,rhs_dsLayout);
                                }
                                if (lhs.pipeline==rhs.pipeline)
                                {
                                    // first dset
                                    for (auto i=0u; i<4u; i++)
                                    {
                                        if (lhs.descriptorSets[i]==rhs.descriptorSets[i])
                                            continue;
                                        return Cmp<const void*>()(lhs.descriptorSets[i].get(),rhs.descriptorSets[i].get());
                                    }
                                    // then vertex bindings
                                    for (auto i=0u; i<asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
                                    {
                                        if (lhs.vertexBufferBindings[i].buffer==rhs.vertexBufferBindings[i].buffer)
                                        {
                                            if (lhs.vertexBufferBindings[i].offset,rhs.vertexBufferBindings[i].offset)
                                                continue;
                                            return Cmp<uint64_t>()(lhs.vertexBufferBindings[i].offset,rhs.vertexBufferBindings[i].offset);
                                        }
                                        return Cmp<const void*>()(lhs.vertexBufferBindings[i].buffer.get(),rhs.vertexBufferBindings[i].buffer.get());
                                    }
                                    // then index binding
                                    if (lhs.indexBufferBinding==rhs.indexBufferBinding)
                                    {
                                        // then drawcall stuff
                                        if (lhs.indexType==rhs.indexType)
                                        {
                                            if (lhs.drawCommandStride==rhs.drawCommandStride)
                                            {
                                                if (lhs.drawCountOffset==rhs.drawCountOffset)
                                                {
                                                    if (lhs.drawCallOffset==rhs.drawCallOffset)
                                                    {
                                                        if (lhs.drawMaxCount==rhs.drawMaxCount)
                                                            return equalRetval;
                                                        return Cmp<uint32_t>()(lhs.drawMaxCount,rhs.drawMaxCount);
                                                    }
                                                    return Cmp<uint32_t>()(lhs.drawCallOffset,rhs.drawCallOffset);
                                                }
                                                return Cmp<uint32_t>()(lhs.drawCountOffset,rhs.drawCountOffset);
                                            }
                                            return Cmp<uint32_t>()(lhs.drawCommandStride,rhs.drawCommandStride);
                                        }
                                        return Cmp<uint32_t>()(lhs.indexType,rhs.indexType);
                                    }
                                    return Cmp<const void*>()(lhs.indexBufferBinding.get(),rhs.indexBufferBinding.get());
                                }
                                return Cmp<const void*>()(lhs.pipeline.get(),rhs.pipeline.get());
                            }
                            return Cmp<size_t>()(lhs_pcranges.size(),rhs_pcranges.size());
                        }
                        return Cmp<uint32_t>()(lhs.pipeline->getCachedCreationParams().subpassIx,rhs.pipeline->getCachedCreationParams().subpassIx);
                    }
                    return Cmp<const void*>()(lhs.pipeline->getRenderpass(),rhs.pipeline->getRenderpass());
                }
        };

        //
        inline auto& getDrawcallMetadataVector()
        {
            m_needsSorting = DefaultOrder::invalidTypeID;
            return m_drawCallMetadataStorage;
        }
        inline const auto& getDrawcallMetadataVector() const {return m_drawCallMetadataStorage;}

        // commandbuffer must have the subpass already begun
        // by setting `drawCountBuffer=nullptr` you disable the use of count buffers
        // (commands are still sorted as if it was used, if you want to ignore draw counts, set `DrawcallInfo::drawCountOffset` on all elements to invalid)
        template<typename draw_call_order_t=DefaultOrder>
        void bake(IGPUCommandBuffer* cmdbuf, const IGPURenderpass* renderpass, const uint32_t subpassIndex, const IGPUBuffer* drawIndirectBuffer, const IGPUBuffer* drawCountBuffer)
        {
            assert(cmdbuf&&renderpass&&subpassIndex<renderpass->getSubpassCount() && drawIndirectBuffer);
            if (m_needsSorting!=draw_call_order_t::typeID)
            {
                std::sort(m_drawCallMetadataStorage.begin(),m_drawCallMetadataStorage.end(), typename draw_call_order_t::less());
                m_needsSorting = draw_call_order_t::typeID;
            }

            const SearchObject searchObj = {renderpass,subpassIndex};
            const auto begin = std::lower_bound(m_drawCallMetadataStorage.begin(),m_drawCallMetadataStorage.end(),searchObj, typename draw_call_order_t::renderpass_subpass_comp());
            const auto end = std::upper_bound(m_drawCallMetadataStorage.begin(),m_drawCallMetadataStorage.end(),searchObj, typename draw_call_order_t::renderpass_subpass_comp());
            if (begin==end)
                return;

            bake_impl(cmdbuf->getOriginDevice()->getPhysicalDevice()->getLimits().indirectDrawCount, drawIndirectBuffer, drawCountBuffer)(cmdbuf, begin, end);
        }

    protected:
        core::vector<DrawcallInfo> m_drawCallMetadataStorage;
        uint64_t m_needsSorting = DefaultOrder::invalidTypeID;

        using call_iterator = typename decltype(m_drawCallMetadataStorage)::const_iterator;
        struct bake_impl
        {
            public:
                bake_impl(const bool _drawCountEnabled, const IGPUBuffer* _drawIndirectBuffer, const IGPUBuffer* _drawCountBuffer)
                    : drawCountEnabled(_drawCountEnabled), drawIndirectBuffer(_drawIndirectBuffer), drawCountBuffer(_drawCountBuffer) {}

                inline void operator()(IGPUCommandBuffer* cmdbuf, const call_iterator begin, const call_iterator end)
                {
                    for (auto it=begin; it!=end;)
                    {
                        if (it->pipeline.get()!=pipeline)
                        {
                            pipeline = it->pipeline.get();
                            cmdbuf->bindGraphicsPipeline(pipeline);
                        }
                        assert(it->pipeline->getCachedCreationParams().subpassIx==pipeline->getCachedCreationParams().subpassIx);
                        for (; it!=end&&it->pipeline.get()==pipeline; it++)
                        {
                            const auto currentLayout = pipeline->getLayout();
                            // repush constants iff dirty
                            bool incompatiblePushConstants = !layout || !layout->isCompatibleForPushConstants(currentLayout);
                            const auto currentPushConstantRange = currentLayout->getPushConstantRanges();
                            for (const auto& rng : currentPushConstantRange)
                            {
                                const uint8_t* src = it->pushConstantData+rng.offset;
                                if (incompatiblePushConstants || memcmp(src,pushConstants+rng.offset,rng.size)!=0)
                                    cmdbuf->pushConstants(currentLayout,rng.stageFlags,rng.offset,rng.size,src);
                            }
                            pushConstants = it->pushConstantData;
                            // rebind descriptor sets iff dirty
                            const auto unmodifiedSetCount = [&]() -> uint32_t
                            {
                                if (incompatiblePushConstants)
                                    return 0u;
                                for (auto i=0u; i<IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
                                if (it->descriptorSets[i].get()!=descriptorSets[i])
                                    return i;
                                return layout->isCompatibleUpToSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT-1,currentLayout)+1;
                            }();
                            const auto nonNullDSEnd = [&]() -> uint32_t
                            {
                                for (auto i=IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i!=unmodifiedSetCount;)
                                if (it->descriptorSets[--i])
                                    return i+1u;
                                return unmodifiedSetCount;
                            }();
                            if (nonNullDSEnd!=unmodifiedSetCount)
                                cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS,currentLayout,unmodifiedSetCount,nonNullDSEnd-unmodifiedSetCount,&it->descriptorSets->get()+unmodifiedSetCount); // TODO: support dynamic offsets later
                            for (auto i=unmodifiedSetCount; i<nonNullDSEnd; i++)
                                descriptorSets[i] = it->descriptorSets[i].get();
                            layout = currentLayout;
                            // change vertex bindings iff dirty
                            const auto unmodifiedBindingCount = [&]() -> uint32_t
                            {
                                for (auto i=0u; i<asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
                                if (it->vertexBufferBindings[i]!=vertexBindings[i])
                                    return i;
                                return asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT;
                            }();
                            const auto nonNullBindingEnd = [&]() -> uint32_t
                            {
                                for (auto i=asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i!=unmodifiedBindingCount;)
                                if (it->vertexBufferBindings[--i].buffer)
                                    return i+1u;
                                return unmodifiedBindingCount;
                            }();
                            const auto newBindingCount = nonNullBindingEnd-unmodifiedBindingCount;
                            std::copy_n(it->vertexBufferBindings+unmodifiedBindingCount,newBindingCount,vertexBindings);
                            if (nonNullBindingEnd!=unmodifiedBindingCount)
                                cmdbuf->bindVertexBuffers(unmodifiedBindingCount,newBindingCount,vertexBindings+unmodifiedBindingCount);
                            // change index bindings iff dirty
                            if (it->indexBufferBinding.get()!=indexBuffer || it->indexType!=indexType)
                            {
                                switch (it->indexType)
                                {
                                    case asset::EIT_16BIT:
                                        [[fallthrough]];
                                    case asset::EIT_32BIT:
                                        indexType = static_cast<asset::E_INDEX_TYPE>(it->indexType);
                                        cmdbuf->bindIndexBuffer({0ull,it->indexBufferBinding},indexType);
                                        break;
                                    default:
                                        cmdbuf->bindIndexBuffer({},asset::EIT_UNKNOWN);
                                        indexType = asset::EIT_UNKNOWN;
                                        break;
                                }
                                indexBuffer = it->indexBufferBinding.get();
                            }
                            // now we're ready to record a few drawcalls
                            const bool indexed = indexType!=asset::EIT_UNKNOWN;
                            asset::SBufferBinding<const IGPUBuffer> indirectBinding = {it->drawCallOffset,drawIndirectBuffer};
                            const uint32_t drawMaxCount=it->drawMaxCount, drawCommandStride=it->drawCommandStride;
                            if (drawCountBuffer && it->drawCountOffset!=IDrawIndirectAllocator::invalid_draw_count_ix)
                            {
                                assert(drawCountEnabled);
                                asset::SBufferBinding<const IGPUBuffer> countBinding = {it->drawCountOffset,drawCountBuffer};
                                if (indexed)
                                    cmdbuf->drawIndexedIndirectCount(indirectBinding,countBinding,drawMaxCount,drawCommandStride);
                                else
                                    cmdbuf->drawIndirectCount(indirectBinding,countBinding,drawMaxCount,drawCommandStride);
                            }
                            else
                            {
                                if (indexed)
                                    cmdbuf->drawIndexedIndirect(indirectBinding,drawMaxCount,drawCommandStride);
                                else
                                    cmdbuf->drawIndirect(indirectBinding,drawMaxCount,drawCommandStride);
                            }
                        }
                    }
                }
            private:
                //
                const bool drawCountEnabled;
                core::smart_refctd_ptr<const IGPUBuffer> drawIndirectBuffer;
                core::smart_refctd_ptr<const IGPUBuffer> drawCountBuffer;
                //
                const IGPUGraphicsPipeline* pipeline = nullptr;
                const uint8_t* pushConstants = nullptr;
                const IGPUDescriptorSet* descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = {nullptr};
                const IGPUPipelineLayout* layout = nullptr;
                asset::SBufferBinding<const IGPUBuffer> vertexBindings[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT] = {};
                asset::E_INDEX_TYPE indexType = asset::EIT_UNKNOWN;
                const IGPUBuffer* indexBuffer = nullptr;
        };
};

}
#endif