// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_SUBPASS_KILN_H_INCLUDED_
#define _NBL_VIDEO_C_SUBPASS_KILN_H_INCLUDED_


#include "nbl/video/IGPUMeshBuffer.h"
#include "nbl/video/utilities/IDrawIndirectAllocator.h"

#include <functional>


namespace nbl::video
{
    
class CSubpassKiln
{
    public:
        // for finding upper and lower bounds of subpass drawcalls
        struct SearchObject
        {
            const IGPURenderpass* renderpass;
            uint32_t subpassIndex;
        };
        //
        struct DrawcallInfo
        {
            alignas(16) uint8_t pushConstantData[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE]; // could try to push it to 64, if we had containers capable of such allocations
            core::smart_refctd_ptr<const IGPUGraphicsPipeline> pipeline;
            core::smart_refctd_ptr<const IGPUDescriptorSet> descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = {};
            asset::SBufferBinding<IGPUBuffer> vertexBufferBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
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
                            return lhs.subpassIndex<rhs.pipeline->getSubpassIndex();
                        return lhs.renderpass<renderpass;
                    }
                    inline bool operator()(const DrawcallInfo& lhs, const SearchObject& rhs)
                    {
                        const auto renderpass = lhs.pipeline->getRenderpass();
                        if (lhs.pipeline->getRenderpass()==rhs.renderpass)
                            return lhs.pipeline->getSubpassIndex()<rhs.subpassIndex;
                        return renderpass<rhs.renderpass;
                    }
                };

            private:
                template<bool equalRetval, template<class> class Cmp>
                static inline bool chainComparator(const DrawcallInfo& lhs, const DrawcallInfo& rhs) // why isnt something like this in the STL? (apparently std::tie could be used here?)
                {
                    if (lhs.pipeline->getRenderpass()==rhs.pipeline->getRenderpass())
                    {
                        if (lhs.pipeline->getSubpassIndex()==rhs.pipeline->getSubpassIndex())
                        {
                            auto lhs_indep = lhs.pipeline->getRenderpassIndependentPipeline();
                            auto rhs_indep = rhs.pipeline->getRenderpassIndependentPipeline();
                            auto lhs_layout = lhs_indep->getLayout();
                            auto rhs_layout = rhs_indep->getLayout();
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
                                    for (auto i=0u; i<IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
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
                        return Cmp<uint32_t>()(lhs.pipeline->getSubpassIndex(),rhs.pipeline->getSubpassIndex());
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
            assert(cmdbuf&&renderpass&&subpassIndex<renderpass->getSubpasses().size()&&drawIndirectBuffer);
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

            const auto& features = cmdbuf->getOriginDevice()->getPhysicalDevice()->getFeatures();
            const bool drawCountEnabled = features.drawIndirectCount;

            if (features.multiDrawIndirect)
                bake_impl<true>(drawCountEnabled,drawIndirectBuffer,drawCountBuffer)(cmdbuf,begin,end);
            else
                bake_impl<false>(drawCountEnabled,drawIndirectBuffer,drawCountBuffer)(cmdbuf,begin,end);
        }

    protected:
        core::vector<DrawcallInfo> m_drawCallMetadataStorage;
        uint64_t m_needsSorting = DefaultOrder::invalidTypeID;

        using call_iterator = typename decltype(m_drawCallMetadataStorage)::const_iterator;
        template<bool multiDrawEnabled>
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
                        assert(it->pipeline->getRenderpassIndependentPipeline()==pipeline->getRenderpassIndependentPipeline());
                        assert(it->pipeline->getSubpassIndex()==pipeline->getSubpassIndex());
                        for (; it!=end&&it->pipeline.get()==pipeline; it++)
                        {
                            const auto currentLayout = pipeline->getRenderpassIndependentPipeline()->getLayout();
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
                                for (auto i=0u; i<IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
                                if (it->vertexBufferBindings[i].buffer.get()!=vertexBindingBuffers[i] || it->vertexBufferBindings[i].offset!=vertexBindingOffsets[i])
                                    return i;
                                return IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
                            }();
                            const auto nonNullBindingEnd = [&]() -> uint32_t
                            {
                                for (auto i=IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i!=unmodifiedBindingCount;)
                                if (it->vertexBufferBindings[--i].buffer)
                                    return i+1u;
                                return unmodifiedBindingCount;
                            }();
                            for (auto i=unmodifiedBindingCount; i<nonNullBindingEnd; i++)
                            {
                                vertexBindingBuffers[i] = it->vertexBufferBindings[i].buffer.get();
                                vertexBindingOffsets[i] = it->vertexBufferBindings[i].offset;
                            }
                            if (nonNullBindingEnd!=unmodifiedBindingCount)
                                cmdbuf->bindVertexBuffers(unmodifiedBindingCount,nonNullBindingEnd-unmodifiedBindingCount,vertexBindingBuffers+unmodifiedBindingCount,vertexBindingOffsets+unmodifiedBindingCount);
                            // change index bindings iff dirty
                            if (it->indexBufferBinding.get()!=indexBuffer || it->indexType!=indexType)
                            {
                                switch (it->indexType)
                                {
                                    case asset::EIT_16BIT:
                                        [[fallthrough]];
                                    case asset::EIT_32BIT:
                                        indexType = static_cast<asset::E_INDEX_TYPE>(it->indexType);
                                        cmdbuf->bindIndexBuffer(it->indexBufferBinding.get(),0ull,indexType);
                                        break;
                                    default:
                                        cmdbuf->bindIndexBuffer(nullptr,0ull,asset::EIT_UNKNOWN);
                                        indexType = asset::EIT_UNKNOWN;
                                        break;
                                }
                                indexBuffer = it->indexBufferBinding.get();
                            }
                            // now we're ready to record a few drawcalls
                            const bool indexed = indexType!=asset::EIT_UNKNOWN;
                            const uint32_t drawCallOffset=it->drawCallOffset, drawCountOffset=it->drawCountOffset, drawMaxCount=it->drawMaxCount, drawCommandStride=it->drawCommandStride;
                            if (drawCountBuffer && drawCountOffset!=IDrawIndirectAllocator::invalid_draw_count_ix)
                            {
                                assert(drawCountEnabled && multiDrawEnabled);
                                if (indexed)
                                    cmdbuf->drawIndexedIndirectCount(drawIndirectBuffer,drawCallOffset,drawCountBuffer,drawCountOffset,drawMaxCount,drawCommandStride);
                                else
                                    cmdbuf->drawIndirectCount(drawIndirectBuffer,drawCallOffset,drawCountBuffer,drawCountOffset,drawMaxCount,drawCommandStride);
                            }
                            else
                            {
                                if (indexed)
                                {
                                    if constexpr (multiDrawEnabled)
                                        cmdbuf->drawIndexedIndirect(drawIndirectBuffer,drawCallOffset,drawMaxCount,drawCommandStride);
                                    else
                                    for (auto i=0u; i<drawMaxCount; i++)
                                        cmdbuf->drawIndexedIndirect(drawIndirectBuffer,i*drawCommandStride+drawCallOffset,1u,sizeof(asset::DrawElementsIndirectCommand_t));
                                }
                                else
                                {
                                    if constexpr (multiDrawEnabled)
                                        cmdbuf->drawIndirect(drawIndirectBuffer,drawCallOffset,drawMaxCount,drawCommandStride);
                                    else
                                    for (auto i=0u; i<drawMaxCount; i++)
                                        cmdbuf->drawIndirect(drawIndirectBuffer,i*drawCommandStride+drawCallOffset,1u,sizeof(asset::DrawArraysIndirectCommand_t));
                                }
                            }
                        }
                    }
                }
            private:
                //
                const bool drawCountEnabled;
                const IGPUBuffer* drawIndirectBuffer;
                const IGPUBuffer* drawCountBuffer;
                //
                const IGPUGraphicsPipeline* pipeline = nullptr;
                const uint8_t* pushConstants = nullptr;
                const IGPUDescriptorSet* descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = {nullptr};
                const IGPUPipelineLayout* layout = nullptr;
                const IGPUBuffer* vertexBindingBuffers[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {nullptr};
                size_t vertexBindingOffsets[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {0ull};
                asset::E_INDEX_TYPE indexType = asset::EIT_UNKNOWN;
                const IGPUBuffer* indexBuffer = nullptr;
        };
};

}

#endif