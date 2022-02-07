// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/COpenGLPipelineCache.h"

using namespace nbl;
using namespace video;

static int compare_desc_layouts(const IGPUDescriptorSetLayout* A, const IGPUDescriptorSetLayout* B)
{
    // a null descriptor set layout is the smallest
    if(!A)
        return B ? -1 : 0;
    if(!B)
        return A ? 1 : 0;

    // non null descriptor set layouts, we can compare more
    auto count = A->getBindings().size();
    auto lendiff = int64_t(count) - int64_t(B->getBindings().size());
    if(lendiff != 0)
        return lendiff;

    const auto* lhs = A->getBindings().begin();
    const auto* rhs = B->getBindings().begin();
    for(decltype(count) i = 0; i < count; ++i)
    {
        const auto& l = lhs[i];
        const auto& r = rhs[i];
        if(l.binding == r.binding)
        {
            if(l.type == r.type)
            {
                if(l.count == r.count)
                {
                    if(l.stageFlags == r.stageFlags)
                    {
                        if(!l.samplers && r.samplers)
                        {
                            return -1;
                        }
                        if(l.samplers && !r.samplers)
                        {
                            return 1;
                        }
                        for(uint32_t s = 0u; s < l.count; ++s)
                            if(l.samplers[s] != r.samplers[s])
                            {
                                return l.samplers[s].get() - r.samplers[s].get();
                            }
                        continue;  //dont let it just return ( static_cast<int32_t>(l.stageFlags)-static_cast<int32_t>(r.stageFlags) )
                    }
                    return static_cast<int32_t>(l.stageFlags) - static_cast<int32_t>(r.stageFlags);
                }
                return static_cast<int32_t>(l.count) - static_cast<int32_t>(r.count);
            }
            return static_cast<int32_t>(l.type) - static_cast<int32_t>(r.type);
        }
        return static_cast<int32_t>(l.binding) - static_cast<int32_t>(r.binding);
    }

    return 0;
}

bool COpenGLPipelineCache::SCacheKey::operator<(const COpenGLPipelineCache::SCacheKey& _rhs) const
{
    if(hash < _rhs.hash)
        return true;
    if(_rhs.hash < hash)
        return false;
    if(info < _rhs.info)
        return true;
    if(_rhs.info < info)
        return false;
    for(uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        int cmp = compare_desc_layouts(layout->getDescriptorSetLayout(i), _rhs.layout->getDescriptorSetLayout(i));
        if(cmp != 0)
            return cmp < 0;
    }
    return true;
}

core::smart_refctd_ptr<asset::ICPUPipelineCache> COpenGLPipelineCache::convertToCPUCache() const
{
    asset::ICPUPipelineCache::entries_map_t out_entries;

    std::string uuid;
    {
        uuid = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        uuid += reinterpret_cast<const char*>(glGetString(GL_RENDERER));

        uuid += std::to_string(COpenGLExtensionHandler::Version);

        std::string exts;
        for(uint32_t k = 0u; k < COpenGLExtensionHandler::NBL_OpenGL_Feature_Count; ++k)
        {
            if(COpenGLExtensionHandler::FeatureAvailable[k])
                exts += OpenGLFeatureStrings[k];
        }
        uuid += exts;
    }

    const std::lock_guard<std::mutex> _(m_bin_cache_mutex);

    for(const auto& in_entry : m_cache)
    {
        uint32_t bndCnt = 0u;
        uint32_t bndPerSet[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for(uint32_t j = 0u; j < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++j)
        {
            auto dsl = in_entry.first.layout->getDescriptorSetLayout(j);
            bndPerSet[j] = dsl ? dsl->getBindings().size() : 0u;
            bndCnt += bndPerSet[j];
        }

        uint32_t scCnt = in_entry.first.info.m_entries ? in_entry.first.info.m_entries->size() : 0ull;

        auto meta_buf = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint8_t>>(asset::ICPUPipelineCache::SGLKeyMeta::calcMetaSize(bndCnt, scCnt));
        asset::ICPUPipelineCache::SGLKeyMeta* meta = reinterpret_cast<asset::ICPUPipelineCache::SGLKeyMeta*>(meta_buf->data());
        memcpy(meta->bindingsPerSet, bndPerSet, sizeof(meta->bindingsPerSet));
        memcpy(meta->spirvHash, in_entry.first.hash.data(), sizeof(meta->spirvHash));

        for(uint32_t j = 0u, k = 0u; j < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++j)
        {
            auto* dsl = in_entry.first.layout->getDescriptorSetLayout(j);
            if(!dsl)
                continue;

            for(const auto& b : dsl->getBindings())
            {
                meta->bindings[k].binding = b.binding;
                meta->bindings[k].count = b.count;
                meta->bindings[k].stageFlags = b.stageFlags;
                meta->bindings[k].type = b.type;

                ++k;
            }
        }
        auto* scEntries = reinterpret_cast<asset::ICPUPipelineCache::SGLKeyMeta::SSpecInfo::SEntry*>(meta_buf->data() + meta_buf->size() - sizeof(asset::ICPUPipelineCache::SGLKeyMeta::SSpecInfo::SEntry) * scCnt);
        {
            uint32_t k = 0u;
            auto* entryList = in_entry.first.info.m_entries.get();
            if(entryList)
                for(const auto& e : *entryList)
                {
                    scEntries[k].id = e.specConstID;
                    assert(e.size == 4u);
                    memcpy(&scEntries[k].value, reinterpret_cast<const uint8_t*>(in_entry.first.info.m_backingBuffer->getPointer()) + e.offset, 4u);
                }
        }

        asset::ICPUPipelineCache::SCacheKey cpukey;
        cpukey.gpuid.backend = asset::ICPUPipelineCache::EB_OPENGL;
        cpukey.gpuid.UUID = uuid;
        cpukey.meta = std::move(meta_buf);
        asset::ICPUPipelineCache::SCacheVal cpuval;
        cpuval.extra = in_entry.second.binary.format;
        cpuval.bin = in_entry.second.binary.binary;

        out_entries.insert({std::move(cpukey), std::move(cpuval)});
    }

    return core::make_smart_refctd_ptr<asset::ICPUPipelineCache>(std::move(out_entries));
}