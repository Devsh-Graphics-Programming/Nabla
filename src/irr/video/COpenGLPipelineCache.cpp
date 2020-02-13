#include "irr/video/COpenGLPipelineCache.h"

using namespace irr;
using namespace video;

core::smart_refctd_ptr<asset::ICPUPipelineCache> COpenGLPipelineCache::convertToCPUCache() const
{
	asset::ICPUPipelineCache::entries_map_t entries;

	size_t sz = m_cache.getSize();
	auto buf = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<decltype(m_cache)::MutablePairType>>(sz);
	m_cache.outputAll(sz, buf->data());

	std::string uuid;
	{
		uuid = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
		uuid += reinterpret_cast<const char*>(glGetString(GL_RENDERER));

		uuid += std::to_string(COpenGLExtensionHandler::Version);

		std::string exts;
		for (uint32_t k = 0u; k < COpenGLExtensionHandler::IRR_OpenGL_Feature_Count; ++k)
		{
			if (COpenGLExtensionHandler::FeatureAvailable[k])
				exts += OpenGLFeatureStrings[k];
		}
		uuid += exts;
	}

	for (size_t i = 0ull; i < sz; ++i)
	{
		uint32_t bndCnt = 0u;
		uint32_t bndPerSet[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
		for (uint32_t j = 0u; j < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++j)
		{
			auto dsl = (*buf)[i].second.layout->getDescriptorSetLayout(j);
			bndPerSet[j] = dsl ? dsl->getBindings().length() : 0u;
			bndCnt += bndPerSet[j];
		}

		uint32_t scCnt = (*buf)[i].first.info.m_entries->size();

		auto meta_buf = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint8_t>>(asset::ICPUPipelineCache::SGLKeyMeta::calcMetaSize(bndCnt, scCnt));
		asset::ICPUPipelineCache::SGLKeyMeta* meta = reinterpret_cast<asset::ICPUPipelineCache::SGLKeyMeta*>(meta_buf->data());
		memcpy(meta->bindingsPerSet, bndPerSet, sizeof(meta->bindingsPerSet));
		memcpy(meta->spirvHash, (*buf)[i].first.hash.data(), sizeof(meta->spirvHash));

		for (uint32_t j = 0u, k = 0u; j < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++j)
		{
			auto* dsl = (*buf)[i].second.layout->getDescriptorSetLayout(j);
			if (!dsl)
				continue;
				
			for (const auto& b : dsl->getBindings())
			{
				meta->bindings[k].binding = b.binding;
				meta->bindings[k].count = b.count;
				meta->bindings[k].stageFlags = b.stageFlags;
				meta->bindings[k].type = b.type;

				++k;
			}
			
		}
		auto* scEntries = reinterpret_cast<asset::ICPUPipelineCache::SGLKeyMeta::SSpecInfo::SEntry*>(meta_buf->data()+meta_buf->size()-sizeof(asset::ICPUPipelineCache::SGLKeyMeta::SSpecInfo::SEntry)*scCnt);
		{
			uint32_t k = 0u;
			for (const auto& e : *((*buf)[i].first.info.m_entries))
			{
				scEntries[k].id = e.specConstID;
				assert(e.size==4u);
				memcpy(&scEntries[k].value, reinterpret_cast<uint8_t*>((*buf)[i].first.info.m_backingBuffer->getPointer())+e.offset, 4u);
			}
		}

		asset::ICPUPipelineCache::SCacheKey cpukey;
		cpukey.gpuid.backend = asset::ICPUPipelineCache::EB_OPENGL;
		cpukey.gpuid.UUID = uuid;
		cpukey.meta = std::move(meta_buf);
		asset::ICPUPipelineCache::SCacheVal cpuval;
		cpuval.extra = (*buf)[i].second.binary.format;
		cpuval.bin = (*buf)[i].second.binary.binary;

		entries.insert({std::move(cpukey), std::move(cpuval)});
	}

	return core::make_smart_refctd_ptr<asset::ICPUPipelineCache>(std::move(entries));
}