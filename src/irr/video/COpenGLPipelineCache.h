#ifndef __IRR_C_OPENGL_PIPELINE_CACHE_H_INCLUDED__
#define __IRR_C_OPENGL_PIPELINE_CACHE_H_INCLUDED__

#include "irr/video/IGPUPipelineCache.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLSpecializedShader.h"
#include "irr/video/COpenGLPipelineLayout.h"
#include "irr/core/Types.h"
#include "spirv_cross/spirv_parser.hpp"
#include <array>

namespace irr { namespace video
{

class COpenGLPipelineCache final : public IGPUPipelineCache
{
public:
	struct SCacheVal {
		COpenGLSpecializedShader::SProgramBinary binary;
		core::smart_refctd_ptr<COpenGLPipelineLayout> layout;
		core::smart_refctd_dynamic_array<GLint> locations;
	};
	struct SCacheKey {
		std::array<uint64_t, 4> hash;
		COpenGLSpecializedShader::SInfo info;
		bool operator<(const SCacheKey& _rhs) const
		{
			if (hash < _rhs.hash) return true;
			if (_rhs.hash < hash) return false;
			return info < _rhs.info;
		}
	};

	void merge(uint32_t _count, const IGPUPipelineCache** _srcCaches) override
	{
		const std::lock_guard<std::mutex> _(m_mutex);

		for (uint32_t i = 0u; i < _count; ++i)
		{
			const auto& src = static_cast<const COpenGLPipelineCache*>(_srcCaches[i])->m_cache;
			m_cache.insert(src.begin(), src.end());
		}
	}

	core::smart_refctd_ptr<asset::ICPUPipelineCache> convertToCPUCache() const override;

	COpenGLSpecializedShader::SProgramBinary find(const SCacheKey& _key, const COpenGLPipelineLayout* _layout) const
	{
		const std::lock_guard<std::mutex> _(m_mutex);

		auto rng = m_cache.equal_range(_key);
		for (auto it = rng.first; it != rng.second; ++it)
			if (_layout->isCompatibleUpToSet(COpenGLPipelineLayout::DESCRIPTOR_SET_COUNT-1u, it->second.layout.get())==(COpenGLPipelineLayout::DESCRIPTOR_SET_COUNT-1u))
				return it->second.binary;
		return {0,nullptr};
	}
	//assumes that m_cache does not already contain an item with key==_key and layout fully compatible with _val.layout
	void insert(SCacheKey&& _key, SCacheVal&& _val, const asset::ICPUBuffer* _spirv)
	{
		const std::lock_guard<std::mutex> _(m_mutex);
		/*
		auto it = m_parsedSpirvs.find(_key.hash);
		if (it == m_parsedSpirvs.end())
		{
			spirv_cross::Parser parser(reinterpret_cast<const uint32_t*>(_spirv->getPointer()), _spirv->getSize()/4ull);
			auto parsed = parser.get_parsed_ir();

			m_parsedSpirvs.insert(it, {_key.hash,std::move(parsed)});
		}
		*/
#ifdef _IRR_DEBUG
		assert(!find(_key, _val.layout.get()).binary);
#endif

		m_cache.insert({std::move(_key),std::move(_val)});
	}

private:
	core::multimap<SCacheKey, SCacheVal> m_cache;
	mutable std::mutex m_mutex;
	//core::map<std::array<uint64_t, 4>, spirv_cross::ParsedIR> m_parsedSpirvs;
};

}}

#endif