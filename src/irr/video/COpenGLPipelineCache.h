#ifndef __IRR_C_OPENGL_PIPELINE_CACHE_H_INCLUDED__
#define __IRR_C_OPENGL_PIPELINE_CACHE_H_INCLUDED__

#include "irr/video/IGPUPipelineCache.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLSpecializedShader.h"
#include "irr/video/COpenGLPipelineLayout.h"
#include "irr/core/Types.h"
#include "spirv_cross/spirv_parser.hpp"
#include "CConcurrentObjectCache.h"
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
		size_t sz = 0ull;
		for (uint32_t i = 0u; i < _count; ++i)
			sz = std::max(sz, static_cast<const COpenGLPipelineCache*>(_srcCaches[i])->m_cache.getSize());

		auto buf = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<decltype(m_cache)::MutablePairType>>(sz);
		for (uint32_t i = 0u; i < _count; ++i)
		{
			const auto& src = static_cast<const COpenGLPipelineCache*>(_srcCaches[i])->m_cache;
			src.outputAll(sz, buf->data());
			for (size_t j = 0ull; j < sz; ++j)
				m_cache.insert((*buf)[j].first, (*buf)[j].second);
		}
	}

	COpenGLSpecializedShader::SProgramBinary find(const SCacheKey& _key, const COpenGLPipelineLayout* _layout) const
	{
		if (m_cache.getSize()==0ull)
			return {0,nullptr};

		size_t sz = m_cache.getSize();
		auto buf = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SCacheVal>>(sz);
		m_cache.findAndStoreRange(_key, sz, buf->data());
		for (size_t i = 0ull; i < sz; ++i)
			if (_layout->isCompatibleUpToSet(COpenGLPipelineLayout::DESCRIPTOR_SET_COUNT-1u, (*buf)[i].layout.get())==(COpenGLPipelineLayout::DESCRIPTOR_SET_COUNT-1u))
				return (*buf)[i].binary;
		return {0,nullptr};
	}
	//assumes that m_cache does not already contain an item with key==_key and layout fully compatible with _val.layout
	void insert(SCacheKey&& _key, SCacheVal&& _val, const asset::ICPUBuffer* _spirv)
	{
		/*
		auto it = m_parsedSpirvs.find(_key.hash);
		if (it == m_parsedSpirvs.end())
		{
			spirv_cross::Parser parser(reinterpret_cast<const uint32_t*>(_spirv->getPointer()), _spirv->getSize()/4ull);
			auto parsed = parser.get_parsed_ir();

			m_parsedSpirvs.insert(it, {_key.hash,std::move(parsed)});
		}
		*/
		m_cache.insert(std::move(_key), std::move(_val));
	}

private:
	//TODO make it thread-safe using CConcurrentObjectCache
	//core::multimap<SCacheKey, SCacheVal> m_cache;
	core::CConcurrentMultiObjectCache<SCacheKey, SCacheVal> m_cache;
	//core::map<std::array<uint64_t, 4>, spirv_cross::ParsedIR> m_parsedSpirvs;
};

}}

#endif