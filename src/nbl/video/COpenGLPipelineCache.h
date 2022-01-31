// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_PIPELINE_CACHE_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_PIPELINE_CACHE_H_INCLUDED__

#include "nbl/video/IGPUPipelineCache.h"
#include "nbl/video/COpenGLSpecializedShader.h"
#include "nbl/video/COpenGLPipelineLayout.h"
#include "nbl/core/decl/Types.h"
#include "nbl_spirv_cross/spirv_parser.hpp"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include <array>

namespace nbl { namespace video
{

class COpenGLPipelineCache final : public IGPUPipelineCache
{
public:
	struct SCacheVal {
		COpenGLSpecializedShader::SProgramBinary binary;
	};
	struct SCacheKey {
		std::array<uint64_t, 4> hash;
		COpenGLSpecializedShader::SInfo info;
		core::smart_refctd_ptr<COpenGLPipelineLayout> layout;
		asset::IShader::E_SHADER_STAGE shaderStage = asset::IShader::ESS_UNKNOWN;

		bool operator<(const SCacheKey& _rhs) const;
	};

	using IGPUPipelineCache::IGPUPipelineCache;

	void merge(uint32_t _count, const IGPUPipelineCache** _srcCaches) override
	{
		const std::lock_guard<std::mutex> _1_(m_bin_cache_mutex);
		const std::lock_guard<std::mutex> _2_(m_parsed_cache_mutex);

		for (uint32_t i = 0u; i < _count; ++i)
		{
			{
				const auto& src = static_cast<const COpenGLPipelineCache*>(_srcCaches[i])->m_cache;
				m_cache.insert(src.begin(), src.end());
			}
			{
				const auto& src = static_cast<const COpenGLPipelineCache*>(_srcCaches[i])->m_parsedSpirvs;
				m_parsedSpirvs.insert(src.begin(), src.end());
			}
		}
	}

	core::smart_refctd_ptr<asset::ICPUPipelineCache> convertToCPUCache(IOpenGL_FunctionTable* gl) const;

	COpenGLSpecializedShader::SProgramBinary find(const SCacheKey& _key) const
	{
		const std::lock_guard<std::mutex> _(m_bin_cache_mutex);

		auto found = m_cache.find(_key);
		if (found!=m_cache.end())
			return found->second.binary;
		return {0,nullptr};
	}
	const spirv_cross::ParsedIR* findParsedSpirv(const std::array<uint64_t, 4>& _key)
	{
		const std::lock_guard<std::mutex> _(m_parsed_cache_mutex);

		auto found = m_parsedSpirvs.find(_key);
		if (found!=m_parsedSpirvs.end())
			return &found->second;

		return nullptr;
	}

	//assumes that m_cache does not already contain an item with key==_key and layout fully compatible with _val.layout
	void insert(SCacheKey&& _key, SCacheVal&& _val)
	{
		const std::lock_guard<std::mutex> _(m_bin_cache_mutex);
#ifdef _NBL_DEBUG
		assert(!find(_key).binary);
#endif

		m_cache.insert({std::move(_key),std::move(_val)});
	}
	void insertParsedSpirv(const std::array<uint64_t, 4>& _key, const asset::ICPUBuffer* _spirv)
	{
		const std::lock_guard<std::mutex> _(m_parsed_cache_mutex);

		auto found = m_parsedSpirvs.find(_key);
		if (found==m_parsedSpirvs.end())
		{
			spirv_cross::Parser parser(reinterpret_cast<const uint32_t*>(_spirv->getPointer()), _spirv->getSize()/4ull);
			parser.parse();
			spirv_cross::ParsedIR& parsed = parser.get_parsed_ir();

			m_parsedSpirvs.insert({_key, std::move(parsed)});
		}
	}

private:
	core::map<SCacheKey, SCacheVal> m_cache;
	mutable std::mutex m_bin_cache_mutex;
	core::map<std::array<uint64_t, 4>, spirv_cross::ParsedIR> m_parsedSpirvs;
	mutable std::mutex m_parsed_cache_mutex;
};

}}

#endif