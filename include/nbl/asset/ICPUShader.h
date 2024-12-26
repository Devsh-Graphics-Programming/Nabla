// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_SHADER_H_INCLUDED_
#define _NBL_ASSET_I_CPU_SHADER_H_INCLUDED_

#include <algorithm>
#include <string>


#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/IShader.h"

namespace nbl::asset
{

//! CPU Version of Unspecialized Shader
/*
	@see IShader
	@see IAsset
*/

class ICPUShader : public IAsset, public IShader
{
	protected:
		virtual ~ICPUShader() = default;

	public:
		using SSpecInfo = IShader::SSpecInfo<ICPUShader>;

		ICPUShader(core::smart_refctd_ptr<ICPUBuffer>&& code, const E_SHADER_STAGE stage, E_CONTENT_TYPE contentType, std::string&& filepathHint)
			: IShader(stage, std::move(filepathHint)), m_code(std::move(code)), m_contentType(contentType) {}

		ICPUShader(const char* code, const E_SHADER_STAGE stage, const E_CONTENT_TYPE contentType, std::string&& filepathHint)
			: ICPUShader(ICPUBuffer::create({ strlen(code) + 1u }), stage, contentType, std::move(filepathHint))
		{
			assert(contentType != E_CONTENT_TYPE::ECT_SPIRV); // because using strlen needs `code` to be null-terminated
			memcpy(m_code->getPointer(), code, m_code->getSize());
		}

		constexpr static inline auto AssetType = ET_SHADER;
		inline E_TYPE getAssetType() const override { return AssetType; }

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto buf = (_depth > 0u && m_code) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_code->clone(_depth-1u)) : m_code;
			return core::smart_refctd_ptr<ICPUShader>(new ICPUShader(std::move(buf), getStage(), m_contentType, std::string(getFilepathHint())), core::dont_grab);
		}

		//!
		inline size_t getDependantCount() const override {return 1;}

		const ICPUBuffer* getContent() const { return m_code.get(); };

		inline E_CONTENT_TYPE getContentType() const { return m_contentType; }
		
		inline bool isContentHighLevelLanguage() const
		{
			return (m_contentType == E_CONTENT_TYPE::ECT_GLSL || m_contentType == E_CONTENT_TYPE::ECT_HLSL);
		}

		bool setShaderStage(const E_SHADER_STAGE stage)
		{
			if(!isMutable())
				return m_shaderStage == stage;
			m_shaderStage = stage;
			return true;
		}

		bool setFilePathHint(std::string&& filepathHint)
		{
			if(!isMutable())
				return false;
			m_filepathHint = std::move(filepathHint);
			return true;
		}

	protected:
		inline IAsset* getDependant_impl(const size_t ix) override {return m_code.get();}

		const core::smart_refctd_ptr<ICPUBuffer> m_code;
		const E_CONTENT_TYPE m_contentType;
};

}
#endif
