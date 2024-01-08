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
			: ICPUShader(core::make_smart_refctd_ptr<ICPUBuffer>(strlen(code) + 1u), stage, contentType, std::move(filepathHint))
		{
			assert(contentType != E_CONTENT_TYPE::ECT_SPIRV); // because using strlen needs `code` to be null-terminated
			memcpy(m_code->getPointer(), code, m_code->getSize());
		}

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_SHADER;
		inline E_TYPE getAssetType() const override { return AssetType; }

		size_t conservativeSizeEstimate() const override 
		{
			size_t estimate = m_code->getSize();
			estimate += getFilepathHint().size();
			return estimate;
		}

		core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto buf = (_depth > 0u && m_code) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_code->clone(_depth-1u)) : m_code;
			auto cp = core::smart_refctd_ptr<ICPUShader>(new ICPUShader(std::move(buf), getStage(), m_contentType, std::string(getFilepathHint())), core::dont_grab);
			clone_common(cp.get());

			return cp;
		}

		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
				m_code->convertToDummyObject(referenceLevelsBelowToConvert-1u);
		}

		const ICPUBuffer* getContent() const { return m_code.get(); };

		inline E_CONTENT_TYPE getContentType() const { return m_contentType; }
		
		inline bool isContentHighLevelLanguage() const
		{
			return (m_contentType == E_CONTENT_TYPE::ECT_GLSL || m_contentType == E_CONTENT_TYPE::ECT_HLSL);
		}

		bool setShaderStage(const E_SHADER_STAGE stage)
		{
			if(isImmutable_debug())
				return m_shaderStage == stage;
			m_shaderStage = stage;
			return true;
		}

		bool setFilePathHint(std::string&& filepathHint)
		{
			if(isImmutable_debug())
				return false;
			m_filepathHint = std::move(filepathHint);
			return true;
		}

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUShader*>(_other);
			if (m_contentType != other->m_contentType)
				return false;
			if (getFilepathHint() != other->getFilepathHint())
				return false;
			if (getStage() != other->getStage())
				return false;
			if (!m_code->canBeRestoredFrom(other->m_code.get()))
				return false;

			return true;
		}

	protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUShader*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;

				restoreFromDummy_impl_call(m_code.get(), other->m_code.get(), _levelsBelow);
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			return m_code->isAnyDependencyDummy(_levelsBelow);
		}

		const core::smart_refctd_ptr<ICPUBuffer> m_code;
		const E_CONTENT_TYPE m_contentType;
};

}
#endif
