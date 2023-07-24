// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_SHADER_H_INCLUDED__
#define __NBL_ASSET_I_CPU_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>


#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/IShader.h"

namespace nbl
{
namespace asset
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

		ICPUShader(core::smart_refctd_ptr<ICPUBuffer>&& code, const E_SHADER_STAGE stage, E_CONTENT_TYPE contentType, std::string&& filepathHint)
			: IShader(stage, std::move(filepathHint)), m_code(std::move(code))
			, m_contentType(contentType)
		{}

		ICPUShader(
			const char* code,
			const E_SHADER_STAGE stage,
			E_CONTENT_TYPE contentType,
			std::string&& filepathHint) 
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

		void convertToDummyObject_impl(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			convertToDummyObject_common(referenceLevelsBelowToConvert);
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


	protected:
		bool compatible(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUShader*>(_other);
			if (m_contentType != other->m_contentType)
				return false;
			if (getFilepathHint() != other->getFilepathHint())
				return false;
			if (getStage() != other->getStage())
				return false;
			return true;
		}

		nbl::core::vector<core::smart_refctd_ptr<IAsset>> getMembersToRecurse() const override { return { m_code }; }

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			return m_code->isAnyDependencyDummy(_levelsBelow);
		}

		core::smart_refctd_ptr<ICPUBuffer> m_code;
		E_CONTENT_TYPE m_contentType;
};

}
}

#endif
