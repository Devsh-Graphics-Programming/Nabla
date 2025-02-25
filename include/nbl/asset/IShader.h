// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_SHADER_H_INCLUDED_
#define _NBL_ASSET_I_SHADER_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/builtin/hlsl/enums.hlsl"

#include <algorithm>
#include <string>


namespace spirv_cross
{
	class ParsedIR;
	class Compiler;
	struct SPIRType;
}

namespace nbl::asset
{

//! Interface class for Unspecialized Shaders
/*
	The purpose for the class is for storing raw HLSL code to be compiled
	or already compiled (but unspecialized) SPIR-V code.
*/
class IShader : public IAsset
{
	public:
		enum class E_CONTENT_TYPE : uint8_t
		{
			ECT_UNKNOWN = 0,
			ECT_GLSL,
			ECT_HLSL,
			ECT_SPIRV,
		};
		//
		inline IShader(core::smart_refctd_ptr<ICPUBuffer>&& code, const E_CONTENT_TYPE contentType, std::string&& filepathHint) :
			m_filepathHint(std::move(filepathHint)), m_code(std::move(code)), m_contentType(contentType) {}
		inline IShader(const char* code, const E_CONTENT_TYPE contentType, std::string&& filepathHint) :
			m_filepathHint(std::move(filepathHint)), m_code(ICPUBuffer::create({strlen(code)+1u})), m_contentType(contentType)
		{
			assert(contentType!=E_CONTENT_TYPE::ECT_SPIRV); // because using strlen needs `code` to be null-terminated
			memcpy(m_code->getPointer(),code,m_code->getSize());
		}
		// forwarding
		template<typename CodeT>
		inline IShader(CodeT&& code, const E_CONTENT_TYPE contentType, const std::string_view& filepathHint) :
			IShader(std::forward(code),contentType,std::string(filepathHint)) {}
		
		//
		constexpr static inline auto AssetType = ET_SHADER;
		inline E_TYPE getAssetType() const override { return AssetType; }
		
		//
		inline size_t getDependantCount() const override { return 1; }
		
		//
		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth=~0u) const override
		{
			auto buf = (_depth>0u && m_code) ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_code->clone(_depth-1u)):m_code;
			return core::make_smart_refctd_ptr<IShader>(std::move(buf),m_contentType,std::string(m_filepathHint));
		}


		// The file path hint is extemely important for resolving includes if the content type is NOT SPIR-V
		inline const std::string& getFilepathHint() const { return m_filepathHint; }
		bool setFilePathHint(std::string&& filepathHint)
		{
			if(!isMutable())
				return false;
			m_filepathHint = std::move(filepathHint);
			return true;
		}

		//
		const ICPUBuffer* getContent() const { return m_code.get(); };
		
		//
		inline E_CONTENT_TYPE getContentType() const { return m_contentType; }
		inline bool isContentHighLevelLanguage() const
		{
			switch (m_contentType)
			{
				case E_CONTENT_TYPE::ECT_SPIRV:
					return false;
				default:
					break;
			}
			return true;
		}

		// TODO: `void setContent(core::smart_refctd_ptr<const ICPUBuffer>&&,const E_CONTENT_TYPE)`


		// alias for legacy reasons
		using E_SHADER_STAGE = hlsl::ShaderStage;

	protected:
		virtual ~IShader() = default;

		inline IAsset* getDependant_impl(const size_t ix) override {return m_code.get();}

		std::string m_filepathHint;
		core::smart_refctd_ptr<ICPUBuffer> m_code;
		E_CONTENT_TYPE m_contentType;
};
}

#endif
