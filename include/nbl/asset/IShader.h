// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_SHADER_H_INCLUDED__
#define __NBL_ASSET_I_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>


#include "nbl/core/declarations.h"

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
	The purpose for the class is for storing raw GLSL code
	to be compiled or already compiled (but unspecialized) 
	SPIR-V code. Such a shader has to be passed
	to Specialized Shader constructor.
*/

class NBL_API IShader : public virtual core::IReferenceCounted
{
	public:
		enum E_SHADER_STAGE : uint32_t
		{
			ESS_VERTEX = 1 << 0,
			ESS_TESSELATION_CONTROL = 1 << 1,
			ESS_TESSELATION_EVALUATION = 1 << 2,
			ESS_GEOMETRY = 1 << 3,
			ESS_FRAGMENT = 1 << 4,
			ESS_COMPUTE = 1 << 5,
			ESS_TASK = 1 << 6,
			ESS_MESH = 1 << 7,
			ESS_RAYGEN = 1 << 8,
			ESS_ANY_HIT = 1 << 9,
			ESS_CLOSEST_HIT = 1 << 10,
			ESS_MISS = 1 << 11,
			ESS_INTERSECTION = 1 << 12,
			ESS_CALLABLE = 1 << 13,
			ESS_UNKNOWN = 0,
			ESS_ALL_GRAPHICS = 0x1f,
			ESS_ALL = 0xffffffff
		};

		struct buffer_contains_glsl_t {};
		_NBL_STATIC_INLINE const buffer_contains_glsl_t buffer_contains_glsl = {};

		IShader(const E_SHADER_STAGE shaderStage, std::string&& filepathHint)
			: m_shaderStage(shaderStage), m_filepathHint(std::move(filepathHint))
		{}

		inline E_SHADER_STAGE getStage() const { return m_shaderStage; }

		inline const std::string& getFilepathHint() const { return m_filepathHint; }

		static inline void insertDefines(std::string& _glsl, const core::SRange<const char* const>& _defines)
		{
			if (_defines.empty())
				return;

			std::ostringstream insertion;
			for (auto def : _defines)
			{
				insertion << "#define "<<def<<"\n";
			}
			insertAfterVersionAndPragmaShaderStage(_glsl,std::move(insertion));
		}

	// TODO: can make it protected again AFTER we get rid of `COpenGLShader::k_openGL2VulkanExtensionMap`
	//protected:
		static inline void insertAfterVersionAndPragmaShaderStage(std::string& _glsl, std::ostringstream&& _ins)
		{
			auto findLineJustAfterVersionOrPragmaShaderStageDirective = [&_glsl] () -> size_t
			{
				size_t hashPos = _glsl.find_first_of('#');
				if (hashPos >= _glsl.length())
					return _glsl.npos;
				if (_glsl.compare(hashPos, 8, "#version"))
					return _glsl.npos;

				size_t searchPos = hashPos + 8ull;

				size_t hashPos2 = _glsl.find_first_of('#', hashPos + 8ull);
				if (hashPos2 < _glsl.length())
				{
					char pragma_stage_str[] = "#pragma shader_stage";
					if (_glsl.compare(hashPos2, sizeof(pragma_stage_str) - 1ull, pragma_stage_str) == 0)
						searchPos = hashPos2 + sizeof(pragma_stage_str) - 1ull;
				}
				size_t nlPos = _glsl.find_first_of('\n', searchPos);

				return (nlPos >= _glsl.length()) ? _glsl.npos : nlPos + 1ull;
			};

			const size_t pos = findLineJustAfterVersionOrPragmaShaderStageDirective();
			if (pos == _glsl.npos)
				return;

			const size_t ln = std::count(_glsl.begin(), _glsl.begin() + pos, '\n') + 1;//+1 to count from 1

			_ins << "#line "<<std::to_string(ln)<<"\n";
			_glsl.insert(pos,_ins.str());
		}

protected:
	E_SHADER_STAGE m_shaderStage;
	std::string m_filepathHint;
};

}

#endif
