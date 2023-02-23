// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_SHADER_H_INCLUDED_
#define _NBL_ASSET_I_SHADER_H_INCLUDED_

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

class IShader : public virtual core::IReferenceCounted
{
	public:
		// TODO: make this enum class
		enum E_SHADER_STAGE : uint32_t
		{
			ESS_UNKNOWN = 0,
			ESS_VERTEX = 1 << 0,
			ESS_TESSELLATION_CONTROL = 1 << 1,
			ESS_TESSELLATION_EVALUATION = 1 << 2,
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
			ESS_ALL_GRAPHICS = 0x0000001F,
			ESS_ALL = 0x7fffffff
		};

		enum class E_CONTENT_TYPE : uint8_t
		{
			ECT_UNKNOWN = 0,
			ECT_GLSL,
			ECT_HLSL,
			ECT_SPIRV,
		};

		IShader(const E_SHADER_STAGE shaderStage, std::string&& filepathHint)
			: m_shaderStage(shaderStage), m_filepathHint(std::move(filepathHint))
		{}

		inline E_SHADER_STAGE getStage() const { return m_shaderStage; }

		inline const std::string& getFilepathHint() const { return m_filepathHint; }

protected:
	E_SHADER_STAGE m_shaderStage;
	std::string m_filepathHint;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IShader::E_SHADER_STAGE)

}

#endif
