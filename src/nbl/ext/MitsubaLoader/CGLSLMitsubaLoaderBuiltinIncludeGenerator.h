// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_GLSL_MITSUBA_LOADER_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __C_GLSL_MITSUBA_LOADER_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "nbl/asset/utils/IShaderCompiler.h"


namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

class CGLSLMitsubaLoaderBuiltinIncludeGenerator : public IShaderCompiler::IIncludeGenerator
{
    public:
		using Base = IShaderCompiler::IIncludeGenerator;
		using Base::Base;

		std::string_view getPrefix() const override { return "nbl/builtin/glsl/ext/MitsubaLoader"; };

    private:
		static std::string getMaterialCompilerStuff(const std::string& _path)
		{
			auto args = parseArgumentsFromPath(_path.substr(_path.rfind(".glsl")+6, _path.npos));

			const auto str = "#define _NBL_EXT_MITSUBA_LOADER_VT_STORAGE_VIEW_COUNT " + args.front() + "\n";

			return str +
				"#include \"nbl/builtin/glsl/ext/MitsubaLoader/material_compiler_compatibility_impl.glsl\"\n";
		}

	protected:
		core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
		{
			core::vector<std::pair<std::regex, HandleFunc_t>> retval;

			const std::string num = "[0-9]+";
			retval.insert(retval.begin(),
				{ 
					std::regex{"glsl/ext/MitsubaLoader/material_compiler_compatibility\\.glsl/"+num},
					&getMaterialCompilerStuff
				}
			);
			return retval;
		}
};

}}}


#endif