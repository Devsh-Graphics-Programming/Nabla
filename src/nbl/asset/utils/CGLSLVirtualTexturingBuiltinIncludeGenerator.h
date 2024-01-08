// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GLSL_VIRTUAL_TEXTURING_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_GLSL_VIRTUAL_TEXTURING_BUILTIN_INCLUDE_LOADER_H_INCLUDED__


#include "nbl/asset/utils/ICPUVirtualTexture.h"
#include "nbl/asset/utils/IShaderCompiler.h"


namespace nbl
{
namespace asset
{

class CGLSLVirtualTexturingBuiltinIncludeGenerator : public IShaderCompiler::IIncludeGenerator
{
	public:
		using Base = IShaderCompiler::IIncludeGenerator;
		using Base::Base;

		std::string_view getPrefix() const override { return "nbl/builtin/glsl/virtual_texturing"; };

	private:
		static std::string getVTfunctions(const std::string& _path)
		{
			auto args = parseArgumentsFromPath(_path.substr(_path.rfind(".glsl")+6, _path.npos));
			if (args.size()<2u)
				return {};

			constexpr uint32_t
				ix_pg_sz_log2 = 0u,
				ix_tile_padding = 1u;

			const uint32_t pg_sz_log2 = std::atoi(args[ix_pg_sz_log2].c_str());
			const uint32_t tile_padding = std::atoi(args[ix_tile_padding].c_str());

			ICPUVirtualTexture::SMiptailPacker::rect tilePacking[ICPUVirtualTexture::MAX_PHYSICAL_PAGE_SIZE_LOG2];
			//this could be cached..
			ICPUVirtualTexture::SMiptailPacker::computeMiptailOffsets(tilePacking, pg_sz_log2, tile_padding);

			auto tilePackingOffsetsStr = [&] {
				std::string offsets;
				for (uint32_t i = 0u; i < pg_sz_log2; ++i)
					offsets += "vec2(" + std::to_string(tilePacking[i].x+tile_padding) + "," + std::to_string(tilePacking[i].y+tile_padding) + ")" + (i == (pg_sz_log2 - 1u) ? "" : ",");
				return offsets;
			};

			using namespace std::string_literals;
			std::string s = R"(
#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_TEXTURING_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_TEXTURING_FUNCTIONS_INCLUDED_
)";
			s += "\n\n#define _NBL_VT_IMPL_PAGE_SZ " + std::to_string(1u<<pg_sz_log2) + "u" +
				"\n#define _NBL_VT_IMPL_PAGE_SZ_LOG2 " + args[ix_pg_sz_log2] + "u" +
				"\n#define _NBL_VT_IMPL_TILE_PADDING " + args[ix_tile_padding] + "u" +
				"\n#define _NBL_VT_IMPL_PADDED_TILE_SIZE uint(_NBL_VT_IMPL_PAGE_SZ+2*_NBL_VT_IMPL_TILE_PADDING)" +
				"\n\nconst vec2 packingOffsets[] = vec2[_NBL_VT_IMPL_PAGE_SZ_LOG2+1]( vec2(" + std::to_string(tile_padding) + ")," + tilePackingOffsetsStr() + ");";
			s += R"(
#include "nbl/builtin/glsl/virtual_texturing/impl_functions.glsl"

#endif
)";
			return s;
		}

	protected:
		core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
		{
			core::vector<std::pair<std::regex, HandleFunc_t>> retval;

			const std::string num = "[0-9]+";
			retval.insert(retval.begin(),
				{ 
					std::regex{"glsl/virtual_texturing/functions\\.glsl/"+num+"/"+num},
					&getVTfunctions
				}
			);
			return retval;
		}
};

}}
#endif