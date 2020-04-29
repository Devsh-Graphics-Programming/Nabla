#ifndef _IRR_C_GLSL_VIRTUAL_TEXTURING_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define _IRR_C_GLSL_VIRTUAL_TEXTURING_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/CGraphicsPipelineLoaderMTL.h"

namespace irr {
namespace asset
{

class CGLSLVirtualTexturingBuiltinIncludeLoader : public IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/virtual_texturing/"; }

private:
	static core::vector<std::string> parseArgumentsFromPath(const std::string& _path)
	{
		core::vector<std::string> args;

		std::stringstream ss{ _path };
		std::string arg;
		while (std::getline(ss, arg, '/'))
			args.push_back(std::move(arg));

		return args;
	}
	static std::string getDescriptors(const std::string& _path)
	{
		auto args = parseArgumentsFromPath(_path.substr(_path.find_first_of('/') + 1, _path.npos));
		constexpr uint32_t
			set_ix		= 0u,
			pgt_bnd_ix	= 1u,
			f_bnd_ix	= 2u,
			i_bnd_ix	= 3u,
			u_bnd_ix	= 4u,
			f_cnt_ix	= 5u,
			i_cnt_ix	= 6u,
			u_cnt_ix	= 7u;

		return 
"\n#ifndef _IRR_VT_DESCRIPTOR_SET"
"\n#define _IRR_VT_DESCRIPTOR_SET " + args[set_ix] +
"\n#endif" +
"\n#ifndef _IRR_VT_PAGE_TABLE_BINDING"
"\n#define _IRR_VT_PAGE_TABLE_BINDING " + args[pgt_bnd_ix] +
"\n#endif" +
"\n#ifndef _IRR_VT_FLOAT_VIEWS"
"\n#define _IRR_VT_FLOAT_VIEWS_BINDING " + args[f_bnd_ix] +
"\n#define _IRR_VT_FLOAT_VIEWS_COUNT " + args[f_cnt_ix] +
"\n#endif" +
"\n#ifndef _IRR_VT_INT_VIEWS"
"\n#define _IRR_VT_INT_VIEWS_BINDING " + args[i_bnd_ix] +
"\n#define _IRR_VT_INT_VIEWS_COUNT " + args[i_cnt_ix] +
"\n#endif" +
"\n#ifndef _IRR_VT_UINT_VIEWS"
"\n#define _IRR_VT_UINT_VIEWS_BINDING " + args[u_bnd_ix] +
"\n#define _IRR_VT_UINT_VIEWS_COUNT " + args[u_cnt_ix] +
"\n#endif" +
R"(
layout(set=_IRR_VT_DESCRIPTOR_SET, binding=_IRR_VT_PAGE_TABLE_BINDING) uniform usampler2DArray pageTable;
#if _IRR_VT_FLOAT_VIEWS_COUNT
layout(set=_IRR_VT_DESCRIPTOR_SET, binding=_IRR_VT_FLOAT_VIEWS_BINDING) uniform sampler2DArray physicalTileStorageFormatView[_IRR_VT_FLOAT_VIEWS_COUNT];
#endif
#if _IRR_VT_INT_VIEWS_COUNT
layout(set=_IRR_VT_DESCRIPTOR_SET, binding=_IRR_VT_INT_VIEWS_BINDING) uniform isampler2DArray iphysicalTileStorageFormatView[_IRR_VT_INT_VIEWS_COUNT];
#endif
#if _IRR_VT_UINT_VIEWS_COUNT
layout(set=_IRR_VT_DESCRIPTOR_SET, binding=_IRR_VT_UINT_VIEWS_BINDING) uniform usampler2DArray uphysicalTileStorageFormatView[_IRR_VT_UINT_VIEWS_COUNT];
#endif
)";
	}
	static std::string getExtensions(const std::string&)
	{
		return R"(
#extension GL_EXT_nonuniform_qualifier  : enable

//tmp dirty fix for weird behavior of renderdoc
//uncomment below if having troubles in renderdoc and you're sure GL_NV_gpu_shader5 is available
#define RUNNING_IN_RENDERDOC
#ifdef RUNNING_IN_RENDERDOC
	#define IRR_GL_NV_gpu_shader5
#endif

#if 1
//#ifdef IRR_GL_NV_gpu_shader5
    #define IRR_GL_EXT_nonuniform_qualifier // TODO: we need to overhaul our GLSL preprocessing system to match what SPIRV-Cross actually does
#endif

#ifndef IRR_GL_EXT_nonuniform_qualifier
	#error "SPIR-V Cross did not implement GL_KHR_shader_subgroup_ballot on GLSL yet!"
#endif
)";
	}
    static std::string getVTfunctions(const std::string& _path)
    {
		auto args = parseArgumentsFromPath(_path.substr(_path.find_first_of('/')+1, _path.npos));
		if (args.size()<6u)
			return {};

		constexpr uint32_t
			ix_pg_sz_log2 = 0u,
			ix_tile_padding = 1u,
			ix_get_pgtab_sz_log2_name = 2u,
			ix_get_phys_pg_tex_sz_rcp_name = 3u,
			ix_get_vtex_sz_rcp_name = 4u,
			ix_get_layer2pid_name = 5u;

		const uint32_t pg_sz_log2 = std::atoi(args[ix_pg_sz_log2].c_str());
		const uint32_t tile_padding = std::atoi(args[ix_tile_padding].c_str());

		ICPUVirtualTexture::SMiptailPacker::rect tilePacking[ICPUVirtualTexture::MAX_PHYSICAL_PAGE_SIZE_LOG2];
		//this could be cached..
		ICPUVirtualTexture::SMiptailPacker::computeMiptailOffsets(tilePacking, pg_sz_log2, tile_padding);

		auto tilePackingOffsetsStr = [&] {
			std::string offsets;
			for (uint32_t i = 0u; i < pg_sz_log2; ++i)
				offsets += "vec2(" + std::to_string(tilePacking[i].x) + "," + std::to_string(tilePacking[i].y) + ")" + (i == (pg_sz_log2 - 1u) ? "" : ",");
			return offsets;
		};

		using namespace std::string_literals;
		std::string s;
		s += "\n\n#define PAGE_SZ " + std::to_string(1u<<pg_sz_log2) + "u" +
			"\n#define PAGE_SZ_LOG2 " + args[ix_pg_sz_log2] + "u" +
			"\n#define TILE_PADDING " + args[ix_tile_padding] + "u" +
			"\n#define PADDED_TILE_SIZE uint(PAGE_SZ+2*TILE_PADDING)" +
			"\n\nconst vec2 packingOffsets[] = vec2[PAGE_SZ_LOG2+1]( vec2(0.0,0.0)," + tilePackingOffsetsStr() + ");";
		s+=
R"(
#define irr_glsl_STextureData_WRAP_REPEAT 0u
#define irr_glsl_STextureData_WRAP_CLAMP 1u
#define irr_glsl_STextureData_WRAP_MIRROR 2u

#ifndef _IRR_PHYSICAL_ADDR_SPEC_DEFINED_
#define irr_glsl_ADDR_X_MASK 0xfu
#define irr_glsl_ADDR_Y_MASK 0xfu
#define irr_glsl_ADDR_Y_SHIFT 4u
#define irr_glsl_ADDR_LAYER_SHIFT 8u
#endif //!_IRR_PHYSICAL_ADDR_SPEC_DEFINED_

vec3 irr_glsl_unpackPageID(in uint pageID)
{
	// this is optimal, don't touch
	uvec2 pageXY = uvec2(pageID,pageID>>irr_glsl_ADDR_Y_SHIFT)&uvec2(irr_glsl_ADDR_X_MASK,irr_glsl_ADDR_Y_MASK);
	return vec3(vec2(pageXY),float(pageID>>irr_glsl_ADDR_LAYER_SHIFT));
}
uvec2 irr_glsl_unpackWrapModes(in uvec2 texData)
{
    return (texData>>uvec2(28u,30u)) & uvec2(0x03u);
}
uint irr_glsl_unpackMaxMipInVT(in uvec2 texData)
{
    return (texData.y>>24)&0x0fu;
}
vec3 irr_glsl_unpackVirtualUV(in uvec2 texData)
{
	// assert that PAGE_SZ_LOG2<8 , or change the line to uvec3(texData.yy<<uvec2(PAGE_SZ_LOG2,PAGE_SZ_LOG2-8u),texData.y>>16u)
    uvec3 unnormCoords = uvec3(texData.y<<PAGE_SZ_LOG2,texData.yy>>uvec2(8u-PAGE_SZ_LOG2,16u))&uvec3(uvec2(0xffu)<<PAGE_SZ_LOG2,0xffu);
    return vec3(unnormCoords);
}
vec2 irr_glsl_unpackSize(in uvec2 texData)
{
	return vec2(texData.x&0xffffu,texData.x>>16u);
}
)";
		s += R"(
#ifndef _IRR_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_
  #error "You need to define irr_glsl_VT_getPgTabSzLog2,irr_glsl_VT_getPhysPgTexSzRcp,irr_glsl_VT_getVTexSzRcp,irr_glsl_VT_layer2pid before including this header"
#endif
)";
        s += R"(
vec3 irr_glsl_vTextureGrad_helper(in uint formatID, in vec3 virtualUV, in int clippedLoD, in int levelInTail)
{
    uvec2 pageID = textureLod(pageTable,virtualUV,clippedLoD).xy;

	const uint pageTableSizeLog2 = irr_glsl_VT_getPgTabSzLog2(formatID);
    const float phys_pg_tex_sz_rcp = irr_glsl_VT_getPhysPgTexSzRcp(formatID);
	// assert that pageTableSizeLog2<23

	// this will work because pageTables are always square and PoT and IEEE754
	uint thisLevelTableSize = (pageTableSizeLog2-uint(clippedLoD))<<23;

	vec2 tileCoordinate = uintBitsToFloat(floatBitsToUint(virtualUV.xy)+thisLevelTableSize);
	tileCoordinate = fract(tileCoordinate); // optimize this fract at some point
	tileCoordinate = uintBitsToFloat(floatBitsToUint(tileCoordinate)+uint((PAGE_SZ_LOG2-levelInTail)<<23));
    tileCoordinate += packingOffsets[levelInTail];
	tileCoordinate += vec2(TILE_PADDING,TILE_PADDING);
	tileCoordinate *= phys_pg_tex_sz_rcp;

	vec3 physicalUV = irr_glsl_unpackPageID(levelInTail!=0 ? pageID.y:pageID.x);
	physicalUV.xy *= vec2(PAGE_SZ+2*TILE_PADDING)*phys_pg_tex_sz_rcp;

	// add the in-tile coordinate
	physicalUV.xy += tileCoordinate;
    return physicalUV;
}

float irr_glsl_lengthManhattan(vec2 v)
{
	v = abs(v);
    return v.x+v.y;
}
float irr_glsl_lengthSq(in vec2 v)
{
  return dot(v,v);
}
// textureGrad emulation
vec4 irr_glsl_vTextureGrad(in uint formatID, in vec3 virtualUV, in mat2 dOriginalScaledUV, in int originalMaxFullMip)
{
	// returns what would have been `textureGrad(originalTexture,gOriginalUV[0],gOriginalUV[1])
	// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap15.html#textures-normalized-operations
	const float kMaxAnisotropy = float(2u*TILE_PADDING);
	// you can use an approx `log2` if you know one
#if APPROXIMATE_FOOTPRINT_CALC
	// bounded by sqrt(2)
	float p_x_2_log2 = log2(irr_glsl_lengthManhattan(dOriginalScaledUV[0]));
	float p_y_2_log2 = log2(irr_glsl_lengthManhattan(dOriginalScaledUV[1]));
	const float kMaxAnisoLogOffset = log2(kMaxAnisotropy);
#else
	float p_x_2_log2 = log2(irr_glsl_lengthSq(dOriginalScaledUV[0]));
	float p_y_2_log2 = log2(irr_glsl_lengthSq(dOriginalScaledUV[1]));
	const float kMaxAnisoLogOffset = log2(kMaxAnisotropy)*2.0;
#endif
	bool xIsMajor = p_x_2_log2>p_y_2_log2;
	float p_min_2_log2 = xIsMajor ? p_y_2_log2:p_x_2_log2;
	float p_max_2_log2 = xIsMajor ? p_x_2_log2:p_y_2_log2;

	float LoD = max(p_min_2_log2,p_max_2_log2-kMaxAnisoLogOffset);
#if APPROXIMATE_FOOTPRINT_CALC
	LoD += 0.5;
#else
	LoD *= 0.5;
#endif
	// WARNING: LoD_high will round up when LoD negative, its not a floor
	int LoD_high = int(LoD);

	// are we performing minification
	bool positiveLoD = LoD>0.0;
	// magnification samples LoD 0, else clip to max representable in VT
	int clippedLoD = positiveLoD ? min(LoD_high,originalMaxFullMip):0;

	// if minification is being performaed then get tail position
	int levelInTail = LoD_high-clippedLoD;
	// have to do trilinear only if doing minification AND larger than 1x1 footprint
	bool haveToDoTrilinear = levelInTail<int(PAGE_SZ_LOG2) && positiveLoD;
	levelInTail = haveToDoTrilinear ? levelInTail:(positiveLoD ? int(PAGE_SZ_LOG2):0);

	// get the higher resolution mip-map level
	vec3 hiPhysCoord = irr_glsl_vTextureGrad_helper(formatID,virtualUV,clippedLoD,levelInTail);
	// get lower if needed (speculative execution, had to move divergent indexing to a single place)
    vec3 loPhysCoord;
	// speculative if (haveToDoTrilinear)
	{
		// now we have absolute guarantees that both LoD_high and LoD_low are in the valid original mip range
		bool highNotInLastFull = LoD_high<originalMaxFullMip;
		clippedLoD = highNotInLastFull ? (clippedLoD+1):clippedLoD;
		levelInTail = highNotInLastFull ? levelInTail:(levelInTail+1);
		loPhysCoord = irr_glsl_vTextureGrad_helper(formatID,virtualUV,clippedLoD,levelInTail);
	}

	vec4 hiMip_retval;
    vec4 loMip;
#ifdef IRR_GL_EXT_nonuniform_qualifier
    hiMip_retval = textureGrad(physicalTileStorageFormatView[nonuniformEXT(formatID)],hiPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
    if (haveToDoTrilinear)
        loMip = textureGrad(physicalTileStorageFormatView[nonuniformEXT(formatID)],loPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
#else
    uint64_t outstandingSampleMask = ballotARB(true);
    // maybe unroll a few times manually
    while (outstandingSampleMask!=uint64_t(0u))
    {
		uvec2 tmp = unpackUint2x32(outstandingSampleMask);
        uint subgroupFormatID = readInvocationARB(formatID,tmp[1]!=0u ? 32u:findLSB(tmp[0]));
        bool canSample = subgroupFormatID==formatID; // do I need this? && (outstandingSampleMask&gl_SubGroupEqMaskARB)==gl_SubGroupEqMaskARB;
        outstandingSampleMask ^= ballotARB(canSample);
        if (canSample)
        {
            hiMip_retval = textureGrad(physicalTileStorageFormatView[subgroupFormatID],hiPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
            if (haveToDoTrilinear)
                loMip = textureGrad(physicalTileStorageFormatView[subgroupFormatID],loPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
        }
    }
#endif
    if (haveToDoTrilinear)
	    hiMip_retval = mix(hiMip_retval,loMip,LoD-float(LoD_high));
    return hiMip_retval;
}

float irr_glsl_wrapTexCoord(float tc, in uint mode)
{
    switch (mode)
    {
    case irr_glsl_STextureData_WRAP_REPEAT: tc = fract(tc); break;
    case irr_glsl_STextureData_WRAP_CLAMP:  tc = clamp(tc, 0.0, 1.0); break;
    case irr_glsl_STextureData_WRAP_MIRROR: tc = 1.0 - abs(mod(tc,2.0)-1.0); break;
    default: break;
    }
    return tc;
}
vec4 irr_glsl_textureVT(in uvec2 _texData, in vec2 uv, in mat2 dUV)
{
    vec2 originalSz = irr_glsl_unpackSize(_texData);
	dUV[0] *= originalSz;
	dUV[1] *= originalSz;

    uvec2 wrap = irr_glsl_unpackWrapModes(_texData);
    uv.x = irr_glsl_wrapTexCoord(uv.x,wrap.x);
    uv.y = irr_glsl_wrapTexCoord(uv.y,wrap.y);

	vec3 virtualUV = irr_glsl_unpackVirtualUV(_texData);

    uint formatID = irr_glsl_VT_layer2pid(uint(virtualUV.z));

    virtualUV.xy += uv*originalSz;
    virtualUV.xy *= irr_glsl_VT_getVTexSzRcp(formatID);

    return irr_glsl_vTextureGrad(formatID, virtualUV, dUV, int(irr_glsl_unpackMaxMipInVT(_texData)));
}
)";
		return s;
    }
protected:
	core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
	{
		const std::string id = "[a-zA-Z_][a-zA-Z0-9_]*";
		const std::string num = "[0-9]+";
		return {
			//functions.glsl/pg_sz_log2/tile_padding/get_pgtab_sz_log2_name/get_phys_pg_tex_sz_rcp_name/get_vtex_sz_rcp_name/get_layer2pid
			{ 
				std::regex{"functions\\.glsl/"+num+"/"+num+"/"+id+"/"+id+"/"+id+"/"+id},
				&getVTfunctions
			},
			{std::regex{"extensions\\.glsl"}, &getExtensions},
			//descriptors.glsl/set/pgt_bnd/f_bnd/i_bnd/u_bnd/f_cnt/i_cnt/u_cnt
			{std::regex{"descriptors\\.glsl/"+num+"/"+num+"/"+num+"/"+num+"/"+num+"/"+num+"/"+num+"/"+num}, &getDescriptors}
		};
	}
};

}}
#endif