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
	static std::string getDescriptors(const std::string&)
	{
		return
R"(
#ifndef _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_DESCRIPTORS_INCLUDED_
#define _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_DESCRIPTORS_INCLUDED_

#ifndef _IRR_VT_DESCRIPTOR_SET
#define _IRR_VT_DESCRIPTOR_SET 0
#endif
#ifndef _IRR_VT_PAGE_TABLE_BINDING
#define _IRR_VT_PAGE_TABLE_BINDING 0
#endif
#ifndef _IRR_VT_FLOAT_VIEWS
#define _IRR_VT_FLOAT_VIEWS_BINDING 1 
#define _IRR_VT_FLOAT_VIEWS_COUNT 15
#endif
#ifndef _IRR_VT_INT_VIEWS
#define _IRR_VT_INT_VIEWS_BINDING 2
#define _IRR_VT_INT_VIEWS_COUNT 15
#endif
#ifndef _IRR_VT_UINT_VIEWS
#define _IRR_VT_UINT_VIEWS_BINDING 3
#define _IRR_VT_UINT_VIEWS_COUNT 15
#endif

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

#endif
)";
	}
	static std::string getExtensions(const std::string&)
	{
		return R"(
#ifndef _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_EXTENSIONS_INCLUDED_
#define _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_EXTENSIONS_INCLUDED_

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

#endif
)";
	}
    static std::string getVTfunctions(const std::string& _path)
    {
		auto args = parseArgumentsFromPath(_path.substr(_path.find_first_of('/')+1, _path.npos));
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
#ifndef _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_FUNCTIONS_INCLUDED_
#define _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_FUNCTIONS_INCLUDED_
)";
		s += "\n\n#define PAGE_SZ " + std::to_string(1u<<pg_sz_log2) + "u" +
			"\n#define PAGE_SZ_LOG2 " + args[ix_pg_sz_log2] + "u" +
			"\n#define TILE_PADDING " + args[ix_tile_padding] + "u" +
			"\n#define PADDED_TILE_SIZE uint(PAGE_SZ+2*TILE_PADDING)" +
			"\n\nconst vec2 packingOffsets[] = vec2[PAGE_SZ_LOG2+1]( vec2(" + std::to_string(tile_padding) + ")," + tilePackingOffsetsStr() + ");";
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
  #error "You need to define irr_glsl_VT_getPgTabSzLog2(),irr_glsl_VT_getPhysPgTexSzRcp(uint formatID),irr_glsl_VT_getVTexSzRcp(),irr_glsl_VT_layer2pid(uint layer) before including this header"
#endif
)";
        s += R"(
vec3 irr_glsl_vTexture_helper(in uint formatID, in vec3 virtualUV, in int clippedLoD, in int levelInTail)
{
    uvec2 pageID = textureLod(pageTable,virtualUV,clippedLoD).xy;

	const uint pageTableSizeLog2 = irr_glsl_VT_getPgTabSzLog2();
    const float phys_pg_tex_sz_rcp = irr_glsl_VT_getPhysPgTexSzRcp(formatID);
	// assert that pageTableSizeLog2<23

	// this will work because pageTables are always square and PoT and IEEE754
	uint thisLevelTableSize = (pageTableSizeLog2-uint(clippedLoD))<<23;

	vec2 tileCoordinate = uintBitsToFloat(floatBitsToUint(virtualUV.xy)+thisLevelTableSize);
	tileCoordinate = fract(tileCoordinate); // optimize this fract at some point
	tileCoordinate = uintBitsToFloat(floatBitsToUint(tileCoordinate)+uint((PAGE_SZ_LOG2-levelInTail)<<23));
    tileCoordinate += packingOffsets[levelInTail];
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
vec4 irr_glsl_vTextureGrad_impl(in uint formatID, in vec3 virtualUV, in mat2 dOriginalScaledUV, in int originalMaxFullMip)
{
	// returns what would have been `textureGrad(originalTexture,gOriginalUV[0],gOriginalUV[1])
	// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap15.html#textures-normalized-operations
	const float kMaxAnisotropy = float(2u*TILE_PADDING);
	// you can use an approx `log2` if you know one
#if _IRR_APPROXIMATE_FOOTPRINT_CALC_
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
#if _IRR_APPROXIMATE_FOOTPRINT_CALC_
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
	vec3 hiPhysCoord = irr_glsl_vTexture_helper(formatID,virtualUV,clippedLoD,levelInTail);
	// get lower if needed (speculative execution, had to move divergent indexing to a single place)
    vec3 loPhysCoord;
	// speculative if (haveToDoTrilinear)
	{
		// now we have absolute guarantees that both LoD_high and LoD_low are in the valid original mip range
		bool highNotInLastFull = LoD_high<originalMaxFullMip;
		clippedLoD = highNotInLastFull ? (clippedLoD+1):clippedLoD;
		levelInTail = highNotInLastFull ? levelInTail:(levelInTail+1);
		loPhysCoord = irr_glsl_vTexture_helper(formatID,virtualUV,clippedLoD,levelInTail);
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

vec4 irr_glsl_vTextureGrad(in uvec2 _texData, in vec2 uv, in mat2 dUV)
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
    virtualUV.xy *= irr_glsl_VT_getVTexSzRcp();

    return irr_glsl_vTextureGrad_impl(formatID, virtualUV, dUV, int(irr_glsl_unpackMaxMipInVT(_texData)));
}
//apparently glslang doesnt support line continuation characters (backslash) :c
/*
#ifdef IRR_GL_EXT_nonuniform_qualifier
	#define _IRR_DIVERGENT_SAMPLING_IMPL(retval_t, physicalSamplerName) return textureLod(physicalSamplerName[nonuniformEXT(formatID)], physCoord, lod)
#else
	#define _IRR_DIVERGENT_SAMPLING_IMPL(retval_t, physicalSamplerName) \
    retval_t retval;\
    uint64_t outstandingSampleMask = ballotARB(true);\
    while (outstandingSampleMask != uint64_t(0u))\ 
    {\
        uvec2 tmp = unpackUint2x32(outstandingSampleMask);\
        uint subgroupFormatID = readInvocationARB(formatID, tmp[1] != 0u ? 32u : findLSB(tmp[0]));\
        bool canSample = subgroupFormatID == formatID;\
        outstandingSampleMask ^= ballotARB(canSample);\
        if (canSample)\
            retval = textureLod(physicalSamplerName[subgroupFormatID], physCoord, lod);\
    }\
    return retval
#endif

#define _IRR_DEFINE_VT_INTEGER_FUNCTIONS(funcName, implFuncName, retval_t, physicalSamplerName)\
retval_t implFuncName(in uint formatID, in vec3 virtualUV, in uint lod, in int originalMaxFullMip)\
{\
    int nonnegativeLod = int(lod);\
    int clippedLoD = min(nonnegativeLod,originalMaxFullMip);\
    int levelInTail = nonnegativeLod - clippedLoD;\
    \
    vec3 physCoord = irr_glsl_vTexture_helper(formatID, virtualUV, clippedLoD, levelInTail);\
	_IRR_DIVERGENT_SAMPLING_IMPL(retval_t, physicalSamplerName);\
}\
retval_t funcName(in uvec2 _texData, in vec2 uv, in uint lod)\
{\
    vec2 originalSz = irr_glsl_unpackSize(_texData);\

    uvec2 wrap = irr_glsl_unpackWrapModes(_texData);\
    uv.x = irr_glsl_wrapTexCoord(uv.x, wrap.x);\
    uv.y = irr_glsl_wrapTexCoord(uv.y, wrap.y);\
    \
    vec3 virtualUV = irr_glsl_unpackVirtualUV(_texData);\
    uint formatID = irr_glsl_VT_layer2pid(uint(virtualUV.z));\
    virtualUV.xy += uv * originalSz;\
    virtualUV.xy *= irr_glsl_VT_getVTexSzRcp();\
	\
    return irr_glsl_vTextureLod_impl(formatID, virtualUV, lod, int(irr_glsl_unpackMaxMipInVT(_texData)));\
}

_IRR_DEFINE_VT_INTEGER_FUNCTIONS(irr_glsl_iVTextureLod, irr_glsl_iVTextureLod_impl, ivec4, iphysicalTileStorageFormatView)
_IRR_DEFINE_VT_INTEGER_FUNCTIONS(irr_glsl_uVTextureLod, irr_glsl_uVTextureLod_impl, uvec4, uphysicalTileStorageFormatView)
#undef _IRR_DIVERGENT_SAMPLING_IMPL
*/

#endif //!_IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_FUNCTIONS_INCLUDED_
)";
		return s;
    }
protected:
	core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
	{
		const std::string num = "[0-9]+";
		return {
			//functions.glsl/pg_sz_log2/tile_padding
			{ 
				std::regex{"functions\\.glsl/"+num+"/"+num},
				&getVTfunctions
			},
			{std::regex{"extensions\\.glsl"}, &getExtensions},
			{std::regex{"descriptors\\.glsl"}, &getDescriptors}
		};
	}
};

}}
#endif