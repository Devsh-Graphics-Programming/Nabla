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
    static std::string getVTfunctions(const std::string& _path)
    {
		auto args = parseArgumentsFromPath(_path.substr(_path.find_first_of('/')+1, _path.npos));

		using namespace std::string_literals;
        std::string s = "#include <irr/builtin/glsl/texture_packer/utils.glsl";
		for (uint32_t i = 0u; i < 4u; ++i)
			s += "/" + args[i];
        s += ">";
		s +=
			"\n#define irr_glsl_VT_pgtabName "s + args[4] +
			"\n#define irr_glsl_VT_physPgTexName " + args[5] +
			"\n#define irr_glsl_VT_getPgTabSzLog2 " + args[6] +
			"\n#define irr_glsl_VT_getPhysPgTexSzRcp " + args[7] +
			"\n#define irr_glsl_VT_getVTexSzRcp " + args[8] +
			"\n#define irr_glsl_VT_layer2pid " + args[9]
			;
        s += R"(

#extension GL_EXT_nonuniform_qualifier  : enable

#ifdef IRR_GL_NV_gpu_shader5
    #define IRR_GL_EXT_nonuniform_qualifier // TODO: we need to overhaul our GLSL preprocessing system to match what SPIRV-Cross actually does
#endif

#ifndef IRR_GL_EXT_nonuniform_qualifier
    #extension GL_ARB_gpu_shader_int64      : require
    #extension GL_ARB_shader_ballot         : require
#endif

vec3 vTextureGrad_helper(in uint formatID, in vec3 virtualUV, in int clippedLoD, in int levelInTail)
{
    uvec2 pageID = textureLod(irr_glsl_VT_pgtabName,virtualUV,clippedLoD).xy;

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

	vec3 physicalUV = unpackPageID(levelInTail!=0 ? pageID.y:pageID.x);
	physicalUV.xy *= vec2(PAGE_SZ+2*TILE_PADDING)*phys_pg_tex_sz_rcp;

	// add the in-tile coordinate
	physicalUV.xy += tileCoordinate;
    return physicalUV;
}

float lengthManhattan(vec2 v)
{
	v = abs(v);
    return v.x+v.y;
}
float lengthSq(in vec2 v)
{
  return dot(v,v);
}
// textureGrad emulation
vec4 vTextureGrad(in uint formatID, in vec3 virtualUV, in mat2 dOriginalScaledUV, in int originalMaxFullMip)
{
	// returns what would have been `textureGrad(originalTexture,gOriginalUV[0],gOriginalUV[1])
	// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap15.html#textures-normalized-operations
	const float kMaxAnisotropy = float(2u*TILE_PADDING);
	// you can use an approx `log2` if you know one
#if APPROXIMATE_FOOTPRINT_CALC
	// bounded by sqrt(2)
	float p_x_2_log2 = log2(lengthManhattan(dOriginalScaledUV[0]));
	float p_y_2_log2 = log2(lengthManhattan(dOriginalScaledUV[1]));
	const float kMaxAnisoLogOffset = log2(kMaxAnisotropy);
#else
	float p_x_2_log2 = log2(lengthSq(dOriginalScaledUV[0]));
	float p_y_2_log2 = log2(lengthSq(dOriginalScaledUV[1]));
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
	vec3 hiPhysCoord = vTextureGrad_helper(formatID,virtualUV,clippedLoD,levelInTail);
	// get lower if needed (speculative execution, had to move divergent indexing to a single place)
    vec3 loPhysCoord;
	// speculative if (haveToDoTrilinear)
	{
		// now we have absolute guarantees that both LoD_high and LoD_low are in the valid original mip range
		bool highNotInLastFull = LoD_high<originalMaxFullMip;
		clippedLoD = highNotInLastFull ? (clippedLoD+1):clippedLoD;
		levelInTail = highNotInLastFull ? levelInTail:(levelInTail+1);
		loPhysCoord = vTextureGrad_helper(formatID,virtualUV,clippedLoD,levelInTail);
	}

	vec4 hiMip_retval;
    vec4 loMip;
#ifdef IRR_GL_EXT_nonuniform_qualifier
    hiMip_retval = textureGrad(irr_glsl_VT_physPgTexName[nonuniformEXT(formatID)],hiPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
    if (haveToDoTrilinear)
        loMip = textureGrad(irr_glsl_VT_physPgTexName[nonuniformEXT(formatID)],loPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
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
            hiMip_retval = textureGrad(irr_glsl_VT_physPgTexName[subgroupFormatID],hiPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
            if (haveToDoTrilinear)
                loMip = textureGrad(irr_glsl_VT_physPgTexName[subgroupFormatID],loPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
        }
    }
#endif
    if (haveToDoTrilinear)
	    hiMip_retval = mix(hiMip_retval,loMip,LoD-float(LoD_high));
    return hiMip_retval;
}

float wrapTexCoord(float tc, in uint mode)
{
    switch (mode)
    {
    case irr_glsl_WRAP_REPEAT: tc = fract(tc); break;
    case irr_glsl_WRAP_CLAMP:  tc = clamp(tc, 0.0, 1.0); break;
    case irr_glsl_WRAP_MIRROR: tc = 1.0 - abs(mod(tc,2.0)-1.0); break;
    default: break;
    }
    return tc;
}
vec4 textureVT(in uvec2 _texData, in vec2 uv, in mat2 dUV)
{
    vec2 originalSz = unpackSize(_texData);
	dUV[0] *= originalSz;
	dUV[1] *= originalSz;

    uvec2 wrap = unpackWrapModes(_texData);
    uv.x = wrapTexCoord(uv.x,wrap.x);
    uv.y = wrapTexCoord(uv.y,wrap.y);

	vec3 virtualUV = unpackVirtualUV(_texData);

    uint formatID = irr_glsl_VT_layer2pid(uint(virtualUV.z));

    virtualUV.xy += uv*originalSz;
    virtualUV.xy *= irr_glsl_VT_getVTexSzRcp(formatID);

    return vTextureGrad(formatID, virtualUV, dUV, int(unpackMaxMipInVT(_texData)));
}
)";
		return s;
    }
protected:
	core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
	{
		const std::string id = "[a-zA-Z_][a-zA-Z0-9_]*";
		return {
			//functions.glsl/addr_x_bits/addr_y_bits/pg_sz_log2/tile_padding/pgtab_tex_name/phys_pg_tex_name/get_pgtab_sz_log2_name/get_phys_pg_tex_sz_rcp_name/get_vtex_sz_rcp_name/get_layer2pid
			{ 
				std::regex{"functions\\.glsl/[0-9]+/[0-9]+/[0-9]+/[0-9]+/"+id+"/"+id+"/"+id+"/"+id+"/"+id+"/"+id},
				&getVTfunctions
			},
		};
	}
};

}}
#endif