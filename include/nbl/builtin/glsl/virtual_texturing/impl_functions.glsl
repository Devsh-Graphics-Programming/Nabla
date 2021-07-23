// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_TEXTURING_IMPL_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_TEXTURING_IMPL_FUNCTIONS_INCLUDED_

#define nbl_glsl_STextureData_WRAP_REPEAT 0u
#define nbl_glsl_STextureData_WRAP_CLAMP 1u
#define nbl_glsl_STextureData_WRAP_MIRROR 2u

#ifndef _NBL_PHYSICAL_ADDR_SPEC_DEFINED_
#define nbl_glsl_ADDR_X_MASK 0xfu
#define nbl_glsl_ADDR_Y_MASK 0xfu
#define nbl_glsl_ADDR_Y_SHIFT 4u
#define nbl_glsl_ADDR_LAYER_SHIFT 8u
#endif //!_NBL_PHYSICAL_ADDR_SPEC_DEFINED_

vec3 nbl_glsl_unpackPageID(in uint pageID)
{
    // this is optimal, don't touch
    uvec2 pageXY = uvec2(pageID, pageID >> nbl_glsl_ADDR_Y_SHIFT)& uvec2(nbl_glsl_ADDR_X_MASK, nbl_glsl_ADDR_Y_MASK);
    return vec3(vec2(pageXY), float(pageID >> nbl_glsl_ADDR_LAYER_SHIFT));
}
uvec2 nbl_glsl_unpackWrapModes(in uvec2 texData)
{
    return (texData >> uvec2(28u, 30u))& uvec2(0x03u);
}
uint nbl_glsl_unpackMaxMipInVT(in uvec2 texData)
{
    return bitfieldExtract(texData.y,24,4);
}
vec3 nbl_glsl_unpackVirtualUV(in uvec2 texData)
{
    // assert that _NBL_VT_IMPL_PAGE_SZ_LOG2<8 , or change the line to uvec3(texData.yy<<uvec2(_NBL_VT_IMPL_PAGE_SZ_LOG2,_NBL_VT_IMPL_PAGE_SZ_LOG2-8u),texData.y>>16u)
    uvec3 unnormCoords = uvec3(texData.y<<_NBL_VT_IMPL_PAGE_SZ_LOG2, texData.yy >> uvec2(8u-_NBL_VT_IMPL_PAGE_SZ_LOG2, 16u))& uvec3(uvec2(0xffu)<<_NBL_VT_IMPL_PAGE_SZ_LOG2, 0xffu);
    return vec3(unnormCoords);
}
vec2 nbl_glsl_unpackSize(in uvec2 texData)
{
    return vec2(texData.x & 0xffffu, texData.x >> 16u);
}

float nbl_glsl_wrapTexCoord(float tc, in uint mode)
{
    switch (mode)
    {
        case nbl_glsl_STextureData_WRAP_REPEAT: tc = fract(tc); break;
        case nbl_glsl_STextureData_WRAP_CLAMP:  tc = clamp(tc, 0.0, 1.0); break;
        case nbl_glsl_STextureData_WRAP_MIRROR: tc = 1.0 - abs(mod(tc, 2.0) - 1.0); break;
        default: break;
    }
    return tc;
}

#ifndef _NBL_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_
  #error "You need to define nbl_glsl_VT_getPgTabSzLog2(),nbl_glsl_VT_getPhysPgTexSzRcp(uint layer),nbl_glsl_VT_getVTexSzRcp(),nbl_glsl_VT_layer2pid(uint layer) before including this header"
#endif

vec3 nbl_glsl_vTexture_helper(in uint formatID, in vec3 virtualUV, in int clippedLoD, in int levelInTail)
{
    uvec2 pageID = textureLod(pageTable,virtualUV,clippedLoD).xy;

	const uint pageTableSizeLog2 = nbl_glsl_VT_getPgTabSzLog2();
    const float phys_pg_tex_sz_rcp = nbl_glsl_VT_getPhysPgTexSzRcp(uint(virtualUV.z));
	// assert that pageTableSizeLog2<23

	// this will work because pageTables are always square and PoT and IEEE754
	uint thisLevelTableSize = (pageTableSizeLog2-uint(clippedLoD))<<23;

	vec2 tileCoordinate = uintBitsToFloat(floatBitsToUint(virtualUV.xy)+thisLevelTableSize);
	tileCoordinate = fract(tileCoordinate); // optimize this fract at some point
	tileCoordinate = uintBitsToFloat(floatBitsToUint(tileCoordinate)+uint((_NBL_VT_IMPL_PAGE_SZ_LOG2-levelInTail)<<23));
    tileCoordinate += packingOffsets[levelInTail];
	tileCoordinate *= phys_pg_tex_sz_rcp;

	vec3 physicalUV = nbl_glsl_unpackPageID(levelInTail!=0 ? pageID.y:pageID.x);
	physicalUV.xy *= vec2(_NBL_VT_IMPL_PAGE_SZ+2*_NBL_VT_IMPL_TILE_PADDING)*phys_pg_tex_sz_rcp;

	// add the in-tile coordinate
	physicalUV.xy += tileCoordinate;
    return physicalUV;
}

#include <nbl/builtin/glsl/math/functions.glsl>

#if _NBL_VT_FLOAT_VIEWS_COUNT
// textureGrad emulation
vec4 nbl_glsl_vTextureGrad_impl(in uint formatID, in vec3 virtualUV, in mat2 dOriginalScaledUV, in uint originalMaxFullMip)
{
	// returns what would have been `textureGrad(originalTexture,gOriginalUV[0],gOriginalUV[1])
	// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap15.html#textures-normalized-operations
	const float kMaxAnisotropy = float(2u*_NBL_VT_IMPL_TILE_PADDING);
	// you can use an approx `log2` if you know one
#ifdef _NBL_APPROXIMATE_TEXEL_FOOTPRINT_FROM_DERIVATIVE_CACL_
	// bounded by sqrt(2)
	float p_x_2_log2 = log2(nbl_glsl_lengthManhattan(dOriginalScaledUV[0]));
	float p_y_2_log2 = log2(nbl_glsl_lengthManhattan(dOriginalScaledUV[1]));
	const float kMaxAnisoLogOffset = log2(kMaxAnisotropy);
#else
	float p_x_2_log2 = log2(nbl_glsl_lengthSq(dOriginalScaledUV[0]));
	float p_y_2_log2 = log2(nbl_glsl_lengthSq(dOriginalScaledUV[1]));
	const float kMaxAnisoLogOffset = log2(kMaxAnisotropy)*2.0;
#endif
	bool xIsMajor = p_x_2_log2>p_y_2_log2;
	float p_min_2_log2 = xIsMajor ? p_y_2_log2:p_x_2_log2;
	float p_max_2_log2 = xIsMajor ? p_x_2_log2:p_y_2_log2;

	float LoD = max(p_min_2_log2,p_max_2_log2-kMaxAnisoLogOffset);
#ifdef _NBL_APPROXIMATE_TEXEL_FOOTPRINT_FROM_DERIVATIVE_CACL_
	LoD += 0.5;
#else
	LoD *= 0.5;
#endif
    LoD += 5.f;
	// WARNING: LoD_high will round up when LoD negative, its not a floor
	int LoD_high = int(LoD);

    bool haveToDoTrilinear = false; // have to do trilinear only if doing minification AND larger than 1x1 footprint
    int levelInTail = 0;
    int clippedLoD = 0;
    if (LoD>0.f) // are we performing minification
    {
        const bool bbbb = LoD_high>=originalMaxFullMip;
	    // clip to max representable in VT, originalMaxFullMip is always -1 in case of no miplevel taking at least 1 full page
	    clippedLoD = bbbb ? (int(max(originalMaxFullMip,1u))-1):LoD_high;

        if (bbbb)
        {
	        // if minification is being performaed then get tail position
	        levelInTail = LoD_high-clippedLoD;
	        // have to do trilinear only if doing minification AND larger than 1x1 footprint
	        haveToDoTrilinear = LoD_high-clippedLoD<int(_NBL_VT_IMPL_PAGE_SZ_LOG2);
	        levelInTail = haveToDoTrilinear ? levelInTail:int(_NBL_VT_IMPL_PAGE_SZ_LOG2);
            levelInTail += originalMaxFullMip!=0u ? 0:1;
        }
        else
        {
            levelInTail += originalMaxFullMip!=0u ? 0:1;
            haveToDoTrilinear = true;
        }
    }
    else
        levelInTail = originalMaxFullMip!=0u ? 0:1;

	// get the higher resolution mip-map level
	vec3 hiPhysCoord = nbl_glsl_vTexture_helper(formatID,virtualUV,clippedLoD,levelInTail);
	// get lower if needed (speculative execution, had to move divergent indexing to a single place)
    vec3 loPhysCoord;
	// speculative if (haveToDoTrilinear)
	{
		// now we have absolute guarantees that both LoD_high and LoD_low are in the valid original mip range
		bool highNotInLastFull = originalMaxFullMip>=1u && LoD_high<(int(originalMaxFullMip)-1);
		clippedLoD = highNotInLastFull ? (clippedLoD+1):clippedLoD;
		levelInTail = highNotInLastFull ? levelInTail:(levelInTail+1);
        levelInTail = min(levelInTail, int(_NBL_VT_IMPL_PAGE_SZ_LOG2));
		loPhysCoord = nbl_glsl_vTexture_helper(formatID,virtualUV,clippedLoD,levelInTail);
	}

	vec4 hiMip_retval;
    vec4 loMip;
#ifdef NBL_GL_EXT_nonuniform_qualifier
    hiMip_retval = textureGrad(physicalTileStorageFormatView[nonuniformEXT(formatID)],hiPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
    if (haveToDoTrilinear)
        loMip = textureGrad(physicalTileStorageFormatView[nonuniformEXT(formatID)],loPhysCoord,dOriginalScaledUV[0],dOriginalScaledUV[1]);
#else
    uvec2 outstandingSampleMask = subgroupBallot(true).xy;
    // maybe unroll a few times manually
    while (outstandingSampleMask!=uvec2(0u))
    {
		uvec2 tmp = outstandingSampleMask;
        uint subgroupFormatID = subgroupBroadcast(formatID,tmp[1]!=0u ? 32u:findLSB(tmp[0]));
        bool canSample = subgroupFormatID==formatID; // do I need this? && (outstandingSampleMask&gl_SubGroupEqMaskARB)==gl_SubGroupEqMaskARB;
        outstandingSampleMask ^= subgroupBallot(canSample).xy;
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

vec4 nbl_glsl_vTextureGrad(in uvec2 _texData, in vec2 uv, in mat2 dUV)
{
    vec2 originalSz = nbl_glsl_unpackSize(_texData);
	dUV[0] *= originalSz;
	dUV[1] *= originalSz;

    uvec2 wrap = nbl_glsl_unpackWrapModes(_texData);
    uv.x = nbl_glsl_wrapTexCoord(uv.x,wrap.x);
    uv.y = nbl_glsl_wrapTexCoord(uv.y,wrap.y);

	vec3 virtualUV = nbl_glsl_unpackVirtualUV(_texData);

    uint formatID = nbl_glsl_VT_layer2pid(uint(virtualUV.z));

    virtualUV.xy += uv*originalSz;
    virtualUV.xy *= nbl_glsl_VT_getVTexSzRcp();

    return nbl_glsl_vTextureGrad_impl(formatID, virtualUV, dUV, nbl_glsl_unpackMaxMipInVT(_texData));
}
#endif //_NBL_VT_FLOAT_VIEWS_COUNT

#if _NBL_VT_INT_VIEWS_COUNT
ivec4 nbl_glsl_iVTextureLod_impl(in uint formatID, in vec3 virtualUV, in uint lod, in uint originalMaxFullMip)
{
    int nonnegativeLod = int(lod);
    int clippedLoD = min(nonnegativeLod,originalMaxFullMip);
    int levelInTail = nonnegativeLod - clippedLoD;
    
    vec3 physCoord = nbl_glsl_vTexture_helper(formatID, virtualUV, clippedLoD, levelInTail);
#ifdef NBL_GL_EXT_nonuniform_qualifier
	return textureLod(iphysicalTileStorageFormatView[nonuniformEXT(formatID)], physCoord, lod);
#else
    ivec4 retval;
    uint64_t outstandingSampleMask = ballotARB(true);
    while (outstandingSampleMask != uint64_t(0u)) 
    {
        uvec2 tmp = unpackUint2x32(outstandingSampleMask);
        uint subgroupFormatID = readInvocationARB(formatID, tmp[1] != 0u ? 32u : findLSB(tmp[0]));
        bool canSample = subgroupFormatID == formatID;
        outstandingSampleMask ^= ballotARB(canSample);
        if (canSample)
            retval = textureLod(iphysicalTileStorageFormatView[subgroupFormatID], physCoord, lod);
    }
    return retval;
#endif
}
ivec4 nbl_glsl_iVTextureLod(in uvec2 _texData, in vec2 uv, in uint lod)
{
    vec2 originalSz = nbl_glsl_unpackSize(_texData);
	
    uvec2 wrap = nbl_glsl_unpackWrapModes(_texData);
    uv.x = nbl_glsl_wrapTexCoord(uv.x, wrap.x);
    uv.y = nbl_glsl_wrapTexCoord(uv.y, wrap.y);
    
    vec3 virtualUV = nbl_glsl_unpackVirtualUV(_texData);
    uint formatID = nbl_glsl_VT_layer2pid(uint(virtualUV.z));
    virtualUV.xy += uv * originalSz;
    virtualUV.xy *= nbl_glsl_VT_getVTexSzRcp();
	
    return nbl_glsl_iVTextureLod_impl(formatID, virtualUV, lod, nbl_glsl_unpackMaxMipInVT(_texData));
}
#endif //_NBL_VT_INT_VIEWS_COUNT

#if _NBL_VT_UINT_VIEWS_COUNT
uvec4 nbl_glsl_uVTextureLod_impl(in uint formatID, in vec3 virtualUV, in uint lod, in uint originalMaxFullMip)
{
    int nonnegativeLod = int(lod);
    int clippedLoD = min(nonnegativeLod,originalMaxFullMip);
    int levelInTail = nonnegativeLod - clippedLoD;
    
    vec3 physCoord = nbl_glsl_vTexture_helper(formatID, virtualUV, clippedLoD, levelInTail);
#ifdef NBL_GL_EXT_nonuniform_qualifier
	return textureLod(uphysicalTileStorageFormatView[nonuniformEXT(formatID)], physCoord, lod);
#else
    uvec4 retval;
    uint64_t outstandingSampleMask = ballotARB(true);
    while (outstandingSampleMask != uint64_t(0u)) 
    {
        uvec2 tmp = unpackUint2x32(outstandingSampleMask);
        uint subgroupFormatID = readInvocationARB(formatID, tmp[1] != 0u ? 32u : findLSB(tmp[0]));
        bool canSample = subgroupFormatID == formatID;
        outstandingSampleMask ^= ballotARB(canSample);
        if (canSample)
            retval = textureLod(uphysicalTileStorageFormatView[subgroupFormatID], physCoord, lod);
    }
    return retval;
#endif
}
uvec4 nbl_glsl_uVTextureLod(in uvec2 _texData, in vec2 uv, in uint lod)
{
    vec2 originalSz = nbl_glsl_unpackSize(_texData);
	
    uvec2 wrap = nbl_glsl_unpackWrapModes(_texData);
    uv.x = nbl_glsl_wrapTexCoord(uv.x, wrap.x);
    uv.y = nbl_glsl_wrapTexCoord(uv.y, wrap.y);
    
    vec3 virtualUV = nbl_glsl_unpackVirtualUV(_texData);
    uint formatID = nbl_glsl_VT_layer2pid(uint(virtualUV.z));
    virtualUV.xy += uv * originalSz;
    virtualUV.xy *= nbl_glsl_VT_getVTexSzRcp();
	
    return nbl_glsl_uVTextureLod_impl(formatID, virtualUV, lod, nbl_glsl_unpackMaxMipInVT(_texData));
}
#endif //_NBL_VT_UINT_VIEWS_COUNT

/*
#ifdef NBL_GL_EXT_nonuniform_qualifier
	#define _NBL_DIVERGENT_SAMPLING_IMPL(retval_t, physicalSamplerName) return textureLod(physicalSamplerName[nonuniformEXT(formatID)], physCoord, lod)
#else
	#define _NBL_DIVERGENT_SAMPLING_IMPL(retval_t, physicalSamplerName) \
    retval_t retval; \
    uint64_t outstandingSampleMask = ballotARB(true); \
    while (outstandingSampleMask != uint64_t(0u)) \ 
    { \
        uvec2 tmp = unpackUint2x32(outstandingSampleMask); \
        uint subgroupFormatID = readInvocationARB(formatID, tmp[1] != 0u ? 32u : findLSB(tmp[0])); \
        bool canSample = subgroupFormatID == formatID; \
        outstandingSampleMask ^= ballotARB(canSample); \
        if (canSample) \
            retval = textureLod(physicalSamplerName[subgroupFormatID], physCoord, lod); \
    } \
    return retval
#endif

//problem is with this line below "unexpected LEFT_BRACE", no idea why
#define _NBL_DEFINE_VT_INTEGER_FUNCTIONS(funcName, implFuncName, retval_t, physicalSamplerName) retval_t implFuncName##(in uint formatID, in vec3 virtualUV, in uint lod, in int originalMaxFullMip) \
{ \
    int nonnegativeLod = int(lod); \
    int clippedLoD = min(nonnegativeLod,originalMaxFullMip); \
    int levelInTail = nonnegativeLod - clippedLoD; \
    \
    vec3 physCoord = nbl_glsl_vTexture_helper(formatID, virtualUV, clippedLoD, levelInTail); \
	_NBL_DIVERGENT_SAMPLING_IMPL(retval_t, physicalSamplerName); \
} \
retval_t funcName(in uvec2 _texData, in vec2 uv, in uint lod) \
{ \
    vec2 originalSz = nbl_glsl_unpackSize(_texData); \
	\
    uvec2 wrap = nbl_glsl_unpackWrapModes(_texData); \
    uv.x = nbl_glsl_wrapTexCoord(uv.x, wrap.x); \
    uv.y = nbl_glsl_wrapTexCoord(uv.y, wrap.y); \
    \
    vec3 virtualUV = nbl_glsl_unpackVirtualUV(_texData); \
    uint formatID = nbl_glsl_VT_layer2pid(uint(virtualUV.z)); \
    virtualUV.xy += uv * originalSz; \
    virtualUV.xy *= nbl_glsl_VT_getVTexSzRcp(); \
	\
    return nbl_glsl_vTextureLod_impl(formatID, virtualUV, lod, nbl_glsl_unpackMaxMipInVT(_texData)); \
}

_NBL_DEFINE_VT_INTEGER_FUNCTIONS(nbl_glsl_iVTextureLod, nbl_glsl_iVTextureLod_impl, ivec4, iphysicalTileStorageFormatView)
_NBL_DEFINE_VT_INTEGER_FUNCTIONS(nbl_glsl_uVTextureLod, nbl_glsl_uVTextureLod_impl, uvec4, uphysicalTileStorageFormatView)
#undef _NBL_DIVERGENT_SAMPLING_IMPL
*/

#endif //!_NBL_BUILTIN_GLSL_VIRTUAL_TEXTURING_IMPL_FUNCTIONS_INCLUDED_