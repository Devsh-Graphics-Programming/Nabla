#version 430 core

layout(set = 0, binding = 0) uniform usampler2DArray pageTable[3]; // MAKE PAGE TABLE AN ARRAY FFS!!!
layout(set = 0, binding = 3) uniform sampler2DArray physPgTex[3];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

// constants
#define ADDR_LAYER_SHIFT 12u
#define ADDR_Y_SHIFT 6u
#define ADDR_X_PREMASK 
#define ADDR_X_MASK uint((0x1u<<ADDR_Y_SHIFT)-1u)
#define ADDR_Y_MASK uint((0x1u<<(ADDR_LAYER_SHIFT-ADDR_Y_SHIFT))-1u)

#define PAGE_SZ 128
#define PAGE_SZ_LOG2 7
#define TILE_PADDING 8
#define PADDED_TILE_SIZE uint(PAGE_SZ+2*TILE_PADDING)

const uvec2 packingOffsets[] = uvec2[PAGE_SZ_LOG2+1](
	uvec2(TILE_PADDING,TILE_PADDING),uvec2(0xdeadbeefu),uvec2(0xdeadbeefu),uvec2(0xdeadbeefu),uvec2(0xdeadbeefu),uvec2(0xdeadbeefu),uvec2(0xdeadbeefu),uvec2(0xdeadbeefu)
);
// end constants

vec3 unpackPageID(in uint pageID)
{
	// this is optimal, don't touch
	uvec2 pageXY = uvec2(pageID,pageID>>ADDR_Y_SHIFT)&uvec2(ADDR_X_MASK,ADDR_Y_MASK);
	uvec2 pageOffset = pageXY*PADDED_TILE_SIZE;
	return vec3(vec2(pageOffset),pageID>>ADDR_LAYER_SHIFT);
}

vec4 vTextureGrad_helper(in vec3 virtualUV, int LoD, in mat2 gradients, in ivec2 originalTextureSz, in int clippedTextureLoD)
{
    uvec2 pageID = textureLod(pageTable[1],virtualUV,LoD).xy;

	// WANT: to get rid of this `textureSize` call
	float tilesInLodLevel = float(textureSize(pageTable[1],LoD).x);
	// TODO: rename to tileCoordinate if the dimensions will stay like this
	vec2 tileFractionalCoordinate = fract(virtualUV.xy*tilesInLodLevel);

	int levelInTail = max(LoD-clippedTextureLoD,0);
	tileFractionalCoordinate *= float(PAGE_SZ)*intBitsToFloat((127-levelInTail)<<23); // IEEE754 hack
	tileFractionalCoordinate += packingOffsets[levelInTail];

	vec3 physicalUV = unpackPageID(levelInTail!=0 ? pageID.y:pageID.x);
	// add the in-tile coordinate
	physicalUV.xy += tileFractionalCoordinate;
	// scale from absolute coordinate to normalized (could actually use a denormalized sampler for this)
	physicalUV.xy /= vec2(textureSize(physPgTex[1],0).xy);

	return textureGrad(physPgTex[1],physicalUV,gradients[0],gradients[1]);
}

float lengthSq(in vec2 v)
{
  return dot(v,v);
}
// textureGrad emulation
vec4 vTextureGrad(in vec3 virtualUV, in mat2 dOriginalUV, in vec2 originalTextureSize, in int clippedTextureLoD)
{
  // returns what would have been `textureGrad(originalTexture,gOriginalUV[0],gOriginalUV[1])
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap15.html#textures-normalized-operations
  const float kMaxAnisotropy = float(2*TILE_PADDING);
  const float kMaxAnisoLogOffset = log2(kMaxAnisotropy);
  // you can use an approx `log2` if you know one
  float p_x_2_log2 = log2(lengthSq(dOriginalUV[0]*originalTextureSize));
  float p_y_2_log2 = log2(lengthSq(dOriginalUV[1]*originalTextureSize));
  bool xIsMajor = p_x_2_log2>p_y_2_log2;
  float p_min_2_log2 = xIsMajor ? p_y_2_log2:p_x_2_log2;
  float p_max_2_log2 = xIsMajor ? p_x_2_log2:p_y_2_log2;
  float LoD = max(p_min_2_log2,p_max_2_log2-kMaxAnisoLogOffset);
  int LoD_high = int(LoD);
#ifdef DRIVER_HAS_GRADIENT_SCALING_ISSUES // don't use if you don't have to
  // normally when sampling an imageview with only 1 miplevel using `textureGrad` the gradient vectors can be scaled arbitrarily because we cannot select any other mip-map
  // however driver might want to try and get "clever" on us and decide that since `p_min` is huge, it wont bother with anisotropy
  float anisotropy = min(exp2((p_max_2_log2-p_min_2_log2)*0.5),kMaxAnisotropy);
  dOriginalUV[0] = normalize(dOriginalUV[0])*(1.0/float(MEGA_TEXTURE_SIZE));
  dOriginalUV[1] = normalize(dOriginalUV[1])*(1.0/float(MEGA_TEXTURE_SIZE));
  dOriginalUV[xIsMajor ? 0:1] *= anisotropy;
#endif
  ivec2 originalTexSz_i = ivec2(originalTextureSize);
  return mix(
			vTextureGrad_helper(virtualUV,LoD_high,dOriginalUV,originalTexSz_i,clippedTextureLoD),
			vTextureGrad_helper(virtualUV,LoD_high+1,dOriginalUV,originalTexSz_i,clippedTextureLoD),
			LoD-float(LoD_high)
			);
}

/*
vec2 unpackVirtualUV(in uvec2 texData)
{
	return unpackUnorm2x16(texData.x);
}
vec2 unpackSize(in uvec2 texData)
{
	return unpackUnorm2x16(texData.y);
}
vec4 textureVT(in uvec2 _texData, in vec2 uv, in mat2 dUV)
{
    vec2 scale = unpackSize(_texData);
    vec2 virtualUV = unpackVirtualUV(_texData);
    virtualUV += scale*uv;
    return vTextureGrad(virtualUV, dUV, scale*float(PAGE_SZ)*vec2(textureSize(pgTabTex[1],0)));
}
*/

void main()
{
	// dummy
	int originalTextureLoD = 11;
	// new STexData
	ivec2 originalTextureSize = ivec2(1024,1024);
	ivec3 vtOffsetInPages = ivec3(32,16,0);
	int clippedTextureLoD = max(originalTextureLoD-PAGE_SZ_LOG2-1,0);
	// half dummy
	vec3 virtualUV = vec3(TexCoord*float(PAGE_SZ)/vec2(originalTextureSize)+vec2(vtOffsetInPages.xy)/vec2(textureSize(pageTable[1],0).xy),vtOffsetInPages.z); // the division is just for magnification
	int LoD = 0;
	mat2 gradients = mat2(0.0,0.0,0.0,0.0);

	//proper
	OutColor = vTextureGrad_helper(virtualUV,LoD,gradients,originalTextureSize,clippedTextureLoD);
}

