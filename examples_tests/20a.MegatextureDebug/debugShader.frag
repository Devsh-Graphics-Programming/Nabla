#version 430 core

layout(binding = 0) uniform usampler2D pageTable[3]; // MAKE PAGE TABLE AN ARRAY FFS!!!
layout(binding = 3) uniform sampler2DArray physPgTex[3];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;


#define ADDR_LAYER_SHIFT 12
#define ADDR_Y_SHIFT 6
#define ADDR_X_MASK 0x3fu // X and Y mask COULD BE DIFFERENT, X and Y SIZE COULD BE DIFFERENT!

#define TEXTURE_TILE_PER_DIMENSION 64
#define PAGE_SZ 128
#define PAGE_SZ_LOG2 7
#define TILE_PADDING 8

vec3 unpackPageID(in uint pageID)
{
	return vec3(uvec2(pageID,pageID>>ADDR_Y_SHIFT)&uvec2(ADDR_X_MASK),float(pageID>>ADDR_LAYER_SHIFT));
}

vec2 unpackVirtualUV(in uvec2 texData)
{
	return unpackUnorm2x16(texData.x);
}
vec2 unpackSize(in uvec2 texData)
{
	return unpackUnorm2x16(texData.y);
}

//const vec4 packingOffsets[] = {};

void main()
{
	// dummy
	uint originalMipSize_maxDim = PAGE_SZ;
	vec3 virtualUV = vec3(TexCoord/16.0,0.0); // the division is just for magnification
	float tilesInLodLevel = float(TEXTURE_TILE_PER_DIMENSION);

	// proper
	vec2 tileFractionalCoordinate = fract(virtualUV.xy*tilesInLodLevel);
	// page address and offset decode
    uvec2 pageID = textureLod(pageTable[1],virtualUV.xy,0).xy; // NOT TEXEL FETCH!!!
	// compute physical coord
	vec3 physicalUV = unpackPageID(originalMipSize_maxDim<=(PAGE_SZ/2) ? pageID.y : pageID.x);
	// add corner offset
	physicalUV.xy += vec2(TILE_PADDING)/float(PAGE_SZ+2*TILE_PADDING);
	// add the tile fractional coordinate scaled by difference between unpadded and padded size
	physicalUV.xy += tileFractionalCoordinate*float(PAGE_SZ)/float(PAGE_SZ+2*TILE_PADDING);
	// scale from [0,MaxTileCoord] to [0,1]
	physicalUV.xy /= tilesInLodLevel;
	//
	OutColor = textureLod(physPgTex[1],physicalUV,0.0);
}

