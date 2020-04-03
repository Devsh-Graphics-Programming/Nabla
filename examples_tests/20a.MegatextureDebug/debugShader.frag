#version 430 core

layout(binding = 0) uniform usampler2D pageTable[3]; // MAKE PAGE TABLE AN ARRAY FFS!!!
layout(binding = 3) uniform sampler2DArray physPgTex[3];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

#define ADDR_LAYER_SHIFT 12u
#define ADDR_Y_SHIFT 6u
#define ADDR_X_PREMASK 
#define ADDR_X_MASK ((0x1u<<ADDR_Y_SHIFT)-1u)
#define ADDR_Y_MASK ((0x1u<<(ADDR_LAYER_SHIFT-ADDR_Y_SHIFT))-1u)

#define PAGE_SZ 128
#define PAGE_SZ_LOG2 7
#define TILE_PADDING 8
#define PADDED_TILE_SIZE (PAGE_SZ+2*TILE_PADDING)

vec3 unpackPageID(in uint pageID)
{
	// this is optimal, don't touch
	return vec3(uvec2(pageID,pageID>>ADDR_Y_SHIFT)&uvec2(ADDR_X_MASK,ADDR_Y_MASK),0.0);
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
	vec3 virtualUV = vec3(TexCoord/2.0,0.0); // the division is just for magnification
	vec2 tilesInLodLevel = vec2(textureSize(pageTable[1],0).xy);

	// proper
	vec2 tileFractionalCoordinate = fract(virtualUV.xy*tilesInLodLevel);
	// TODO: use FMAD and a const array of packingOffsets as *packingOffsets.xy+packingOffsets.zw
	// scale by difference between unpadded and padded size
	tileFractionalCoordinate.xy *= float(PAGE_SZ)/float(PADDED_TILE_SIZE);
	// add corner offset
	tileFractionalCoordinate.xy += vec2(TILE_PADDING)/float(PADDED_TILE_SIZE);

	// page address and offset decode
    uvec2 pageID = textureLod(pageTable[1],virtualUV.xy,0).xy; // NOT TEXEL FETCH!!!
	// compute physical coord
	vec3 physicalUV = unpackPageID(originalMipSize_maxDim<=(PAGE_SZ/2) ? pageID.y : pageID.x);
	// add the in-tile coordinate
	physicalUV.xy += tileFractionalCoordinate;
	// scale from [0,MaxTileCoord] to [0,1]
	physicalUV.xy *= vec2(PADDED_TILE_SIZE,PADDED_TILE_SIZE)/vec2(textureSize(physPgTex[1],0).xy);
	//
	OutColor = textureLod(physPgTex[1],physicalUV,0.0);
}

