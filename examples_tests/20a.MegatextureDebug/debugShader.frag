#version 430 core
layout(binding = 0) uniform usampler2D pageTable[3]; // MAKE PAGE TABLE AN ARRAY FFS!!!
layout(binding = 3) uniform sampler2DArray physPgTex[3];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;


#define ADDR_LAYER_SHIFT 12
#define ADDR_Y_SHIFT 6
#define ADDR_X_MASK 0x3fu

#define TEXTURE_TILE_PER_DIMENSION 64
#define PAGE_SZ 128
#define PAGE_SZ_LOG2 7
#define TILE_PADDING 8

vec3 unpackPageID(in uint pageID)
{
	vec2 uv = vec2(float(pageID & ADDR_X_MASK), float((pageID>>ADDR_Y_SHIFT) & ADDR_X_MASK))*(PAGE_SZ+2*TILE_PADDING) + TILE_PADDING;
	uv /= vec2(textureSize(physPgTex[1],0).xy);
	return vec3(uv, float(pageID >> ADDR_LAYER_SHIFT));
    //return vec3(vec2(TILE_PADDING)/vec2(textureSize(physPgTex[1],0).xy), 0.0);
}

void main()
{
	// dummy
	uint originalMipSize_maxDim = PAGE_SZ;
	vec3 virtualUV = vec3(TexCoord,0.0);
	// proper
    uvec2 pageID = textureLod(pageTable[1],virtualUV.xy,0).xy; // NOT TEXEL FETCH!!!
	vec3 physicalUV = unpackPageID(originalMipSize_maxDim<=(PAGE_SZ/2) ? pageID.y : pageID.x);
    OutColor = vec4(physicalUV, 1.0);
}

