#version 330 core
uniform sampler2DMS tex0; //color
uniform sampler2DMS tex1; //depth

#define kBufferSamples 8

in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;

void loadDepthLayer(in ivec2 integerTexCoord, in int expectedLayer, in float prevDepth, out vec2 depthLayer)
{
    if (prevDepth!=0.0)
        depthLayer.x = texelFetch(tex1,integerTexCoord,expectedLayer).x;
    else
        depthLayer.x = 0.0;
    depthLayer.y = intBitsToFloat(expectedLayer);
}

#define SWAP(x,y) \
if (depthLayer[x].r<depthLayer[y].r) \
{ \
    vec2 tmp = depthLayer[x]; \
    depthLayer[x] = depthLayer[y]; \
    depthLayer[y] = tmp; \
}

void fetchAndBlendUnder(inout vec4 compositedColor, in ivec2 integerTexCoord, in vec2 depthLayer)
{
    if (depthLayer.x!=0.0)
    {
        vec4 newCol = texelFetch(tex0,integerTexCoord,floatBitsToInt(depthLayer.y));
        newCol.rgb *= compositedColor.a*newCol.a;
        compositedColor.rgb += newCol.rgb;
        compositedColor.a *= (1.0-newCol.a);
    }
}

void main()
{
    ivec2 integerTexCoord = ivec2(TexCoord*textureSize(tex0));

    vec2 depthLayer[8];
    depthLayer[0] = vec2(texelFetch(tex1,integerTexCoord,0).x,intBitsToFloat(0));
    if (depthLayer[0].x==0.0)
        discard;

    loadDepthLayer(integerTexCoord,1,depthLayer[0].x,depthLayer[1]);
    loadDepthLayer(integerTexCoord,2,depthLayer[1].x,depthLayer[2]);
    loadDepthLayer(integerTexCoord,3,depthLayer[2].x,depthLayer[3]);
    loadDepthLayer(integerTexCoord,4,depthLayer[3].x,depthLayer[4]);
    loadDepthLayer(integerTexCoord,5,depthLayer[4].x,depthLayer[5]);
    loadDepthLayer(integerTexCoord,6,depthLayer[5].x,depthLayer[6]);
    loadDepthLayer(integerTexCoord,7,depthLayer[6].x,depthLayer[7]);


    //sort 2
    if (depthLayer[1].x!=0.0)
        SWAP(0, 1);

    //sort [3,4]
    if (depthLayer[2].x!=0.0)
    {
        SWAP(2, 3);
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(1, 2);
    }

    //sort [5,8]
    if (depthLayer[4].x!=0.0)
    {
        SWAP(4, 5);
        SWAP(6, 7);
        SWAP(4, 6);
        SWAP(5, 7);
        SWAP(5, 6);
        SWAP(0, 4);
        SWAP(1, 5);
        SWAP(1, 4);
        SWAP(2, 6);
        SWAP(3, 7);
        SWAP(3, 6);
        SWAP(2, 4);
        SWAP(3, 5);
        SWAP(3, 4);
    }


    vec4 compositedColor = texelFetch(tex0,integerTexCoord,floatBitsToInt(depthLayer[0].y));
    compositedColor.a = 1.0-compositedColor.a;

    fetchAndBlendUnder(compositedColor,integerTexCoord,depthLayer[1]);
    fetchAndBlendUnder(compositedColor,integerTexCoord,depthLayer[2]);
    fetchAndBlendUnder(compositedColor,integerTexCoord,depthLayer[3]);
    fetchAndBlendUnder(compositedColor,integerTexCoord,depthLayer[4]);
    fetchAndBlendUnder(compositedColor,integerTexCoord,depthLayer[5]);
    fetchAndBlendUnder(compositedColor,integerTexCoord,depthLayer[6]);
    fetchAndBlendUnder(compositedColor,integerTexCoord,depthLayer[7]);


    pixelColor = vec4(compositedColor.rgb,1.0-compositedColor.a);
}
