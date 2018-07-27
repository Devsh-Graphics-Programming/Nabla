#version 420 core
layout(binding = 0) uniform sampler2DMS tex0; //color
layout(binding = 1) uniform sampler2DMS tex1; //depth

#define kBufferSamples 8

in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;

struct FragmentInfo
{
    vec4 color;
    float depth;
};

void loadLayers(in ivec2 integerTexCoord, in int expectedLayer, inout FragmentInfo prev, out FragmentInfo current)
{
    if (prev.depth!=0.0)
    {
        current.depth = texelFetch(tex1,integerTexCoord,expectedLayer).x;
        prev.color = texelFetch(tex0,integerTexCoord,expectedLayer-1);
    }
    else
        current.depth = 0.0;
}

void decodeDepth(inout FragmentInfo fragInfo)
{
    if (fragInfo.depth!=0.0)
    {
        int dAsInt = floatBitsToInt(fragInfo.depth);
        ///fragInfo.depth = intBitsToFloat(((dAsInt&4194303)<<8)|int(fragInfo.color.a*255.0));
        ///fragInfo.depth = intBitsToFloat(((dAsInt&8388607)<<7)|int(fragInfo.color.a*127.0));
        ///fragInfo.depth = intBitsToFloat(((dAsInt&33554431)<<5)|int(fragInfo.color.a*31.0));
        fragInfo.depth = intBitsToFloat(((dAsInt&134217727)<<3)|int(fragInfo.color.a*7.0));

        ///float alpha = float(dAsInt>>22)/243.0;
        ///float alpha = float(dAsInt>>23)/126.0;
        ///float alpha = float(dAsInt>>25)/31.0;
        float alpha = float(dAsInt>>27)/7.0;
        fragInfo.color.a = alpha;
    }
}

#define SWAP(x,y) \
if (layers[x].depth<layers[y].depth) \
{ \
    vec4 tmp_color = layers[x].color; \
    float tmp_depth = layers[x].depth; \
    layers[x].color = layers[y].color; \
    layers[x].depth = layers[y].depth; \
    layers[y].color = tmp_color; \
    layers[y].depth = tmp_depth; \
}

void fetchAndBlendUnder(inout vec4 compositedColor, in FragmentInfo depthLayer)
{
    if (depthLayer.depth!=0.0)
    {
        compositedColor.rgb += depthLayer.color.rgb*compositedColor.a*depthLayer.color.a;
        compositedColor.a *= (1.0-depthLayer.color.a);
    }
}

void main()
{
    ivec2 integerTexCoord = ivec2(TexCoord*textureSize(tex0));

    FragmentInfo layers[8];
    layers[0].depth = texelFetch(tex1,integerTexCoord,0).x;
    if (layers[0].depth==0.0)
        discard;

    loadLayers(integerTexCoord,1,layers[0],layers[1]);
    loadLayers(integerTexCoord,2,layers[1],layers[2]);
    loadLayers(integerTexCoord,3,layers[2],layers[3]);
    loadLayers(integerTexCoord,4,layers[3],layers[4]);
    loadLayers(integerTexCoord,5,layers[4],layers[5]);
    loadLayers(integerTexCoord,6,layers[5],layers[6]);
    loadLayers(integerTexCoord,7,layers[6],layers[7]);
    if (layers[7].depth!=0.0)
        layers[7].color = texelFetch(tex0,integerTexCoord,7);


    decodeDepth(layers[0]);
    decodeDepth(layers[1]);
    decodeDepth(layers[2]);
    decodeDepth(layers[3]);
    decodeDepth(layers[4]);
    decodeDepth(layers[5]);
    decodeDepth(layers[6]);
    decodeDepth(layers[7]);


    //sort 2
    if (layers[1].depth!=0.0)
        SWAP(0, 1);

    //sort [3,4]
    if (layers[2].depth!=0.0)
    {
        SWAP(2, 3);
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(1, 2);
    }

    //sort [5,8]
    if (layers[4].depth!=0.0)
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


    vec4 compositedColor = layers[0].color;
    compositedColor.a = 1.0-compositedColor.a;

    fetchAndBlendUnder(compositedColor,layers[1]);
    fetchAndBlendUnder(compositedColor,layers[2]);
    fetchAndBlendUnder(compositedColor,layers[3]);
    fetchAndBlendUnder(compositedColor,layers[4]);
    fetchAndBlendUnder(compositedColor,layers[5]);
    fetchAndBlendUnder(compositedColor,layers[6]);
    fetchAndBlendUnder(compositedColor,layers[7]);


    pixelColor = vec4(compositedColor.rgb,1.0-compositedColor.a);
}

