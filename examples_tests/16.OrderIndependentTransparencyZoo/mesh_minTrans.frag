#version 420 core

layout(binding = 0) uniform sampler2D tex0;
uniform vec3 selfPos;

in vec2 TexCoord;
in float height;

layout(origin_upper_left) in vec4 gl_FragCoord;
layout(location = 0) out vec4 pixelColor;

void main()
{
    float alpha = min(height*(length(selfPos.xz)*0.02+1.0)*0.03,1.0);
    if (alpha<1.0/255.0)
        discard;

    float storedDepth = gl_FragCoord.z;

    //swap alpha with depth :D
    ///pixelColor = vec4(texture(tex0,TexCoord).rgb,(floatBitsToInt(storedDepth)&255)/255.0);
    ///pixelColor = vec4(texture(tex0,TexCoord).rgb,(floatBitsToInt(storedDepth)&127)/127.0);
    ///pixelColor = vec4(texture(tex0,TexCoord).rgb,(floatBitsToInt(storedDepth)&31)/31.0);
    pixelColor = vec4(texture(tex0,TexCoord).rgb,(floatBitsToInt(storedDepth)&7)/7.0);
    //thanks to IEEE sorting floats is equivalent to sorting ints
    //so put alpha in most significant bits
    //and stuff some of the exponent and mantissa of depth in modified depth so same opacity surfaces resolve properly (nearest also taken into account)
    ///gl_FragDepth = intBitsToFloat((int(alpha*243.0)<<22)|(floatBitsToInt(storedDepth)>>8));
    //other bucketing strategies
    ///gl_FragDepth = intBitsToFloat((int(alpha*126.0)<<23)|(floatBitsToInt(storedDepth)>>7));
    ///gl_FragDepth = intBitsToFloat((int(alpha*31.0)<<25)|(floatBitsToInt(storedDepth)>>5));
    gl_FragDepth = intBitsToFloat((int(alpha*7.0)<<27)|(floatBitsToInt(storedDepth)>>3));

    // use 243.0 instead of 255.0 so that gl_FragDepth < 1.0 always, so full opacity faces resolve their depths properly



    //GPU doesn't take denormalized
    //gl_FragDepth = intBitsToFloat((1<<23)-1);
}
