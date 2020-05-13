#ifndef __IRR_C_GLSL_COLOR_SPACE_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_COLOR_SPACE_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr
{
namespace asset
{    

class CGLSLColorSpaceBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
    {
    public:
        const char* getVirtualDirectoryName() const override { return "glsl/colorspace/"; }

    private:
        static std::string getEncodeCIEXYZ(const std::string&)
        {
            return
R"(#ifndef _IRR_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_
#define _IRR_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_

const mat3 irr_glsl_scRGBtoXYZ = mat3(  vec3( 0.4124564, 0.2126729, 0.0193339),
                                        vec3( 0.3575761, 0.7151522, 0.1191920),
                                        vec3( 0.1804375, 0.0721750, 0.9503041));

const mat3 irr_glsl_sRGBtoXYZ = scRGBtoXYZ;

const mat3 irr_glsl_BT709toXYZ = scRGBtoXYZ;


const mat3 irr_glsl_Display_P3toXYZ = mat3( vec3( 0.4865709, 0.2289746, 0.0000000),
                                            vec3( 0.2656677, 0.6917385, 0.0451134),
                                            vec3( 0.1982173, 0.0792869, 1.0439444));


const mat3 irr_glsl_DCI_P3toXYZ = mat3( vec3(1.0,0.0,0.0),
                                        vec3(0.0,1.0,0.0),
                                        vec3(0.0,0.0,1.0));


const mat3 irr_glsl_BT2020toXYZ = mat3( vec3( 0.6369580, 0.2627002, 0.0000000),
                                        vec3( 0.1446169, 0.6779981, 0.0280727),
                                        vec3( 0.1688810, 0.0593017, 1.0609851));

const mat3 irr_glsl_HDR10_ST2084toXYZ = irr_glsl_BT2020toXYZ;

const mat3 irr_glsl_DOLBYIVISIONtoXYZ = irr_glsl_BT2020toXYZ;

const mat3 irr_glsl_HDR10_HLGtoXYZ = irr_glsl_BT2020toXYZ;


const mat3 irr_glsl_AdobeRGBtoXYZ = mat3(   vec3( 0.57667, 0.29734, 0.02703),
                                            vec3( 0.18556, 0.62736, 0.07069),
                                            vec3( 0.18823, 0.07529, 0.99134));


const mat3 irr_glsl_ACES2065_1toXYZ = mat3( vec3( 0.9525523959, 0.3439664498, 0.0000000000),
                                            vec3( 0.0000000000, 0.7281660966, 0.0000000000),
                                            vec3( 0.0000936786,-0.0721325464, 1.0088251844));


const mat3 irr_glsl_ACEScctoXYZ = mat3( vec3( 0.6624542, 0.2722287,-0.0055746),
                                        vec3( 0.1340042, 0.6740818, 0.6740818),
                                        vec3( 0.1561877, 0.0536895, 1.0103391));

const mat3 irr_glsl_ACESccttoXYZ = irr_glsl_ACEScctoXYZ;
#endif
)";
        }
        static std::string getDecodeCIEXYZ(const std::string&)
        {
            return
R"(#ifndef _IRR_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_
#define _IRR_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_

const mat3 irr_glsl_XYZtoscRGB = mat3(  vec3( 3.2404542,-0.9692660, 0.0556434),
                                        vec3(-1.5371385, 1.8760108,-0.2040259),
                                        vec3(-0.4985314, 0.0415560, 1.0572252));

const mat3 irr_glsl_XYZtosRGB = XYZtoscRGB;

const mat3 irr_glsl_XYZtoBT709 = XYZtoscRGB;

  
const mat3 irr_glsl_XYZtoDisplay_P3 = mat3( vec3( 2.4934969,-0.8294890, 0.0358458),
                                            vec3(-0.9313836, 1.7626641,-0.0761724),
                                            vec3(-0.4027108, 0.0236247, 0.9568845));


const mat3 irr_glsl_XYZtoDCI_P3 = mat3(vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(0.0,0.0,1.0));

 
const mat3 irr_glsl_XYZtoBT2020 = mat3( vec3( 1.7166512,-0.6666844, 0.0176399),
                                        vec3(-0.3556708, 1.6164812,-0.0427706),
                                        vec3(-0.2533663, 0.0157685, 0.9421031));
 
const mat3 irr_glsl_XYZtoHDR10_ST2084 = irr_glsl_XYZtoBT2020;

const mat3 irr_glsl_XYZtoDOLBYIVISION = irr_glsl_XYZtoBT2020;

const mat3 irr_glsl_XYZtoHDR10_HLG = irr_glsl_XYZtoBT2020;


const mat3 irr_glsl_XYZtoAdobeRGB = mat3(   vec3( 2.04159,-0.96924, 0.01344),
                                            vec3(-0.56501, 1.87597,-0.11836),
                                            vec3(-0.34473, 0.04156, 1.01517));


const mat3 irr_glsl_XYZtoACES2065_1 = mat3( vec3( 1.0498110175, 0.0000000000,-0.0000974845),
                                            vec3(-0.4959030231, 1.3733130458, 0.0982400361),
                                            vec3( 0.0000000000, 0.0000000000, 0.9912520182));


const mat3 irr_glsl_XYZtoACEScc = mat3( vec3( 1.6410234,-0.6636629, 0.0117219),
                                        vec3(-0.3248033, 1.6153316,-0.0082844),
                                        vec3(-0.2364247, 0.0167563, 0.9883949));

const mat3 irr_glsl_XYZtoACEScct = irr_glsl_XYZtoACEScc;
#endif
)";
        }
        static std::string getEOTF(const std::string&)
        {
            return
R"(#ifndef _IRR_COLOR_SPACE_EOTF_INCLUDED_
#define _IRR_COLOR_SPACE_EOTF_INCLUDED_

vec3 irr_glsl_eotf_identity(in vec3 nonlinear)
{
    return nonlinear;
}

vec3 irr_glsl_eotf_impl_shared_2_4(in vec3 nonlinear, in float vertex)
{
    bvec3 right = greaterThan(nonlinear,vec3(vertex));
    return mix(nonlinear/12.92,pow((nonlinear+vec3(0.055))/1.055,vec3(2.4)),right);
}

// compatible with scRGB as well
vec3 irr_glsl_eotf_sRGB(in vec3 nonlinear)
{
    bvec3 negatif = lessThan(nonlinear,vec3(0.0));
    vec3 absVal = irr_glsl_eotf_impl_shared_2_4(abs(nonlinear),0.04045);
    return negatif ? (-absVal):absVal;
}

// also known as P3-D65
vec3 irr_glsl_eotf_Display_P3(in vec3 nonlinear)
{
    return irr_glsl_eotf_impl_shared_2_4(nonlinear,0.039000312);
}


vec3 irr_glsl_eotf_DCI_P3_XYZ(in vec3 nonlinear)
{
    return pow(nonlinear*52.37,vec3(2.6));
}

vec3 irr_glsl_eotf_SMPTE_170M(in vec3 nonlinear)
{
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    float alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    vec3 delta = vec3(0.081242858298635); // 0.0812 for all ITU but the BT.2020 12 bit encoding
    return mix(nonlinear/4.5,pow((nonlinear+vec3(alpha-1.0))/alpha,vec3(1.0/0.45)),greaterThanEqual(nonlinear,delta));
}

vec3 irr_glsl_eotf_SMPTE_ST2084(in vec3 nonlinear)
{
    const vec3 invm2 = vec3(1.0/78.84375);
    vec3 common = pow(y,invm2);

    const vec3 c2 = vec3(18.8515625);
    const float c3 = 18.68875;
    const vec3 c1 = vec3(c3+1.0)-c2;

    const vec3 invm1 = vec3(1.0/0.1593017578125);
    return pow(max(common-c1,vec3(0.0))/(c2-common*c3),invm1);
}

// did I do this right by applying the function for every color?
vec3 irr_glsl_eotf_HDR10_HLG(in vec3 nonlinear)
{
    // done with log2 so constants are different
    const float a = 0.1239574303172;
    const vec3 b = vec3(0.02372241);
    const vec3 c = vec3(1.0042934693729); 
    bvec3 right = greaterThan(nonlinear,vec3(0.5));
    return mix(nonlinear*nonlinear/3.0,exp2((nonlinear-c)/a)+b,right);
}

vec3 irr_glsl_eotf_AdobeRGB(in vec3 nonlinear)
{
    return pow(linear,vec3(2.19921875));
}

vec3 irr_glsl_eotf_Gamma_2_2(in vec3 nonlinear)
{
    return pow(linear,vec3(2.2));
}


vec3 irr_glsl_eotf_ACEScc(in vec3 nonlinear)
{
    bvec3 right = greaterThanEqual(nonlinear,vec3(-0.301369863));
    vec3 common = exp2(nonlinear*17.52-vec3(9.72));
    return max(mix(common*2.0-vec3(0.000030517578125),common,right),vec3(65504.0));
}

vec3 irr_glsl_eotf_ACEScct(in vec3 nonlinear)
{
    bvec3 right = greaterThanEqual(nonlinear,vec3(0.155251141552511));
    return max(mix((nonlinear-vec3(0.0729055341958355))/10.5402377416545,exp2(nonlinear*17.52-vec3(9.72)),right),vec3(65504.0));
}
#endif
)";
        }
        static std::string getOETF(const std::string&)
        {
            return
R"(#ifndef _IRR_COLOR_SPACE_OETF_INCLUDED_
#define _IRR_COLOR_SPACE_OETF_INCLUDED_

vec3 irr_glsl_oetf_identity(in vec3 linear)
{
    return linear;
}

vec3 irr_glsl_oetf_impl_shared_2_4(in vec3 linear, in float vertex)
{
    bvec3 right = greaterThan(linear,vec3(vertex));
    return mix(linear*12.92,pow(linear,vec3(1.0/2.4))*1.055-vec3(0.055),right);
}

// compatible with scRGB as well
vec3 irr_glsl_oetf_sRGB(in vec3 linear)
{
    bvec3 negatif = lessThan(linear,vec3(0.0));
    vec3 absVal = irr_glsl_oetf_impl_shared_2_4(abs(linear),0.0031308);
    return negatif ? (-absVal):absVal;
}

// also known as P3-D65
vec3 irr_glsl_oetf_Display_P3(in vec3 linear)
{
    return irr_glsl_oetf_impl_shared_2_4(linear,0.0030186);
}

vec3 irr_glsl_oetf_DCI_P3_XYZ(in vec3 linear)
{
    return pow(linear/52.37,vec3(1.0/2.6));
}

vec3 irr_glsl_oetf_SMPTE_170M(in vec3 linear)
{
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    const float alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    const vec3 beta = vec3(0.018053968510808); // 0.0181 for all ITU but the BT.2020 12 bit encoding, 0.18 otherwise
    return mix(linear*4.5,pow(linear,vec3(0.45))*alpha-vec3(alpha-1.0),greaterThanEqual(linear,beta));
}

vec3 irr_glsl_oetf_SMPTE_ST2084(in vec3 linear)
{
    const vec3 m1 = vec3(0.1593017578125);
    const vec3 m2 = vec3(78.84375);
    const float c2 = 18.8515625;
    const float c3 = 18.68875;
    const vec3 c1 = vec3(c3-c2+1.0);

    vec3 L_m1 = pow(linear,m1);
    return pow((c1+L_m1*c2)/(vec3(1.0)+L_m1*c3),m2);
}

// did I do this right by applying the function for every color?
vec3 irr_glsl_oetf_HDR10_HLG(in vec3 linear)
{
    // done with log2 so constants are different
    const float a = 0.1239574303172;
    const vec3 b = vec3(0.02372241);
    const vec3 c = vec3(1.0042934693729); 
    bvec3 right = greaterThan(linear,vec3(1.0/12.0));
    return mix(sqrt(linear*3.0),log2(linear-b)*a+c,right);
}

vec3 irr_glsl_oetf_AdobeRGB(in vec3 linear)
{
    return pow(linear,vec3(1.0/2.19921875));
}

vec3 irr_glsl_oetf_Gamma_2_2(in vec3 linear)
{
    return pow(linear,vec3(1.0/2.2));
}

vec3 irr_glsl_oetf_ACEScc(in vec3 linear)
{
    bvec3 mid = greaterThanEqual(linear,vec3(0.0));
    bvec3 right = greaterThanEqual(linear,vec3(0.000030517578125));
    return (log2(mix(vec3(0.0000152587890625),vec3(0.0),right)+linear*mix(vec3(0.0),mix(vec3(0.5),vec3(1.0),right),mid))+vec3(9.72))/17.52;
}

vec3 irr_glsl_oetf_ACEScct(in vec3 linear)
{
    bvec3 right = greaterThan(linear,vec3(0.0078125));
    return mix(10.5402377416545*linear+0.0729055341958355,(log2(linear)+vec3(9.72))/17.52,right);
}
#endif
)";
        }

    protected:
        irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
        {
            return {
                { std::regex{"encodeCIEXYZ\\.glsl"}, &getEncodeCIEXYZ },
                { std::regex{"decodeCIEXYZ\\.glsl"}, &getDecodeCIEXYZ },
                { std::regex{"EOTF\\.glsl"}, &getEOTF },
                { std::regex{"OETF\\.glsl"}, &getOETF },
            };
        }
};

}}

#endif//__IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__