#ifndef __IRR_MATERIAL_COMPILER_IR_H_INCLUDED__
#define __IRR_MATERIAL_COMPILER_IR_H_INCLUDED__

#include <irr/core/IReferenceCounted.h>
#include <irr/core/memory/refctd_dynamic_array.h>
#include <irr/asset/ICPUImageView.h>
#include <irr/asset/ICPUSampler.h>

namespace irr {
namespace asset {
namespace material_compiler
{

class IR : public core::IReferenceCounted
{
public:
    struct INode : public core::IReferenceCounted
    {
        virtual ~INode() = default;

        enum E_SYMBOL
        {
            ES_MATERIAL,
            ES_GEOM_MODIFIER,
            ES_FRONT_SURFACE,
            ES_BACK_SURFACE,
            ES_EMISSION,
            ES_BSDF,
            ES_BSDF_COMBINER
        };
        enum E_PARAM_SOURCE
        {
            EPS_CONSTANT,
            EPS_TEXTURE
        };

        struct STextureSource {
            core::smart_refctd_ptr<ICPUImageView> image;
            core::smart_refctd_ptr<ICPUSampler> sampler;
            float scale;

            bool operator==(const STextureSource& rhs) const { return image==rhs.image && sampler==rhs.sampler && scale==rhs.scale; }
        };
        //destructor of actually used variant member has to be called by hand!
        template <typename type_of_const>
        union UTextureOrConstant {
            UTextureOrConstant() : texture{nullptr,nullptr,0.f} {}
            ~UTextureOrConstant() {}

            type_of_const constant;
            STextureSource texture;
        };
        template <typename type_of_const>
        struct SParameter
        {
            ~SParameter() {
                if (source==EPS_TEXTURE)
                    value.texture.~STextureSource();
            }

            SParameter<type_of_const>& operator=(const SParameter<type_of_const>& rhs)
            {
                source = rhs.source;
                if (source == EPS_CONSTANT)
                    value.constant = rhs.value.constant;
                else
                    value.texture = rhs.value.texture;
            }

            bool operator==(const SParameter<type_of_const>& rhs) const {
                if (source!=rhs.source)
                    return false;
                switch (source)
                {
                case EPS_CONSTANT:
                    return value.constant==value.constant;
                case EPS_TEXTURE:
                    return value.texture==value.texture;
                default: return false;
                }
            }

            E_PARAM_SOURCE source;
            UTextureOrConstant<type_of_const> value;
        };

        using children_smart_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<INode>>;

        using color_t = core::vector3df_SIMD;

        INode(E_SYMBOL s) : symbol(s) {}

        children_smart_array_t children;
        E_SYMBOL symbol;
    };

    struct CGeomModifierNode : public INode
    {
        enum E_TYPE
        {
            ET_DISPLACEMENT,
            ET_HEIGHT,
            ET_NORMAL
        };
        enum E_SOURCE
        {
            ESRC_UV_FUNCTION,
            ESRC_TEXTURE
        };

        CGeomModifierNode(E_TYPE t) : INode(ES_GEOM_MODIFIER), type(t) {}

        E_TYPE type;
        E_SOURCE source;
        //TODO some union member for when source==ESRC_UV_FUNCTION, no idea what type
        //in fact when we want to translate MDL function of (u,v) into this IR we could just create an image being a 2D plot of this function with some reasonable quantization (pixel dimensions)
        //union {
            STextureSource texture;
        //};
    };

    struct CEmissionNode : INode
    {
        CEmissionNode() : INode(ES_EMISSION) {}

        float intensity;
    };

    struct CBSDFCombinerNode : INode
    {
        enum E_TYPE
        {
            //mix of N BSDFs
            ET_MIX,
            //blend of 2 BSDFs weighted by constant or texture
            ET_WEIGHT_BLEND,
            //blend of 2 BSDFs weighted by fresnel
            ET_FRESNEL_BLEND,
            //blend of 2 BSDFs weighted by custom direction-based curve
            ET_CUSTOM_CURVE_BLEND,
            ET_DIFFUSE_AND_SPECULAR
        };

        E_TYPE type;

        CBSDFCombinerNode(E_TYPE t) : INode(ES_BSDF_COMBINER), type(t) {}
    };
    struct CBSDFBlendNode : CBSDFCombinerNode
    {
        CBSDFBlendNode() : CBSDFCombinerNode(ET_WEIGHT_BLEND) {}

        SParameter<float> weight;
    };
    struct CBSDFMixNode : CBSDFCombinerNode
    {
        CBSDFMixNode() : CBSDFCombinerNode(ET_MIX) {}

        using weights_t = core::smart_refctd_dynamic_array<float>;
        weights_t weights;
    };

    struct CBSDFNode : INode
    {
        enum E_TYPE
        {
            ET_DIFFTRANS,
            ET_SPECULAR_DELTA,
            ET_MICROFACET_DIFFUSE,
            ET_MICROFACET_SPECULAR,
            ET_SHEEN,
            ET_COATING
        };

        CBSDFNode(E_TYPE t) : INode(ES_BSDF), type(t) {}

        E_TYPE type;
    };
    struct CMicrofacetSpecularBSDFNode : CBSDFNode
    {
        enum E_NDF
        {
            ENDF_BECKMANN,
            ENDF_GGX,
            ENDF_ASHIKHMIN_SHIRLEY,
            ENDF_PHONG
        };
        enum E_SHADOWING_TERM
        {
            EST_SMITH,
            EST_VCAVITIES
        };
        enum E_SCATTER_MODE
        {
            ESM_REFLECT,
            ESM_TRANSMIT
        };

        CMicrofacetSpecularBSDFNode(E_TYPE t = ET_MICROFACET_SPECULAR) : CBSDFNode(t) {}

        E_NDF ndf;
        E_SHADOWING_TERM shadowing;
        E_SCATTER_MODE scatteringMode;
        SParameter<float> alpha_u;
        SParameter<float> alpha_v;
    };
    struct CMicrofacetDiffuseBSDFNode : CBSDFNode
    {
        CMicrofacetDiffuseBSDFNode() : CBSDFNode(ET_MICROFACET_DIFFUSE) {}

        SParameter<color_t> reflectance;
        SParameter<float> alpha_u;
        SParameter<float> alpha_v;
    };
    struct CDifftransBSDFNode : CBSDFNode
    {
        CDifftransBSDFNode() : CBSDFNode(ET_DIFFTRANS) {}

        SParameter<color_t> transmittance;
    };
    struct CCoatingBSDFNode : CMicrofacetSpecularBSDFNode
    {
        CCoatingBSDFNode() : CMicrofacetSpecularBSDFNode(ET_COATING) {}

        SParameter<color_t> sigmaA;
        float thickness;
    };

    core::smart_refctd_ptr<INode> root;
};

}}}

#endif