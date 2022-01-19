// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_MATERIAL_COMPILER_IR_H_INCLUDED__
#define __NBL_ASSET_MATERIAL_COMPILER_IR_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include <nbl/core/containers/refctd_dynamic_array.h>
#include <nbl/asset/ICPUImageView.h>
#include <nbl/asset/ICPUSampler.h>
#include <nbl/core/alloc/LinearAddressAllocator.h>

namespace nbl::asset::material_compiler
{

class IR : public core::IReferenceCounted
{
    class SBackingMemManager
    {
        _NBL_STATIC_INLINE_CONSTEXPR size_t INITIAL_MEM_SIZE = 1ull<<20;
        _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_MEM_SIZE = 1ull<<20;
        _NBL_STATIC_INLINE_CONSTEXPR size_t ALIGNMENT = _NBL_SIMD_ALIGNMENT;

        uint8_t* mem;
        size_t currSz;
        using addr_alctr_t = core::LinearAddressAllocator<uint32_t>;
        addr_alctr_t addrAlctr;

    public:
        SBackingMemManager() : mem(nullptr), currSz(INITIAL_MEM_SIZE), addrAlctr(nullptr, 0u, 0u, ALIGNMENT, MAX_MEM_SIZE) {
            mem = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(currSz, ALIGNMENT));
        }
        ~SBackingMemManager() {
            _NBL_ALIGNED_FREE(mem);
        }

        uint8_t* alloc(size_t bytes)
        {
            auto addr = addrAlctr.alloc_addr(bytes, ALIGNMENT);
            assert(addr != addr_alctr_t::invalid_address);
            //TODO reallocation will invalidate all pointers to nodes, so...
            //1) never reallocate (just have reasonably big buffer for nodes)
            //2) make some node_handle class that will work as pointer but is based on offset instead of actual address
            if (addr+bytes > currSz) {
                size_t newSz = currSz<<1;
                if (newSz > MAX_MEM_SIZE) {
                    addrAlctr.free_addr(addr, bytes);
                    return nullptr;
                }

                void* newMem = _NBL_ALIGNED_MALLOC(newSz, ALIGNMENT);
                memcpy(newMem, mem, currSz);
                _NBL_ALIGNED_FREE(mem);
                mem = reinterpret_cast<uint8_t*>(newMem);
                currSz = newSz;
            }

            return mem+addr;
        }

        uint32_t getAllocatedSize() const
        {
            return addrAlctr.get_allocated_size();
        }

        void freeLastAllocatedBytes(uint32_t _bytes)
        {
            assert(addrAlctr.get_allocated_size() >= _bytes);
            const uint32_t newCursor = addrAlctr.get_allocated_size() - _bytes;
            addrAlctr.reset(newCursor);
        }
    };

protected:
    ~IR()
    {
        //call destructors on all nodes
        for (auto* root : roots)
        {
            core::stack<decltype(root)> s;
            s.push(root);
            while (!s.empty())
            {
                auto* n = s.top();
                s.pop();
                for (auto* c : n->children)
                    s.push(c);

                if (!n->deinited) {
                    n->~INode();
                    n->deinited = true;
                }
            }
        }
    }

    template <typename NodeType, typename ...Args>
    NodeType* allocNode_impl(Args&& ...args)
    {
        uint8_t* ptr = memMgr.alloc(sizeof(NodeType));
        return new (ptr) NodeType(std::forward<Args>(args)...);
    }

public:
    IR() : memMgr() {}

    struct INode;

    void deinitTmpNodes()
    {
        for (INode* n : tmp)
            n->~INode();
        tmp.clear();
        memMgr.freeLastAllocatedBytes(tmpSize);
        tmpSize = 0u;
    }

    void addRootNode(INode* node)
    {
        roots.push_back(node);
    }

    template <typename NodeType, typename ...Args>
    NodeType* allocNode(Args&& ...args)
    {
        tmpSize = 0u;
        return allocNode_impl<NodeType>(std::forward<Args>(args)...);
    }
    template <typename NodeType, typename ...Args>
    NodeType* allocRootNode(Args&& ...args)
    {
        auto* root = allocNode<NodeType>(std::forward<Args>(args)...);
        addRootNode(root);
        return root;
    }
    template <typename NodeType, typename ...Args>
    NodeType* allocTmpNode(Args&& ...args)
    {
        const uint32_t cursor = memMgr.getAllocatedSize();
        auto* node = allocNode_impl<NodeType>(std::forward<Args>(args)...);
        tmp.push_back(node);
        tmpSize += (memMgr.getAllocatedSize() - cursor);
        return node;
    }

    struct INode
    {
        enum E_SYMBOL
        {
            ES_GEOM_MODIFIER,
            ES_EMISSION,
            ES_OPACITY,
            ES_BSDF,
            ES_BSDF_COMBINER
        };

        struct STextureSource {
            core::smart_refctd_ptr<ICPUImageView> image;
            core::smart_refctd_ptr<ICPUSampler> sampler;
            float scale;

            bool operator==(const STextureSource& rhs) const { return image==rhs.image && sampler==rhs.sampler && scale==rhs.scale; }
        };

        enum E_PARAM_SOURCE
        {
            EPS_CONSTANT,
            EPS_TEXTURE
        };

        template <typename type_of_const>
        struct SParameter
        {
            //destructor of actually used variant member has to be called by hand!
            template <typename type_of_const2>
            union UTextureOrConstant {
                UTextureOrConstant() : texture{ nullptr,nullptr,0.f } {}
                ~UTextureOrConstant() {}

                type_of_const2 constant;
                STextureSource texture;
            };
            using TextureOrConstant = UTextureOrConstant<type_of_const>;

            ~SParameter() {
                if (source==EPS_TEXTURE)
                    value.texture.~STextureSource();
            }

            SParameter() = default;
            SParameter<type_of_const>& operator=(const type_of_const& c)
            {
                source = EPS_CONSTANT;
                value.constant = c;

                return *this;
            }
            SParameter(const type_of_const& c)
            {
                *this = c;
            }
            SParameter<type_of_const>& operator=(const STextureSource& t)
            {
                source = EPS_TEXTURE;
                //making sure smart_refctd_ptr assignment wont try to drop() -- .value is union
                value.constant = type_of_const(0);
                value.texture = t;

                return *this;
            }
            SParameter(const STextureSource& t)
            {
                *this = t;
            }
            SParameter<type_of_const>& operator=(STextureSource&& t)
            {
                source = EPS_TEXTURE;
                //making sure smart_refctd_ptr assignment wont try to drop() -- .value is union
                value.constant = type_of_const{};
                value.texture = std::move(t);

                return *this;
            }
            SParameter(STextureSource&& t)
            {
                *this = std::move(t);
            }
            SParameter(const SParameter<type_of_const>& other)
            {
                *this = other;
            }
            SParameter<type_of_const>& operator=(const SParameter<type_of_const>& rhs)
            {
                const auto prevSource = source;
                source = rhs.source;
                if (source == EPS_CONSTANT) {
                    if (prevSource == EPS_TEXTURE)
                        value.texture.~STextureSource();
                    value.constant = rhs.value.constant;
                }
                else {
                    if (prevSource == EPS_CONSTANT)
                        value.constant = type_of_const();
                    value.texture = rhs.value.texture;
                }

                return *this;
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
            TextureOrConstant value;
        };

        _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_CHILDREN = 16ull;
        /*
        * Possible TODO:
        * we could implement the children array in the future as N nodes allocated just after this one (one would only need the child count)... 
            but this would only be possible if the nodes were uniform bytesize.
        That way there would be no artificial limit on max children in our IR (backends will still have limits)
        */
        struct children_array_t {
            INode* array[MAX_CHILDREN] {};
            size_t count = 0ull;

            inline bool operator!=(const children_array_t& rhs) const
            {
                if (count != rhs.count)
                    return true;

                for (uint32_t i = 0u; i < count; ++i)
                    if (array[i]!=rhs.array[i])
                        return true;
                return false;
            }
            inline bool operator==(const children_array_t& rhs) const
            {
                return !operator!=(rhs);
            }

            inline bool find(E_SYMBOL s, size_t* ix = nullptr) const 
            { 
                auto found = std::find_if(begin(),end(),[s](const INode* child){return child->symbol==s;});
                if (found != (array+count))
                {
                    if (ix)
                        ix[0] = std::distance(array,const_cast<INode*const*>(found));
                    return true;
                }
                return false;
            }

            operator bool() const { return count!=0ull; }

            INode** begin() { return array; }
            const INode*const* begin() const { return array; }
            INode** end() { return array+count; }
            const INode*const* end() const { return array+count; }

            inline INode*& operator[](size_t i) { assert(i<count); return array[i]; }
            inline const INode* const& operator[](size_t i) const { assert(i<count); return array[i]; }
        };
        template <typename ...Contents>
        static inline children_array_t createChildrenArray(Contents... children) 
        { 
            children_array_t a;
            const INode* ch[]{ children... };
            memcpy(a.array, ch, sizeof(ch));
            a.count = sizeof...(children);
            return a; 
        }

        using color_t = core::vector3df_SIMD;

        explicit INode(E_SYMBOL s) : symbol(s) {}
        virtual ~INode() = default;

        // TODO: Why does every INode have children!? Leaf BxDFs do not need this!
        children_array_t children;
        E_SYMBOL symbol;
        bool deinited = false;
    };

    INode* copyNode(const INode* _rhs)
    {
        INode* node = nullptr;
        switch (_rhs->symbol)
        {
        case INode::ES_GEOM_MODIFIER:
        {
            auto* rhs = static_cast<const CGeomModifierNode*>(_rhs);
            node = allocNode<CGeomModifierNode>(rhs->type);
            *static_cast<CGeomModifierNode*>(node) = *rhs;
        }
            break;
        case INode::ES_EMISSION:
        {
            auto* rhs = static_cast<const CEmissionNode*>(_rhs);
            node = allocNode<CEmissionNode>();
            *static_cast<CEmissionNode*>(node) = *rhs;
        }
            break;
        case INode::ES_OPACITY:
        {
            auto* rhs = static_cast<const COpacityNode*>(_rhs);
            node = allocNode<COpacityNode>();
            *static_cast<COpacityNode*>(node) = *rhs;
        }
            break;
        case INode::ES_BSDF:
        {
            auto* rhs_bsdf = static_cast<const CBSDFNode*>(_rhs);

            switch (rhs_bsdf->type)
            {
            case CBSDFNode::ET_MICROFACET_DIFFTRANS:
            {
                auto* rhs = static_cast<const CMicrofacetDifftransBSDFNode*>(_rhs);
                node = allocNode<CMicrofacetDifftransBSDFNode>();
                *static_cast<CMicrofacetDifftransBSDFNode*>(node) = *rhs;
            }
            break;
            case CBSDFNode::ET_MICROFACET_DIFFUSE:
            {
                auto* rhs = static_cast<const CMicrofacetDiffuseBSDFNode*>(_rhs);
                node = allocNode<CMicrofacetDiffuseBSDFNode>();
                *static_cast<CMicrofacetDiffuseBSDFNode*>(node) = *rhs;
            }
            break;
            case CBSDFNode::ET_MICROFACET_SPECULAR:
            {
                auto* rhs = static_cast<const CMicrofacetSpecularBSDFNode*>(_rhs);
                node = allocNode<CMicrofacetSpecularBSDFNode>();
                *static_cast<CMicrofacetSpecularBSDFNode*>(node) = *rhs;
            }
            break;
            case CBSDFNode::ET_MICROFACET_COATING:
            {
                auto* rhs = static_cast<const CMicrofacetCoatingBSDFNode*>(_rhs);
                node = allocNode<CMicrofacetCoatingBSDFNode>();
                *static_cast<CMicrofacetCoatingBSDFNode*>(node) = *rhs;
            }
            break;
            case CBSDFNode::ET_MICROFACET_DIELECTRIC:
            {
                auto* rhs = static_cast<const CMicrofacetDielectricBSDFNode*>(_rhs);
                node = allocNode<CMicrofacetDielectricBSDFNode>();
                *static_cast<CMicrofacetDielectricBSDFNode*>(node) = *rhs;
            }
            break;
            default:
            {
                node = allocNode<CBSDFNode>(rhs_bsdf->type);
                *static_cast<CBSDFNode*>(node) = *rhs_bsdf;
            }
            }
        }
            break;
        case INode::ES_BSDF_COMBINER:
        {
            auto* rhs_combiner = static_cast<const CBSDFCombinerNode*>(_rhs);
            switch (rhs_combiner->type)
            {
            case CBSDFCombinerNode::ET_WEIGHT_BLEND:
            {
                auto* rhs = static_cast<const CBSDFBlendNode*>(_rhs);
                node = allocNode<CBSDFBlendNode>();
                *static_cast<CBSDFBlendNode*>(node) = *rhs;
            }
            break;
            case CBSDFCombinerNode::ET_MIX:
            {
                auto* rhs = static_cast<const CBSDFMixNode*>(_rhs);
                node = allocNode<CBSDFMixNode>();
                *static_cast<CBSDFMixNode*>(node) = *rhs;
            }
            break;
            default:
            {
                node = allocNode<CBSDFCombinerNode>(rhs_combiner->type);
                *static_cast<CBSDFCombinerNode*>(node) = *rhs_combiner;
            }
            break;
            }
        }
            break;
        default:
            assert(false);
            return nullptr;
        }

        return node;
    }

    struct CGeomModifierNode : public INode
    {
        enum E_TYPE
        {
            ET_DISPLACEMENT,
            ET_HEIGHT,
            ET_NORMAL,
            ET_DERIVATIVE
        };
        enum E_SOURCE
        {
            ESRC_UV_FUNCTION,
            ESRC_TEXTURE
        };

        CGeomModifierNode(E_TYPE t) : INode(ES_GEOM_MODIFIER), type(t) {}

        E_TYPE type;
        //no other (than texture) source supported for now (uncomment in the future) [far future TODO]
        //E_SOURCE source;
        //TODO some union member for when source==ESRC_UV_FUNCTION, no idea what type
        //in fact when we want to translate MDL function of (u,v) into this IR we could just create an image being a 2D plot of this function with some reasonable quantization (pixel dimensions)
        //union {
            STextureSource texture;
        //};
    };

    struct CEmissionNode : INode
    {
        CEmissionNode() : INode(ES_EMISSION) {}

        color_t intensity = color_t(1.f);
    };

    struct COpacityNode : INode
    {
        COpacityNode() : INode(ES_OPACITY) {}

        SParameter<color_t> opacity;
    };

    struct CBSDFCombinerNode : INode
    {
        enum E_TYPE
        {
            //mix of N BSDFs
            ET_MIX,
            //blend of 2 BSDFs weighted by constant or texture
            ET_WEIGHT_BLEND,
            //for support of nvidia MDL's df::fresnel_layer
            ET_LOL_MDL_SUX_BROKEN_FRESNEL_BLEND,
            //blend of 2 BSDFs weighted by custom direction-based curve
            ET_CUSTOM_CURVE_BLEND
        };

        E_TYPE type;

        CBSDFCombinerNode(E_TYPE t) : INode(ES_BSDF_COMBINER), type(t) {}
    };
    struct CBSDFBlendNode : CBSDFCombinerNode
    {
        CBSDFBlendNode() : CBSDFCombinerNode(ET_WEIGHT_BLEND) {}

        SParameter<color_t> weight;
    };
    struct CBSDFMixNode : CBSDFCombinerNode
    {
        CBSDFMixNode() : CBSDFCombinerNode(ET_MIX) {}

        float weights[MAX_CHILDREN];
    };

    struct CBSDFNode : INode
    {
        enum E_TYPE
        {
            ET_MICROFACET_DIFFTRANS,
            ET_MICROFACET_DIFFUSE,
            ET_MICROFACET_SPECULAR,
            ET_MICROFACET_COATING,
            ET_MICROFACET_DIELECTRIC,
            ET_DELTA_TRANSMISSION
            //ET_SHEEN,
        };

        CBSDFNode(E_TYPE t) :
            INode(ES_BSDF),
            type(t),
            eta(1.33f),
            etaK(0.f)
        {}

        E_TYPE type;
        // TODO: why does this base class have IoR!? Diffuse inherits from this!!!
        color_t eta, etaK;
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
        // TODO: Remove, the NDF fixes the geometrical shadowing and masking function.
        enum E_SHADOWING_TERM
        {
            EST_SMITH,
            EST_VCAVITIES
        };

        CMicrofacetSpecularBSDFNode() : CBSDFNode(ET_MICROFACET_SPECULAR) {}

        void setSmooth(E_NDF _ndf = ENDF_GGX)
        {
            ndf = _ndf;
            alpha_u.source = EPS_CONSTANT;
            alpha_u.value.constant = 0.f;
            alpha_v = alpha_u;
        }

        E_NDF ndf = ENDF_GGX;
        E_SHADOWING_TERM shadowing = EST_SMITH;
        SParameter<float> alpha_u = 0.f;
        SParameter<float> alpha_v = 0.f;

    protected:
        CMicrofacetSpecularBSDFNode(E_TYPE t) : CBSDFNode(t) {}
    };
    struct CMicrofacetDiffuseBxDFBase : CBSDFNode
    {
        CMicrofacetDiffuseBxDFBase(E_TYPE t) : CBSDFNode(t) {}

        void setSmooth()
        {
            alpha_u.source = EPS_CONSTANT;
            alpha_u.value.constant = 0.f;
            alpha_v = alpha_u;
        }

        SParameter<float> alpha_u = 0.f;
        SParameter<float> alpha_v = 0.f;
    };
    struct CMicrofacetDiffuseBSDFNode : CMicrofacetDiffuseBxDFBase
    {
        CMicrofacetDiffuseBSDFNode() : CMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFUSE) {}

        SParameter<color_t> reflectance = color_t(1.f);
    };
    struct CMicrofacetDifftransBSDFNode : CMicrofacetDiffuseBxDFBase
    {
        CMicrofacetDifftransBSDFNode() : CMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFTRANS) {}

        SParameter<color_t> transmittance = color_t(0.5f);
    };
    struct CMicrofacetCoatingBSDFNode : CMicrofacetSpecularBSDFNode
    {
        CMicrofacetCoatingBSDFNode() : CMicrofacetSpecularBSDFNode(ET_MICROFACET_COATING) {}

        SParameter<color_t> thicknessSigmaA;
    };
    struct CMicrofacetDielectricBSDFNode : CMicrofacetSpecularBSDFNode
    {
        CMicrofacetDielectricBSDFNode() : CMicrofacetSpecularBSDFNode(ET_MICROFACET_DIELECTRIC) {}
        bool thin = false;
    };

    SBackingMemManager memMgr;
    core::vector<INode*> roots;

    core::vector<INode*> tmp;
    uint32_t tmpSize = 0u;
};

}

#endif