// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_ASSET_MATERIAL_COMPILER_IR_H_INCLUDED__
#define __NBL_ASSET_MATERIAL_COMPILER_IR_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include <nbl/asset/ICPUImageView.h>
#include <nbl/asset/ICPUSampler.h>

namespace nbl::asset::material_compiler
{

// TODO: Merkle Tree / Hash Consing
class IR : public core::IReferenceCounted
{
    public:
        IR() : memMgr() {}

        struct node_handle_t
        {
            uint32_t byteOffset;

            inline bool operator==(const node_handle_t& other) const
            {
                return byteOffset==other.byteOffset;
            }
        };
        static inline constexpr node_handle_t invalid_node = {0xdeadbeefu};

        class INode
        {
            public:
                enum E_SYMBOL : uint8_t
                {
                    ES_GEOM_MODIFIER,
                    ES_EMISSION,
                    ES_OPACITY,
                    ES_BSDF,
                    ES_BSDF_COMBINER,
                    ES_UNINITIALIZED = 0xffu
                };

                using color_t = core::vector3df_SIMD;
                struct STextureSource
                {
                    core::smart_refctd_ptr<ICPUImageView> image;
                    core::smart_refctd_ptr<ICPUSampler> sampler;
                    float scale;

                    inline bool operator==(const STextureSource& rhs) const { return image == rhs.image && sampler == rhs.sampler && scale == rhs.scale; }
                };

                template <typename type_of_const>
                union SParameter
                {
                    inline SParameter() : texture{ nullptr,nullptr,core::nan<float>() } {}
                    inline SParameter(const type_of_const& c) : SParameter()
                    {
                        *this = c;
                    }
                    inline SParameter(type_of_const&& c) : SParameter(c) {}
                    inline SParameter(const STextureSource& c) : SParameter()
                    {
                        *this = t;
                    }
                    inline SParameter(STextureSource&& t) : SParameter()
                    {
                        *this = std::move(t);
                    }
                    inline SParameter(const SParameter<type_of_const>& other) : SParameter()
                    {
                        *this = other;
                    }
                    inline SParameter(SParameter<type_of_const>&& other) : SParameter()
                    {
                        *this = std::move(other);
                    }

                    inline bool isConstant() const { return core::isnan<float>(texture.scale); }
                    inline ~SParameter()
                    {
                        if (!isConstant())
                            texture.~STextureSource();
                    }

                    inline SParameter<type_of_const>& operator=(const type_of_const& c)
                    {
                        // drop pointers properly
                        if (!isConstant())
                            texture.~STextureSource();
                        constant = c;
                        texture.scale = core::nan<float>();

                        return *this;
                    }
                    inline SParameter<type_of_const>& operator=(const STextureSource& t)
                    {
                        return operator=(STextureSource(t));
                    }
                    inline SParameter<type_of_const>& operator=(STextureSource&& t)
                    {
                        // if wasn't a texture before, need to prevent contents being reinterpreted as pointers
                        if (isConstant())
                            memset(&texture, 0, offsetof(STextureSource, scale));
                        texture = std::move(t);
                        // just in case the scale was a NaN
                        if (isConstant())
                            texture.scale = 0.f;

                        return *this;
                    }
                    inline SParameter<type_of_const>& operator=(const SParameter<type_of_const>& rhs)
                    {
                        if (rhs.isConstant())
                            return operator=(rhs.constant);
                        else
                            return operator=(rhs.texture);
                    }
                    inline SParameter<type_of_const>& operator=(SParameter<type_of_const>&& rhs)
                    {
                        if (rhs.isConstant())
                            return operator=(rhs.constant);
                        else
                            return operator=(std::move(rhs.texture));
                    }

                    inline bool operator==(const SParameter<type_of_const>& rhs) const
                    {
                        if (isConstant())
                        {
                            if (rhs.isConstant())
                                return constant == rhs.constant;
                        }
                        else if (!rhs.isConstant())
                            return texture == rhs.texture;
                        return false;
                    }

                    type_of_const constant;
                    STextureSource texture;
                };

                //
                virtual E_SYMBOL getSymbol() const = 0;

                //
                virtual bool cloneInto(INode* dst) const = 0;

                //
                virtual uint32_t getChildCount() const = 0;
                
                //
                class children_range_t
                {
                    public:
                        const node_handle_t* m_begin;
                        const node_handle_t* m_end;

                        inline operator bool() const {return m_begin!=m_end;}

                        const node_handle_t* begin() const {return m_begin;}
                        const node_handle_t* end() const {return m_end;}
                };
                inline children_range_t getChildren() const
                {
                    auto begin = getChildrenArray();
                    return {begin,begin+getChildCount()};
                }

                //
                inline virtual ~INode()
                {
                    // clear out the vtable ptr
                    memset(this,0,sizeof(INode));
                }

                inline static bool alive(const INode* node)
                {
                    // check by checking vtable ptr
                    for (auto val=reinterpret_cast<const size_t*>(node++); ptrdiff_t(val)<ptrdiff_t(node); val++)
                        if (*val)
                            return true;
                    return false;
                }

            protected:
                friend class IR;

                //
                virtual size_t getSize() const = 0;
                inline size_t getChildrenStorageSize() const
                {
                    return sizeof(node_handle_t)*getChildCount();
                }

                //
                inline const node_handle_t* getChildrenArray() const
                {
                    return reinterpret_cast<const node_handle_t*>(reinterpret_cast<const uint8_t*>(this)+getSize()-getChildrenStorageSize());
                }
                inline node_handle_t* getChildrenArray()
                {
                    return const_cast<node_handle_t*>(const_cast<const INode*>(this)->getChildrenArray());
                }
        };

        template<class NodeType=INode>
        inline const NodeType* getNode(const node_handle_t handle) const
        {
            if (handle.byteOffset<memMgr.getAllocatedSize())
                return reinterpret_cast<const NodeType*>(memMgr.data()+handle.byteOffset);
            return nullptr;
        }
        template<class NodeType=INode>
        inline NodeType* getNode(const node_handle_t handle)
        {
            return const_cast<NodeType*>(const_cast<const IR*>(this)->getNode<NodeType>(handle));
        }

        template <typename NodeType, typename ...Args>
        inline node_handle_t allocNode(const uint32_t childCount, Args&& ...args)
        {
            auto retval = allocTmpNode<NodeType>(childCount,std::forward<Args>(args)...);
            if (retval.byteOffset<memMgr.getAllocatedSize())
                firstTmp.byteOffset = memMgr.getAllocatedSize();
            return retval;
        }

        inline bool addRootNode(const node_handle_t node)
        {
            if (node.byteOffset<firstTmp.byteOffset)
            {
                roots.push_back(node);
                return true;
            }
            assert(false);
            return false;
        }

        template <typename NodeType, typename ...Args>
        inline node_handle_t allocTmpNode(const uint32_t childCount, Args&& ...args)
        {
            auto retval = memMgr.alloc(NodeType::size_of(childCount));
            auto ptr = getNode(retval);
            if (ptr)
                new (ptr) NodeType(childCount,std::forward<Args>(args)...);
            return retval;
        }

        inline void deinitTmpNodes()
        {
            deleteRange(firstTmp);
            firstTmp.byteOffset = memMgr.getAllocatedSize();
        }

        inline node_handle_t copyNode(const INode* _rhs)
        {
            const size_t sz = _rhs->getSize();
            const auto allocation = memMgr.alloc(sz);
            auto copy = getNode(allocation);
            if (!copy)
                return invalid_node;
            if (!_rhs->cloneInto(copy))
            {
                memMgr.trimBackDown(allocation);
                return invalid_node;
            }
            firstTmp.byteOffset = memMgr.getAllocatedSize();
            return allocation;
        }
        template<class NodeT>
        inline NodeT* copyNode(const INode* _rhs)
        {
            return getNode<NodeT>(copyNode(_rhs));
        }

        //
        class ILeafNode : public INode
        {
            public:
                inline uint32_t getChildCount() const final override {return 0u;}
        };
        template<uint32_t kChildrenCount>
        class IFixedChildCountNode : public INode
        {
            public:
                inline uint32_t getChildCount() const final override
                {
                    return k_childrenCount;
                }

            protected:
                static inline constexpr uint32_t k_childrenCount = kChildrenCount;
        };
        class IVariableChildCountNode : public INode
        {
            public:
                inline uint32_t getChildCount() const final override
                {
                    return m_childrenCount;
                }

            protected:
                uint32_t m_childrenCount;
        };
        template<class FinalNodeT>
        class Finalizer final : public FinalNodeT
        {
            protected:
                template<typename... Args>
                Finalizer(const uint32_t childCount, Args&&... args) : FinalNodeT(std::forward<Args>(args)...)
                {
                    if constexpr (std::is_base_of_v<IVariableChildCountNode,FinalNodeT>)
                    {
                        IVariableChildCountNode::m_childrenCount = childCount;
                    }
                    else
                    if constexpr (!std::is_base_of_v<ILeafNode,FinalNodeT>)
                    {
                        assert(k_childrenCount==childCount);
                    }
                }

                static inline size_t size_of(const uint32_t childCount)
                {
                    if constexpr (std::is_base_of_v<IVariableChildCountNode,FinalNodeT>)
                        return FinalNodeT::size_of(childCount);
                    else
                        return sizeof(FinalNodeT)+sizeof(node_handle_t)*childCount;
                }

                inline size_t getSize() const override final {return size_of(getChildCount()); }

                inline bool cloneInto(INode* dst) const override final
                {
                    auto casted = dynamic_cast<FinalNodeT*>(dst);
                    if (!casted)
                        return false;
                    memcpy(casted->getChildrenArray(),INode::getChildrenArray(),INode::getChildrenStorageSize());
                    casted->operator=(*static_cast<const FinalNodeT*>(this));
                    return true;
                }
        };

        struct IGeomModifierNode final : public IFixedChildCountNode<1>
        {
            enum E_TYPE
            {
                ET_DISPLACEMENT,
                ET_HEIGHT,
                ET_NORMAL,
                ET_DERIVATIVE
            };
            /*
            enum E_SOURCE
            {
                ESRC_UV_FUNCTION,
                ESRC_TEXTURE
            };
            */

            IGeomModifierNode(E_TYPE t) : type(t) {}

            E_SYMBOL getSymbol() const override final { return ES_GEOM_MODIFIER; }

            E_TYPE type;
            //no other (than texture) source supported for now (uncomment in the future) [far future TODO]
            //TODO some union member for when source==ESRC_UV_FUNCTION, no idea what type
            //in fact when we want to translate MDL function of (u,v) into this IR we could just create an image being a 2D plot of this function with some reasonable quantization (pixel dimensions)
            //union {
            STextureSource texture;
            //};
        };
        using CGeomModifierNode = Finalizer<IGeomModifierNode>;

        struct IOpacityNode : IFixedChildCountNode<1> // TODO: kill? and replace by blend with transmission?
        {
            E_SYMBOL getSymbol() const override final { return ES_OPACITY; }

            SParameter<color_t> opacity;
        };
        using COpacityNode = Finalizer<IOpacityNode>;

        struct IBSDFCombinerNode
        {
            public:
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

            protected:
                IBSDFCombinerNode(E_TYPE t) : type(t) {}

                E_TYPE type;
        };
        struct IBSDFBlendNode : IFixedChildCountNode<2>, IBSDFCombinerNode
        {
            IBSDFBlendNode() : IBSDFCombinerNode(ET_WEIGHT_BLEND) {}

            E_SYMBOL getSymbol() const override final { return ES_BSDF_COMBINER; }

            SParameter<color_t> weight;
        };
        using CBSDFBlendNode = Finalizer<IBSDFBlendNode>;
        struct IBSDFMixNode final : IVariableChildCountNode, IBSDFCombinerNode
        {
            IBSDFMixNode() : IBSDFCombinerNode(ET_MIX) {}

            E_SYMBOL getSymbol() const override final { return ES_BSDF_COMBINER; }

            static inline size_t size_of(const uint32_t childrenCount)
            {
                return sizeof(IBSDFMixNode)+sizeof(float)*(childrenCount-1u)+sizeof(node_handle_t)*childrenCount;
            }

            inline IBSDFMixNode& operator=(const IBSDFMixNode& other)
            {
                IBSDFCombinerNode::operator=(other);
                std::copy_n(other.weights,other.getChildCount(),weights);
                return *this;
            }

            float weights[1];
        };
        using CBSDFMixNode = Finalizer<IBSDFMixNode>;

        struct IEmissionNode : ILeafNode
        {
            IEmissionNode() : ILeafNode() {}

            E_SYMBOL getSymbol() const override {return ES_EMISSION;}

            color_t intensity = color_t(1.f); // TODO: hoist?
        };
        using CEmissionNode = Finalizer<IEmissionNode>;

        struct IBSDFNode : ILeafNode
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

            inline IBSDFNode(E_TYPE t) : ILeafNode(), type(t) {}

            inline E_SYMBOL getSymbol() const override { return ES_BSDF; }

            E_TYPE type;
        };
        struct IMicrofacetBSDFNode : IBSDFNode
        {
            using IBSDFNode::IBSDFNode;

            virtual void setSmooth()
            {
                alpha_u = 0.f;
                alpha_v = alpha_u;
            }

            SParameter<float> alpha_u = 0.f;
            SParameter<float> alpha_v = 0.f;
        };
        struct IMicrofacetDiffuseBxDFBase : IMicrofacetBSDFNode
        {
            IMicrofacetDiffuseBxDFBase(const E_TYPE t) : IMicrofacetBSDFNode(t)
            {
                assert(t == ET_MICROFACET_DIFFTRANS || t == ET_MICROFACET_DIFFUSE);
            }
        };
        struct IMicrofacetDiffuseBSDFNode final : IMicrofacetDiffuseBxDFBase
        {
            IMicrofacetDiffuseBSDFNode() : IMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFUSE) {}

            SParameter<color_t> reflectance = color_t(1.f); // TODO: optimization, hoist Energy Loss Parameters out of BxDFs
        };
        using CMicrofacetDiffuseBSDFNode = Finalizer<IMicrofacetDiffuseBSDFNode>;
        struct IMicrofacetDifftransBSDFNode final : IMicrofacetDiffuseBxDFBase
        {
            IMicrofacetDifftransBSDFNode() : IMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFTRANS) {}

            SParameter<color_t> transmittance = color_t(0.5f); // TODO: optimization, hoist Energy Loss Parameters out of BxDFs
        };
        using CMicrofacetDifftransBSDFNode = Finalizer<IMicrofacetDifftransBSDFNode>;
        struct ICookTorranceBSDFNode : IMicrofacetBSDFNode
        {
            public:
                enum E_NDF : uint8_t
                {
                    ENDF_BECKMANN,
                    ENDF_GGX,
                    ENDF_ASHIKHMIN_SHIRLEY,
                    ENDF_PHONG
                };

                void setSmooth() override
                {
                    ndf = ENDF_GGX;
                }

                INode::color_t eta, etaK;
                E_NDF ndf = ENDF_GGX;

            protected:
                ICookTorranceBSDFNode(const IBSDFNode::E_TYPE t)
                    : IMicrofacetBSDFNode(t), eta(1.33f), etaK(0.f) {}
        };
        struct IMicrofacetSpecularBSDFNode : ICookTorranceBSDFNode
        {
            IMicrofacetSpecularBSDFNode() : ICookTorranceBSDFNode(ET_MICROFACET_SPECULAR) {}
        };
        using CMicrofacetSpecularBSDFNode = Finalizer<IMicrofacetSpecularBSDFNode>;
        struct IMicrofacetCoatingBSDFNode : ICookTorranceBSDFNode
        {
            IMicrofacetCoatingBSDFNode(const node_handle_t coated) : ICookTorranceBSDFNode(ET_MICROFACET_COATING) {}

            SParameter<color_t> thicknessSigmaA;
        };
        using CMicrofacetCoatingBSDFNode = Finalizer<IMicrofacetCoatingBSDFNode>;
        struct IMicrofacetDielectricBSDFNode : ICookTorranceBSDFNode
        {
            IMicrofacetDielectricBSDFNode() : ICookTorranceBSDFNode(ET_MICROFACET_DIELECTRIC) {}

            bool thin = false;
        };
        using CMicrofacetDielectricBSDFNode = Finalizer<IMicrofacetDielectricBSDFNode>;


    protected:
        inline ~IR()
        {
            deleteRange({ 0u });
        }


        core::vector<node_handle_t> roots;


        class SBackingMemManager
        {
                core::vector<uint8_t> mem;

            public:
                SBackingMemManager()
                {
                    mem.reserve(0x1u<<20);
                }

                inline uint8_t* data() {return const_cast<uint8_t*>(const_cast<const SBackingMemManager*>(this)->data());}
                inline const uint8_t* data() const {return reinterpret_cast<const uint8_t*>(mem.data());}

                inline node_handle_t alloc(size_t bytes)
                {
                    node_handle_t retval = {getAllocatedSize()};
                    mem.resize(mem.size()+bytes);
                    return retval;
                }

                inline uint32_t getAllocatedSize() const
                {
                    return static_cast<uint32_t>(mem.size());
                }

                inline void trimBackDown(const node_handle_t _end)
                {
                    mem.resize(_end.byteOffset);
                }
        };
        SBackingMemManager memMgr;
        // this stuff assumes the use of a linear allocator
        inline void deleteRange(node_handle_t begin)
        {
            for (auto offset=begin; offset.byteOffset<memMgr.getAllocatedSize();)
            {
                auto n = getNode(offset);
                offset.byteOffset += n->getSize();
                assert(INode::alive(n));
                n->~INode();
            }
            memMgr.trimBackDown(begin);
        }
        node_handle_t firstTmp = {0u};
};

}


namespace std
{

template <>
struct hash<nbl::asset::material_compiler::IR::node_handle_t>
{
    std::size_t operator()(const nbl::asset::material_compiler::IR::node_handle_t& handle) const
    {
        return std::hash<uint32_t>()(handle.byteOffset);
    }
};

}

#endif