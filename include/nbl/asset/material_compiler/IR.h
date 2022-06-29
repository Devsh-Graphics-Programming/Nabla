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

/**

TODO:
- Merkle Tree / Hash Consing

**/
class IR : public core::IReferenceCounted
{
    public:
        IR() : memMgr() {}

        struct node_handle_t
        {
            uint32_t byteOffset;
        };
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
                virtual size_t getSize() const = 0;

                //
                virtual bool cloneInto(INode* dst) const = 0;

                //
                virtual uint32_t getChildCount() const = 0;
                struct children_range_t
                {
                    public:
                        struct iterator_t
                        {
                            node_handle_t cursor;
                            uint32_t itemsTillEnd;

                            inline bool operator!=(const iterator_t rhs) const
                            {
                                return itemsTillEnd!=rhs.itemsTillEnd;
                            }
                        } m_begin;

                        inline operator bool() const {return m_begin.itemsTillEnd;}

                        iterator_t begin() {return m_begin;}
                        //const INode* const* begin() const { return array; }
                        iterator_t end() {return {0xdeadbeefu,0u};}
                        //const INode* const* end() const { return array + count; }
                };
                inline children_range_t getChildren() const
                {
                    return {getFirstChild(),getChildCount()};
                }

                //
                inline virtual ~INode()
                {
                    // clear out the vtable ptr
                    memset(this, 0, sizeof(INode));
                }

                inline static bool alive(const INode* node)
                {
                    for (auto val = reinterpret_cast<const size_t*>(node++); ptrdiff_t(val) < ptrdiff_t(node); val++)
                        if (*val)
                            return true;
                    return false;
                }

            protected:
                template<class ConstFinalNodeT>
                inline bool cloneInto_impl(INode* dst) const
                {
                    using FinalNodeT = std::remove_const_t<ConstFinalNodeT>;
                    static_assert(std::is_base_of_v<FinalNodeT, INode>);
                    auto casted = dynamic_cast<FinalNodeT*>(dst);
                    if (!casted)
                        return false;
                    casted->operator=(*static_cast<const FinalNodeT*>(this));
                    return true;
                }

                virtual node_handle_t getFirstChild() const = 0;
        };

        inline const INode* getNode(const node_handle_t handle) const
        {
            if (handle.byteOffset<memMgr.getAllocatedSize())
                return reinterpret_cast<const INode*>(memMgr.data()+handle.byteOffset);
            return nullptr;
        }

        template <typename NodeType, typename ...Args>
        inline node_handle_t allocNode(Args&& ...args)
        {
            auto retval = allocTmpNode<NodeType>(std::forward<Args>(args)...);
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
        inline node_handle_t allocTmpNode(Args&& ...args)
        {
            auto retval = memMgr.alloc(sizeof(NodeType));
            auto ptr = getNode(retval);
            if (ptr)
                new (ptr) NodeType(std::forward<Args>(args)...);
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
                return {0xdeadbeefu};
            if (!_rhs->cloneInto(copy))
            {
                memMgr.trimBackDown(allocation);
                return {0xdeadbeefu};
            }
            firstTmp.byteOffset = memMgr.getAllocatedSize();
            return allocation;
        }

        //
        class ILeafNode : public INode
        {
            public:
                inline uint32_t getChildCount() const final override {return 0u;}

            protected:
                inline node_handle_t getFirstChild() const final override {return {0xdeadbeefu};}
        };
        class IInternalNode : public INode
        {
            public:
                inline uint32_t getChildCount() const final override
                {
                    return m_childrenCount;
                }

            protected:
                inline IInternalNode(const node_handle_t _firstChild, const uint32_t _childrenCount)
                    : m_firstChild(_firstChild), m_childrenCount(_childrenCount) {}
                inline virtual ~IInternalNode() = default;

                inline node_handle_t getFirstChild() const final override
                {
                    return m_firstChild;
                }

                node_handle_t m_firstChild;
                uint32_t m_childrenCount;
        };

        struct CGeomModifierNode final : public IInternalNode
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

            CGeomModifierNode(E_TYPE t, const node_handle_t child) : IInternalNode(child,1u), type(t) {}

            E_SYMBOL getSymbol() const override { return ES_GEOM_MODIFIER; }

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            E_TYPE type;
            //no other (than texture) source supported for now (uncomment in the future) [far future TODO]
            //TODO some union member for when source==ESRC_UV_FUNCTION, no idea what type
            //in fact when we want to translate MDL function of (u,v) into this IR we could just create an image being a 2D plot of this function with some reasonable quantization (pixel dimensions)
            //union {
            STextureSource texture;
            //};
        };

        struct COpacityNode final : IInternalNode // TODO: kill? and replace by blend with transmission?
        {
            COpacityNode(const node_handle_t child) : IInternalNode(child,1u) {}

            E_SYMBOL getSymbol() const override { return ES_OPACITY; }

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            SParameter<color_t> opacity;
        };

        struct CBSDFCombinerNode : IInternalNode
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

                E_SYMBOL getSymbol() const override { return ES_BSDF_COMBINER; }
            protected:
                CBSDFCombinerNode(E_TYPE t, const node_handle_t firstChild, const uint32_t childCount)
                    : IInternalNode(firstChild,childCount), type(t) {}

                E_TYPE type;
        };
        struct CBSDFBlendNode final : CBSDFCombinerNode
        {
            CBSDFBlendNode(const node_handle_t firstChild) : CBSDFCombinerNode(ET_WEIGHT_BLEND,firstChild,2u) {}

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            SParameter<color_t> weight;
        };
        struct CBSDFMixNode final : CBSDFCombinerNode
        {
            CBSDFMixNode(const node_handle_t firstChild, const uint32_t childCount)
                : CBSDFCombinerNode(ET_MIX,firstChild,childCount) {}

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            float weights[16]; // TODO

                /*
                            template <typename ...Contents>
                            static inline children_array_t createChildrenArray(Contents... children)
                            {
                                children_array_t a;
                                const INode* ch[]{ children... };
                                memcpy(a.array, ch, sizeof(ch));
                                a.count = sizeof...(children);
                                return a;
                            }
                */
        };

        struct CEmissionNode final : ILeafNode
        {
            CEmissionNode() : ILeafNode() {}

            E_SYMBOL getSymbol() const override { return ES_EMISSION; }

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            color_t intensity = color_t(1.f); // TODO: hoist?
        };

        struct CBSDFNode : ILeafNode
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

            inline CBSDFNode(E_TYPE t) : ILeafNode(), type(t) {}

            inline E_SYMBOL getSymbol() const override { return ES_BSDF; }

            E_TYPE type;
        };
        struct CMicrofacetDiffuseBxDFBase : CBSDFNode
        {
            CMicrofacetDiffuseBxDFBase(const E_TYPE t) : CBSDFNode(t)
            {
                assert(t == ET_MICROFACET_DIFFTRANS || t == ET_MICROFACET_DIFFUSE);
            }

            void setSmooth()
            {
                alpha_u = 0.f;
                alpha_v = alpha_u;
            }

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            SParameter<float> alpha_u = 0.f;
            SParameter<float> alpha_v = 0.f;
        };
        struct CMicrofacetDiffuseBSDFNode final : CMicrofacetDiffuseBxDFBase
        {
            CMicrofacetDiffuseBSDFNode() : CMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFUSE) {}

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            SParameter<color_t> reflectance = color_t(1.f); // TODO: optimization, hoist Energy Loss Parameters out of BxDFs
        };
        struct CMicrofacetDifftransBSDFNode final : CMicrofacetDiffuseBxDFBase
        {
            CMicrofacetDifftransBSDFNode() : CMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFTRANS) {}

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            SParameter<color_t> transmittance = color_t(0.5f); // TODO: optimization, hoist Energy Loss Parameters out of BxDFs
        };
        struct CMicrofacetBSDFNode
        {
            public:
                enum E_NDF : uint8_t
                {
                    ENDF_BECKMANN,
                    ENDF_GGX,
                    ENDF_ASHIKHMIN_SHIRLEY,
                    ENDF_PHONG
                };

                void setSmooth(E_NDF _ndf = ENDF_GGX)
                {
                    ndf = _ndf;
                    alpha_u = 0.f;
                    alpha_v = alpha_u;
                }

                INode::SParameter<float> alpha_u = 0.f;
                INode::SParameter<float> alpha_v = 0.f;
                INode::color_t eta, etaK;
                E_NDF ndf = ENDF_GGX;

            protected:
                CMicrofacetBSDFNode() : eta(1.33f), etaK(0.f) {}
        };
        struct CMicrofacetSpecularBSDFNode final : CBSDFNode, CMicrofacetBSDFNode
        {
            CMicrofacetSpecularBSDFNode() : CBSDFNode(ET_MICROFACET_SPECULAR) {}

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }
        };
        struct CMicrofacetCoatingBSDFNode final : CBSDFNode, CMicrofacetBSDFNode
        {
            CMicrofacetCoatingBSDFNode(const node_handle_t coated) : CBSDFNode(ET_MICROFACET_COATING) {}

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            SParameter<color_t> thicknessSigmaA;
        };
        struct CMicrofacetDielectricBSDFNode final : CBSDFNode, CMicrofacetBSDFNode
        {
            CMicrofacetDielectricBSDFNode() : CBSDFNode(ET_MICROFACET_DIELECTRIC) {}

            inline size_t getSize() const override { return sizeof(*this); }

            inline bool cloneInto(INode* dst) const override
            {
                return cloneInto_impl<std::remove_pointer_t<decltype(this)>>(dst);
            }

            bool thin = false;
        };


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
        inline INode* getNode(const node_handle_t handle)
        {
            return const_cast<INode*>(const_cast<const IR*>(this)->getNode(handle));
        }
        // this stuff assumes the use of a linear allocator
        inline void deleteRange(const node_handle_t begin)
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

#endif