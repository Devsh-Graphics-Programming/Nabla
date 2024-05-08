#ifndef _NBL_ASSET_I_RENDERPASS_H_INCLUDED_
#define _NBL_ASSET_I_RENDERPASS_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/IImage.h"
#include "nbl/asset/ECommonEnums.h"

#include <compare>

namespace nbl::asset
{

class IRenderpass
{
    public:
        enum class LOAD_OP : uint8_t
        {
            LOAD = 0,
            CLEAR,
            DONT_CARE,
            UNDEFINED
        };
        enum class STORE_OP: uint8_t
        {
            STORE = 0,
            DONT_CARE,
            UNDEFINED
        };
        // For all arrays here we use ArrayNameEnd terminator instead of specifying the count
        struct SCreationParams
        {
                template<typename op_t>
                using Op = op_t;

            public:
                template<typename op_t>
                struct DepthStencilOp final
                {
                    op_t depth : 2 = op_t::DONT_CARE;
                    op_t stencil : 2 = op_t::UNDEFINED;


                    auto operator<=>(const DepthStencilOp<op_t>&) const = default;

                    inline op_t actualStencilOp() const
                    {
                        return stencil!=op_t::UNDEFINED ? stencil:depth;
                    }
                };
                // This is the best we can do, because we can only tell attachments apart by their format.
                // We can't tell the difference between attachments used for rendering, resolve, or input 
                // the attachments limits (1 depth/stencil, MaxColorAttachments color and resolve) only
                // apply PER SUBPASS so overall we can have as many attachments as we like and they
                // can be reused as different types in the subpasses. Except of course not being able to use
                // a depth/stencil attachment in place of a colour one, so we trivially satisfy:
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pDepthStencilAttachment-04440
                struct SAttachmentDescriptionBase
                {
                    E_FORMAT format = EF_UNKNOWN;
                    IImage::E_SAMPLE_COUNT_FLAGS samples : 6 = IImage::ESCF_1_BIT;
                    uint8_t mayAlias : 1 = false;

                    auto operator<=>(const SAttachmentDescriptionBase&) const = default;
                };
                template<typename Layout, template<typename> class op_t>
                struct SAttachmentDescription : SAttachmentDescriptionBase
                {
                    op_t<LOAD_OP> loadOp = {};
                    op_t<STORE_OP> storeOp = {};
                    Layout initialLayout = {};
                    Layout finalLayout = {};

                    auto operator<=>(const SAttachmentDescription&) const = default;
                };
                struct SDepthStencilAttachmentDescription final : SAttachmentDescription<IImage::SDepthStencilLayout,DepthStencilOp>
                {
                    bool valid() const;
                };
                struct SColorAttachmentDescription final : SAttachmentDescription<IImage::LAYOUT,Op>
                {
                    bool valid() const;
                };
                // The arrays pointed to by this array must be terminated by `DepthStencilAttachmentsEnd` value, which implicitly satisfies a few VUIDs
                constexpr static inline SDepthStencilAttachmentDescription DepthStencilAttachmentsEnd = {};
                const SDepthStencilAttachmentDescription* depthStencilAttachments = &DepthStencilAttachmentsEnd;
                // The arrays pointed to by this array must be terminated by `ColorAttachmentsEnd` value, which implicitly satisfies a few VUIDs
                constexpr static inline SColorAttachmentDescription ColorAttachmentsEnd = {};
                const SColorAttachmentDescription* colorAttachments = &ColorAttachmentsEnd;

                struct SSubpassDescription final
                {
                    constexpr static inline uint32_t AttachmentUnused = 0xffFFffFFu;
                    enum class FLAGS : uint8_t
                    {
                        NONE = 0x00,
                        PER_VIEW_ATTRIBUTES_BIT = 0x01,
                        PER_VIEW_POSITION_X_ONLY_BIT = 0x02
                        // TODO: VK_EXT_rasterization_order_attachment_access
                    };

                    template<typename layout_t>
                    struct SAttachmentRef final
                    {
                        public:
                            constexpr static inline bool IsDepthStencil = std::is_base_of_v<IImage::SDepthStencilLayout,layout_t>;
                            using description_t = std::conditional_t<IsDepthStencil,SDepthStencilAttachmentDescription,SColorAttachmentDescription>;

                            // If you leave the `attachmentIndex` as default then it means its not being used
                            uint32_t attachmentIndex = AttachmentUnused;
                            layout_t layout = {};

                            inline bool used() const {return attachmentIndex!=AttachmentUnused;}

                            auto operator<=>(const SAttachmentRef<layout_t>&) const = default;

                            template<bool InputAttachment>
                            bool valid(const description_t* descs, const uint32_t attachmentCount) const;
                    };
                    using SDepthStencilAttachmentRef = SAttachmentRef<IImage::SDepthStencilLayout>;
                    using SColorAttachmentRef = SAttachmentRef<IImage::LAYOUT>;
                    struct SInputAttachmentRef final
                    {
                        // we can tell which one you meant by the aspectMask
                        union
                        {
                            SDepthStencilAttachmentRef asDepthStencil = {};
                            SColorAttachmentRef asColor;
                        };
                        core::bitflag<IImage::E_ASPECT_FLAGS> aspectMask = IImage::E_ASPECT_FLAGS::EAF_NONE;

                        inline bool operator!=(const SInputAttachmentRef& other) const
                        {
                            if (aspectMask!=other.aspectMask)
                                return true;

                            if (aspectMask==IImage::EAF_COLOR_BIT)
                                return asColor!=other.asColor;
                            else
                                return asDepthStencil!=other.asDepthStencil;
                        }

                        inline bool isColor() const {return aspectMask==IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;}

                        inline bool used() const
                        {
                            return isColor() ? asColor.used():asDepthStencil.used();
                        }

                        bool valid(const SCreationParams& params, const uint32_t depthStencilAttachmentCount, const uint32_t colorAttachmentCount) const;
                    };
                    template<class attachment_ref_t>
                    struct SRenderAttachmentsRef
                    {
                        constexpr static inline bool IsDepthStencil = attachment_ref_t::IsDepthStencil;

                        attachment_ref_t render = {};
                        attachment_ref_t resolve = {};

                        auto operator<=>(const SRenderAttachmentsRef<attachment_ref_t>&) const = default;

                        bool valid(const typename attachment_ref_t::description_t* descs, const uint32_t attachmentCount) const;
                    };
                    struct SDepthStencilAttachmentsRef final : SRenderAttachmentsRef<SDepthStencilAttachmentRef>
                    {
                        enum class RESOLVE_MODE : uint8_t
                        {
                            NONE = 0,
                            SAMPLE_ZERO_BIT = 0x00000001,
                            AVERAGE_BIT = 0x00000002,
                            MIN_BIT = 0x00000004,
                            MAX_BIT = 0x00000008
                        };
                        struct ResolveMode
                        {
                            auto operator<=>(const ResolveMode&) const = default;

                            RESOLVE_MODE depth : 4 = RESOLVE_MODE::NONE;
                            RESOLVE_MODE stencil : 4 = RESOLVE_MODE::NONE;
                        } resolveMode;

                        bool valid(const SDepthStencilAttachmentDescription* depthStencilAttachments, const uint32_t depthStencilAttachmentCount) const;
                    };
                    using SColorAttachmentsRef = SRenderAttachmentsRef<SColorAttachmentRef>;


                    //
                    bool operator!=(const SSubpassDescription&) const;
                    inline bool operator==(const SSubpassDescription& rhs) const {return !((*this)!=rhs);}

                    //
                    bool valid(const SCreationParams& params, const uint32_t depthStencilAttachmentCount, const uint32_t colorAttachmentCount) const;


                    //! Field ordering prioritizes ergonomics
                    SDepthStencilAttachmentsRef depthStencilAttachment = {};

                    static inline constexpr auto MaxColorAttachments = 8u;
                    SColorAttachmentsRef colorAttachments[MaxColorAttachments] = {};

                    // The arrays pointed to by this array must be terminated by `InputAttachmentsEnd` value
                    constexpr static inline SInputAttachmentRef InputAttachmentsEnd = {};
                    const SInputAttachmentRef* inputAttachments = &InputAttachmentsEnd;

                    struct SPreserveAttachmentRef
                    {
                        inline bool operator!=(const SPreserveAttachmentRef& other) const
                        {
                            return color!=other.color || index!=other.index;
                        }

                        // for sorting
                        inline bool operator<(const SPreserveAttachmentRef& other) const
                        {
                            return *reinterpret_cast<const uint32_t*>(this)<reinterpret_cast<const uint32_t&>(other);
                        }

                        uint32_t color : 1;
                        uint32_t index : 31;
                    };
                    // The arrays pointed to by this array must be terminated by `PreserveAttachmentsEnd` value, which implicitly satisfies:
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-03073
                    constexpr static inline SPreserveAttachmentRef PreserveAttachmentsEnd = {0,AttachmentUnused};
                    const SPreserveAttachmentRef* preserveAttachments = &PreserveAttachmentsEnd; // TODO: redesign

                    // TODO: shading rate attachment
                    // TODO: rasterization sample count & downgrade viewMask to 8bit

                    uint32_t viewMask = 0u;
                    core::bitflag<FLAGS> flags = FLAGS::NONE;
                    // Do not expose because we don't support Subpass Shading
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pipelineBindPoint-04953
                    //E_PIPELINE_BIND_POINT pipelineBindPoint : 2 = EPBP_GRAPHICS;
                };
                constexpr static inline SSubpassDescription SubpassesEnd = {};
                const SSubpassDescription* subpasses = &SubpassesEnd;

                struct SSubpassDependency final
                {
                    constexpr static inline uint32_t External = ~0u;
                    enum class FLAGS : uint8_t
                    {
                        NONE = 0x00u,
                        BY_REGION = 0x01u,
                        VIEW_LOCAL = 0x02u,
                        DEVICE_GROUP = 0x04u,
                        FEEDBACK_LOOP = 0x08u
                    };

                    uint32_t srcSubpass = External;
                    uint32_t dstSubpass = External;
                    SMemoryBarrier memoryBarrier = {};
                    int8_t viewOffset = 0;
                    core::bitflag<FLAGS> flags = FLAGS::NONE;

                    auto operator<=>(const SSubpassDependency&) const = default;

                    bool valid() const;
                };
                // The arrays pointed to by this array must be terminated by `DependenciesEnd` value
                constexpr static inline SSubpassDependency DependenciesEnd = {};
                const SSubpassDependency* dependencies = &DependenciesEnd;


                // We do this differently than Vulkan so the API is not braindead, by construction we satisfy
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pCorrelatedViewMasks-03056
                static inline constexpr auto MaxMultiviewViewCount = 32u;
                uint8_t viewCorrelationGroup[MaxMultiviewViewCount] = {
                    vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,
                    vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,
                    vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,
                    vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init,vcg_init
                };

            private:
                friend class IRenderpass;
                static inline constexpr auto vcg_init = MaxMultiviewViewCount;
        };

        inline const SCreationParams& getCreationParameters() const { return m_params; }

        inline uint32_t getDepthStencilAttachmentCount() const {return m_depthStencilAttachments ? (m_depthStencilAttachments->size()-1):0;}
        inline uint32_t getColorAttachmentCount() const {return m_colorAttachments ? (m_colorAttachments->size()-1):0;}
        inline uint32_t getDepthStencilLoadOpAttachmentEnd() const {return m_loadOpDepthStencilAttachmentEnd;}
        inline uint32_t getColorLoadOpAttachmentEnd() const {return m_loadOpColorAttachmentEnd;}

        inline uint32_t getSubpassCount() const {return m_subpasses->size()-1;}
        inline uint32_t getDependencyCount() const {return m_subpassDependencies ? (m_subpassDependencies->size()-1):0;}

        inline bool hasViewMasks() const {return m_viewMaskMSB>0;}
        inline int8_t getViewMaskMSB() const {return m_viewMaskMSB;}


        struct SCreationParamValidationResult final
        {
            uint32_t depthStencilAttachmentCount = 0u;
            uint32_t colorAttachmentCount = 0u;
            uint32_t subpassCount = 0u;
            uint32_t totalInputAttachmentCount = 0u;
            uint32_t totalPreserveAttachmentCount = 0u;
            uint32_t dependencyCount = 0u;
            int8_t viewMaskMSB = -1;

            inline operator bool() const {return subpassCount;}

            inline bool hasViewMasks() const {return viewMaskMSB>=0;}
        };
        static SCreationParamValidationResult validateCreationParams(const SCreationParams& params);

        static inline bool disallowedFinalLayout(const IImage::LAYOUT& layout)
        {
            switch (layout)
            {
                case IImage::LAYOUT::UNDEFINED: [[fallthrough]];
                case IImage::LAYOUT::PREINITIALIZED:
                    return true;
                    break;
                default:
                    break;
            }
            return false;
        }
        
        template<bool InputAttachment>
        static inline bool invalidLayout(const IImage::LAYOUT _layout)
        {
            switch (_layout)
            {
                case IImage::LAYOUT::READ_ONLY_OPTIMAL:
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06913
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-attachment-06922
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06914
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-attachment-06923
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06915
                    if constexpr(!InputAttachment)
                        return true;
                    break;
                case IImage::LAYOUT::ATTACHMENT_OPTIMAL:
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-attachment-06921
                    if constexpr(InputAttachment)
                        return true;
                    break;
                case IImage::LAYOUT::UNDEFINED: [[fallthrough]];
                case IImage::LAYOUT::PREINITIALIZED: [[fallthrough]];
                case IImage::LAYOUT::PRESENT_SRC:
                    return true;
                    break;
                default:
                    break;
            }
            return false;
        }

        //
        bool compatible(const IRenderpass* other) const;

    protected:
        IRenderpass(const SCreationParams& params, const SCreationParamValidationResult& counts);
        virtual ~IRenderpass() {}

        SCreationParams m_params;
        // store for pointers in `m_params`
        using depth_stencil_attachments_array_t = core::smart_refctd_dynamic_array<SCreationParams::SDepthStencilAttachmentDescription>;
        using color_attachments_array_t = core::smart_refctd_dynamic_array<SCreationParams::SColorAttachmentDescription>;
        using subpass_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDescription>;
        using input_attachment_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDescription::SInputAttachmentRef>;
        using preserved_attachment_refs_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDescription::SPreserveAttachmentRef>;
        using subpass_deps_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDependency>;
        depth_stencil_attachments_array_t m_depthStencilAttachments;
        color_attachments_array_t m_colorAttachments;
        subpass_array_t m_subpasses;
        input_attachment_array_t m_inputAttachments;
        preserved_attachment_refs_array_t m_preserveAttachments;
        subpass_deps_array_t m_subpassDependencies;
        int8_t m_viewMaskMSB;
        // TODO: remember and document what these were about?
        uint32_t m_loadOpDepthStencilAttachmentEnd = ~0u;
        uint32_t m_loadOpColorAttachmentEnd = ~0u;
};

inline bool IRenderpass::compatible(const IRenderpass* other) const
{
    // If you find yourself spending a lot of time here in your profile, go ahead and implement a precomputed hash and store it in the renderpass
    if (this==other)
        return true;

    // assumes the attachment is used
    auto getAttachment = [](const SCreationParams& params, const bool isColor, uint32_t index) -> SCreationParams::SAttachmentDescriptionBase
    {
            if (isColor)
                return params.colorAttachments[index];
        else
            return params.depthStencilAttachments[index];
    };
    auto refIncompatible = [&]<typename T>(const T& lhs, const T& rhs) -> bool
    {
        // "Two attachment references are compatible if they have matching format and sample count, or are both VK_ATTACHMENT_UNUSED or the pointer that would contain the reference is NULL."
        const bool used = lhs.used();
        if (used!=rhs.used())
            return true;
        else if (used)
        {
            constexpr bool IsColor = !T::IsDepthStencil;
            return getAttachment(m_params,IsColor,lhs.attachmentIndex)!=getAttachment(other->m_params,IsColor,rhs.attachmentIndex);
        }
        return false;
    };

    // As an additional special case, if two render passes have a single subpass, the resolve attachment reference and depth/stencil resolve mode compatibility requirements are ignored.
    const bool checkResolve = m_params.subpasses[1]!=SCreationParams::SubpassesEnd || other->m_params.subpasses[1]!=SCreationParams::SubpassesEnd;
    auto renderIncompatible = [refIncompatible,checkResolve]<typename RenderAttachmentsRef>(const RenderAttachmentsRef& lhs, const RenderAttachmentsRef& rhs) -> bool
    {
        if (refIncompatible(lhs.render,rhs.render))
            return true;
        if (checkResolve)
        {
            if (refIncompatible(lhs.resolve,rhs.resolve))
                return true;
            if constexpr (RenderAttachmentsRef::IsDepthStencil)
            {
                if (lhs.resolveMode!=rhs.resolveMode)
                    return true;
            }
        }
        return false;
    };

    auto getInputAttachment = [getAttachment](const SCreationParams& params, const SCreationParams::SSubpassDescription::SInputAttachmentRef& ref) -> SCreationParams::SAttachmentDescriptionBase
    {
        static_assert(offsetof(SCreationParams::SSubpassDescription::SDepthStencilAttachmentRef,attachmentIndex)==offsetof(SCreationParams::SSubpassDescription::SColorAttachmentRef,attachmentIndex));
        return getAttachment(params,ref.isColor(),ref.asColor.attachmentIndex);
    };

    // Two render passes are compatible if their corresponding color, input, resolve, and depth/stencil attachment references are compatible and if they are otherwise identical except for:
    // - Initial and final image layout in attachment descriptions
    // - Load and store operations in attachment descriptions
    // - Image layout in attachment references
    auto otherSubpassIt = other->m_params.subpasses;
    for (auto subpassIt=m_params.subpasses; *subpassIt!=SCreationParams::SubpassesEnd; otherSubpassIt++,subpassIt++)
    {
        if (*otherSubpassIt==SCreationParams::SubpassesEnd)
            return false;

        // "Two arrays of attachment references are compatible if all corresponding pairs of attachments are compatible."
        // Implicitly satisfied with our API for color and depth attachments:
        // "If the arrays are of different lengths, attachment references not present in the smaller array are treated as VK_ATTACHMENT_UNUSED"
        if (renderIncompatible(subpassIt->depthStencilAttachment,otherSubpassIt->depthStencilAttachment))
            return false;
        for (uint8_t i=0; i<SCreationParams::SSubpassDescription::MaxColorAttachments; i++)
        if (renderIncompatible(subpassIt->colorAttachments[i],otherSubpassIt->colorAttachments[i]))
            return false;

        // "If the arrays are of different lengths, attachment references not present in the smaller array are treated as VK_ATTACHMENT_UNUSED"
        // Actually need to check for input attachments
        auto otherInputIt = otherSubpassIt->inputAttachments;
        constexpr SCreationParams::SSubpassDescription::SInputAttachmentRef InputEnd = {};
        for (auto inputIt=subpassIt->inputAttachments; *inputIt!=InputEnd || *otherInputIt!=InputEnd; )
        {
            const bool lUsed = *inputIt!=InputEnd && inputIt->used();
            const bool rUsed = *otherInputIt!=InputEnd && otherInputIt->used();
            if (lUsed!=rUsed)
                return false;
            else if (lUsed && getInputAttachment(m_params,*inputIt)!=getInputAttachment(other->m_params,*otherInputIt))
                return false;

            if (*inputIt!=InputEnd)
                inputIt++;
            if (*otherInputIt!=InputEnd)
                otherInputIt++;
        }
#if 0 // spec says nothing about having to match Preserve Attachments
        // To implement properly, the preserve attachments would need to be sorted
        auto otherPreserveIt = otherSubpassIt->preserveAttachments;
        constexpr auto PreserveEnd = SCreationParams::SSubpassDescription::PreserveAttachmentsEnd;
        for (auto preserveIt=subpassIt->preserveAttachments; *preserveIt!=PreserveEnd; otherPreserveIt++,preserveIt++)
        {
            if (*otherPreserveIt==PreserveEnd)
                return false;
            if (*preserveIt!=*otherPreserveIt)
                return false;
        }
        if (*otherPreserveIt!=PreserveEnd)
            return false;
#endif
    }
    if (*otherSubpassIt!=SCreationParams::SubpassesEnd)
        return false;

    return true;
}

inline IRenderpass::SCreationParamValidationResult IRenderpass::validateCreationParams(const SCreationParams& params)
{
    SCreationParamValidationResult retval = {};
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pSubpasses-parameter
    if (!params.subpasses)
        return retval;

    // If one subpass has view masks, all must do, see VUID 03058
    retval.viewMaskMSB = hlsl::findMSB(params.subpasses[0].viewMask);
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-viewMask-03057
    if (!retval.hasViewMasks())
    for (auto i=0; i<SCreationParams::MaxMultiviewViewCount; i++)
    if (params.viewCorrelationGroup[i]!=SCreationParams::vcg_init)
        return retval;

    retval.subpassCount = 0xdeadbeefu;
    auto setRetvalFalse = [&retval]()->bool
    {
        retval.subpassCount = 0; return false;
    };

    core::visit_token_terminated_array(params.depthStencilAttachments,SCreationParams::DepthStencilAttachmentsEnd,[&params,setRetvalFalse,&retval](const SCreationParams::SDepthStencilAttachmentDescription& attachment)->bool
    {
        if (!attachment.valid())
            return setRetvalFalse();
        retval.depthStencilAttachmentCount++;
        return true;
    });
    if (!retval)
        return retval;
    core::visit_token_terminated_array(params.colorAttachments,SCreationParams::ColorAttachmentsEnd,[&params,setRetvalFalse,&retval](const SCreationParams::SColorAttachmentDescription& attachment)->bool
    {
        if (!attachment.valid())
            return setRetvalFalse();
        retval.colorAttachmentCount++;
        return true;
    });
    if (!retval)
        return retval;

    retval.subpassCount = 0;
    core::visit_token_terminated_array(params.subpasses,SCreationParams::SubpassesEnd,[&params,setRetvalFalse,&retval](const SCreationParams::SSubpassDescription& subpass)->bool
    {
        if (!subpass.valid(params,retval.depthStencilAttachmentCount,retval.colorAttachmentCount))
            return setRetvalFalse();

        // can't validate:
        // - without allocating unbounded additional memory
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pAttachments-02522
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pAttachments-02523
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-loadOp-03064
        // - without doing silly O(n^2) searches or allocating scratch for a hashmap
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pPreserveAttachments-03074
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-layout-02528


        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-viewMask-03058
        if (bool(subpass.viewMask)!=retval.hasViewMasks())
            return setRetvalFalse();
        else
            retval.viewMaskMSB = core::max<int8_t>(hlsl::findMSB(subpass.viewMask),retval.viewMaskMSB);

        retval.subpassCount++;
        core::visit_token_terminated_array(subpass.inputAttachments,SCreationParams::SSubpassDescription::InputAttachmentsEnd,[&](const SCreationParams::SSubpassDescription::SInputAttachmentRef& inputAttachmentRef)->bool
        {
            if (!inputAttachmentRef.valid(params,retval.depthStencilAttachmentCount,retval.colorAttachmentCount))
                return setRetvalFalse();
            retval.totalInputAttachmentCount++;
            return true;
        });
        core::visit_token_terminated_array(subpass.preserveAttachments,SCreationParams::SSubpassDescription::PreserveAttachmentsEnd,[&](const SCreationParams::SSubpassDescription::SPreserveAttachmentRef& preserveAttachmentRef)->bool
        {
            if (preserveAttachmentRef.index>=(preserveAttachmentRef.color ? retval.colorAttachmentCount:retval.depthStencilAttachmentCount))
                return setRetvalFalse();
            retval.totalPreserveAttachmentCount++;
            return true;
        });
        return retval.subpassCount;
    });
        
    core::visit_token_terminated_array(params.dependencies,SCreationParams::DependenciesEnd,[&params,setRetvalFalse,&retval](const SCreationParams::SSubpassDependency& dependency)->bool
    {
        if (!dependency.valid())
            return setRetvalFalse();

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-srcSubpass-02526
        if (dependency.srcSubpass!=SCreationParams::SSubpassDependency::External && dependency.srcSubpass>=retval.subpassCount)
            return setRetvalFalse();
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-srcSubpass-02527
        if (dependency.dstSubpass!=SCreationParams::SSubpassDependency::External && dependency.dstSubpass>=retval.subpassCount)
            return setRetvalFalse();

        const bool hasViewLocalFlag = dependency.flags.hasFlags(SCreationParams::SSubpassDependency::FLAGS::VIEW_LOCAL);
        if (dependency.srcSubpass==dependency.dstSubpass)
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03060
            if (params.subpasses[dependency.srcSubpass].viewMask && !hasViewLocalFlag) // because of validation rule 03085 we're sure src is valid
                return setRetvalFalse();
        }

        if (retval.hasViewMasks())
        {
        }
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-viewMask-03059
        else if (hasViewLocalFlag)
            return setRetvalFalse();

        retval.dependencyCount++;
        return true;
    });

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-subpassCount-arraylength
    return retval;
}


inline bool IRenderpass::SCreationParams::SDepthStencilAttachmentDescription::valid() const
{
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-format-06698
    if (!isDepthOrStencilFormat(format))
        return false;
    const bool hasStencil = !isDepthOnlyFormat(format);
    const bool hasDepth = !isStencilOnlyFormat(format);
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-finalLayout-00843
    if (hasDepth && disallowedFinalLayout(finalLayout.depth))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-format-06699
    if (hasDepth && loadOp.depth==LOAD_OP::LOAD && initialLayout.depth==IImage::LAYOUT::UNDEFINED)
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-pNext-06705
    if (hasStencil && loadOp.actualStencilOp()==LOAD_OP::LOAD && initialLayout.actualStencilLayout() == IImage::LAYOUT::UNDEFINED)
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescriptionStencilLayout-stencilFinalLayout-03310
    if (hasStencil && disallowedFinalLayout(finalLayout.actualStencilLayout()))
        return false;
    return true;
}

inline bool IRenderpass::SCreationParams::SColorAttachmentDescription::valid() const
{
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-format-06698
    if (format==EF_UNKNOWN || isDepthOrStencilFormat(format))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-finalLayout-00843
    if (disallowedFinalLayout(finalLayout))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-format-06699
    if (loadOp==LOAD_OP::LOAD && initialLayout==IImage::LAYOUT::UNDEFINED)
        return false;
    return true;
}


inline bool IRenderpass::SCreationParams::SSubpassDescription::operator!=(const SSubpassDescription& other) const
{
    if (depthStencilAttachment!=other.depthStencilAttachment)
        return true;
    for (auto i=0u; i<MaxColorAttachments; i++)
    if (colorAttachments[i]!=other.colorAttachments[i])
        return true;
    auto tokenTerminatedSequenceUnequal = []<typename T>(const T* lhs, const T* rhs, const T& endToken) -> bool
    {
        while ((*lhs)!=endToken && (*rhs)!=endToken)
        if ((*lhs)!=(*rhs))
            return true;
        return (*lhs)!=(*rhs);
    };
    if (tokenTerminatedSequenceUnequal(inputAttachments,other.inputAttachments,InputAttachmentsEnd))
        return true;
    if (tokenTerminatedSequenceUnequal(preserveAttachments,other.preserveAttachments,PreserveAttachmentsEnd))
        return true;
    return viewMask!=other.viewMask || flags!=other.flags;
}

inline bool IRenderpass::SCreationParams::SSubpassDescription::valid(const SCreationParams& params, const uint32_t depthStencilAttachmentCount, const uint32_t colorAttachmentCount) const
{
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pNext-06870
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pNext-06871
    if (!depthStencilAttachment.valid(params.depthStencilAttachments,depthStencilAttachmentCount))
        return false;
    for (auto i=0u; i<MaxColorAttachments; i++)
    {
        if (!colorAttachments[i].valid(params.colorAttachments,colorAttachmentCount))
            return false;
    }
    bool invalid = false;
    core::visit_token_terminated_array(inputAttachments,InputAttachmentsEnd,[&](const auto& ref)->bool
    {
        if (!ref.valid(params,depthStencilAttachmentCount,colorAttachmentCount))
        {
            invalid = true;
            return false;
        }
        return true;
    });
    if (invalid)
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-flags-03076
    if (flags.hasFlags(FLAGS::PER_VIEW_POSITION_X_ONLY_BIT) && !flags.hasFlags(FLAGS::PER_VIEW_ATTRIBUTES_BIT))
        return false;
    return true;
}

inline bool IRenderpass::SCreationParams::SSubpassDescription::SDepthStencilAttachmentsRef::valid(const SDepthStencilAttachmentDescription* depthStencilAttachments, const uint32_t depthStencilAttachmentCount) const
{
    if (resolve.used())
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03178
        if (resolveMode.depth==RESOLVE_MODE::NONE && resolveMode.stencil==RESOLVE_MODE::NONE)
            return false;
    }
    return SRenderAttachmentsRef<SDepthStencilAttachmentRef>::valid(depthStencilAttachments,depthStencilAttachmentCount);
}

template<class attachment_ref_t>
inline bool IRenderpass::SCreationParams::SSubpassDescription::SRenderAttachmentsRef<attachment_ref_t>::valid(const typename attachment_ref_t::description_t* descs, const uint32_t attachmentCount) const
{
    if (!render.valid<false>(descs,attachmentCount) || !resolve.valid<false>(descs,attachmentCount))
        return false;
    const bool renderUsed = render.used();
    if (resolve.used())
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03065
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03177
        if (!renderUsed)
            return false;
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pNext-06871

        const auto& renderAttachment = descs[render.attachmentIndex];
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03066
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03179
        if (renderAttachment.samples!=IImage::ESCF_1_BIT)
            return true;
        const auto& resolveAttachment = descs[resolve.attachmentIndex];
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03067
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03180
        if (resolveAttachment.samples==IImage::ESCF_1_BIT)
            return true;
        if constexpr (attachment_ref_t::IsDepthStencil)
        {
            const bool hasDepth = !isStencilOnlyFormat(resolveAttachment.format);
            const bool hasStencil = !isDepthOnlyFormat(resolveAttachment.format);
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03181
            if (hasDepth)
            {
                // too lazy to implement it properly, this is overly conservative
                if (resolveAttachment.format!=renderAttachment.format)
                    return true;
            }
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03182
            if (hasStencil)
            {
                // too lazy to implement it properly, this is overly conservative
                if (resolveAttachment.format!=renderAttachment.format)
                    return true;
            }
        }
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03068
        else if (resolveAttachment.format!=renderAttachment.format)
            return true;
    }
    return true;
}

inline bool IRenderpass::SCreationParams::SSubpassDescription::SInputAttachmentRef::valid(const SCreationParams& params, const uint32_t depthStencilAttachmentCount, const uint32_t colorAttachmentCount) const
{
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02800
    if (!aspectMask.value)
        return false;

    constexpr auto DepthStencilAspects = IImage::E_ASPECT_FLAGS::EAF_DEPTH_BIT|IImage::E_ASPECT_FLAGS::EAF_STENCIL_BIT;
    // Implicit: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02799
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02801
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-04563
    if (isColor())
    {
        if (!asColor.valid<true>(params.colorAttachments,colorAttachmentCount))
            return false;
        if (asColor.used())
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pInputAttachments-02897
            if (isPlanarFormat(params.colorAttachments[asColor.attachmentIndex].format)) // or some other check?
                return false;
        }
    }
    else if (aspectMask.value&DepthStencilAspects)
    {
        if (aspectMask.value&(~DepthStencilAspects))
            return false;
        if (!asDepthStencil.valid<true>(params.depthStencilAttachments,depthStencilAttachmentCount))
            return false;
        const auto& attachmentDesc = params.depthStencilAttachments[asDepthStencil.attachmentIndex];
        if (isStencilOnlyFormat(attachmentDesc.format) && aspectMask!=IImage::EAF_STENCIL_BIT)
            return false;
        if (isDepthOnlyFormat(attachmentDesc.format) && aspectMask!=IImage::EAF_DEPTH_BIT)
            return false;
    }
    else
        return false;
    return true;
}

template<typename layout_t>
template<bool InputAttachment>
inline bool IRenderpass::SCreationParams::SSubpassDescription::SAttachmentRef<layout_t>::valid(const description_t* descs, const uint32_t attachmentCount) const
{
    if (!used())
        return true;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-03051
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pSubpasses-06473
    if (attachmentIndex>=attachmentCount)
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentReference2.html#VUID-VkAttachmentReference2-layout-03077
    if constexpr (IsDepthStencil)
    {
        if (invalidLayout<InputAttachment>(layout.depth))
            return false;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentReferenceStencilLayout.html#VUID-VkAttachmentReferenceStencilLayout-stencilLayout-03318
        if (invalidLayout<InputAttachment>(layout.actualStencilLayout()))
            return false;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pDepthStencilAttachment-02900
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-02651
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-02525
        if (!asset::isDepthOrStencilFormat(descs[attachmentIndex].format))
            return false;
    }
    else
    {
        if (invalidLayout<InputAttachment>(layout))
            return true;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pColorAttachments-02898
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-02899
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-02525
        if (asset::isDepthOrStencilFormat(descs[attachmentIndex].format))
            return false;
    }
    return true;
}


inline bool IRenderpass::SCreationParams::SSubpassDependency::valid() const
{
    if (srcSubpass!=External)
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-03084
        if (srcSubpass>dstSubpass)
            return false;
        else if (srcSubpass==dstSubpass)
        {
            if (bool(memoryBarrier.srcStageMask&PIPELINE_STAGE_FLAGS::FRAMEBUFFER_SPACE_BITS))
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-06810
                if (bool(memoryBarrier.dstStageMask&(~PIPELINE_STAGE_FLAGS::FRAMEBUFFER_SPACE_BITS)))
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-02245
                if (bool(memoryBarrier.dstStageMask&PIPELINE_STAGE_FLAGS::FRAMEBUFFER_SPACE_BITS) && !flags.hasFlags(FLAGS::BY_REGION))
                    return false;
            }
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-viewOffset-02530
            if (viewOffset!=0u)
                return false;
        }
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03054
        const PIPELINE_STAGE_FLAGS kDisallowedFlags = ~PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS;
        if (memoryBarrier.srcStageMask&kDisallowedFlags)
            return false;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03055
        if (dstSubpass!=External && (memoryBarrier.srcStageMask&kDisallowedFlags))
            return false;
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-03085
    else if (dstSubpass==External)
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcAccessMask-03088
    if (!allAccessesFromStages(memoryBarrier.srcStageMask).hasFlags(memoryBarrier.srcAccessMask))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcAccessMask-03089
    if (!allAccessesFromStages(memoryBarrier.dstStageMask).hasFlags(memoryBarrier.dstAccessMask))
        return false;

    if (flags.hasFlags(FLAGS::VIEW_LOCAL))
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-dependencyFlags-03090
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-dependencyFlags-03091
        if (srcSubpass==External || dstSubpass==External)
            return false;
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-dependencyFlags-03092
    else if (viewOffset!=0u)
        return false;
                            
    return true;
}

}

#endif
