#ifndef _NBL_I_RENDERPASS_H_INCLUDED_
#define _NBL_I_RENDERPASS_H_INCLUDED_

#include "nbl/core/declarations/core.h"

#include "nbl/asset/IImage.h"
#include "nbl/asset/ECommonEnums.h"

#include <compare>

namespace nbl::asset
{

// TODO: move this struct
struct SDepthStencilLayout
{
    IImage::E_LAYOUT depth = IImage::EL_UNDEFINED;
    // if you leave `stencilLayout` as undefined this means you want same layout as `depth`
    IImage::E_LAYOUT stencil = IImage::EL_UNDEFINED;

    auto operator<=>(const SDepthStencilLayout&) const = default;

    inline IImage::E_LAYOUT actualStencilLayout() const
    {
        if (stencil!=IImage::EL_UNDEFINED)
            return stencil;
        // synchronization2 will save us doing these if-checks
        if (depth==IImage::EL_DEPTH_ATTACHMENT_OPTIMAL)
            return IImage::EL_STENCIL_ATTACHMENT_OPTIMAL;
        if (depth==IImage::EL_DEPTH_READ_ONLY_OPTIMAL)
            return IImage::EL_STENCIL_READ_ONLY_OPTIMAL;
        return depth;
    }
};

class IRenderpass
{
    public:
        // TODO: move this out somewhere? looks useful
        template<typename T, class func_t>
        static inline void visitTokenTerminatedArray(const T* array, const T& endToken, func_t&& func)
        {
            if (array)
            for (auto it=array; *it!=endToken && func(*it); it++)
            {
            }
        }

        enum class E_LOAD_OP : uint8_t
        {
            LOAD = 0,
            CLEAR,
            DONT_CARE,
            UNDEFINED
        };
        enum class E_STORE_OP: uint8_t
        {
            STORE = 0,
            DONT_CARE,
            UNDEFINED
        };
        enum class E_SUBPASS_DESCRIPTION_FLAGS : uint8_t
        {
            NONE = 0x00,
            PER_VIEW_ATTRIBUTES_BIT = 0x01,
            PER_VIEW_POSITION_X_ONLY_BIT = 0x02
            // TODO: VK_EXT_rasterization_order_attachment_access
        };
        // For all arrays here we use ArrayNameEnd terminator instead of specifying the count
        struct SCreationParams
        {
            public:
                // funny little struct to allow us to use ops as `layout.depth`, `layout.stencil` and `layout`
                struct Layout : SDepthStencilLayout
                {
                    auto operator<=>(const Layout&) const = default;

                    inline void operator=(const IImage::E_LAYOUT& general) {depth=general;}

                    inline operator IImage::E_LAYOUT() const { return depth; }
                };
                // The reason we don't have separate types per depthstencil, color and resolve is because
                // attachments limits (1 depth, MaxColorAttachments color and resolve) only apply PER SUBPASS
                // so overall we can have as many attachments of whatever type and in whatever order we want
                struct SAttachmentDescription
                {
                    // similar idea for load/store ops as for layouts
                    template<typename op_t>
                    struct Op
                    {
                        op_t depth : 2 = op_t::DONT_CARE;
                        op_t stencil : 2 = op_t::UNDEFINED;


                        auto operator<=>(const Op&) const = default;

                        inline void operator=(const Op& general) {SDepthStencilOp<op_t>::depth=general;}

                        inline operator op_t() const { return SDepthStencilOp<op_t>::depth; }
                    
                        inline op_t actualStencilOp() const
                        {
                            return stencil!=op_t::UNDEFINED ? stencil:depth;
                        }
                    };

                    E_FORMAT format = EF_UNKNOWN;
                    IImage::E_SAMPLE_COUNT_FLAGS samples : 6 = IImage::ESCF_1_BIT;
                    uint8_t mayAlias : 1 = false;
                    Op<E_LOAD_OP> loadOp = {};
                    Op<E_STORE_OP> storeOp = {};
                    Layout initialLayout = {};
                    Layout finalLayout = {};

                    auto operator<=>(const SAttachmentDescription&) const = default;

                    inline bool valid() const
                    {
                        auto disallowedFinalLayout = [](const auto& layout)->bool
                        {
                            switch (layout)
                            {
                                case IImage::EL_UNDEFINED: [[fallthrough]];
                                case IImage::EL_PREINITIALIZED:
                                    return true;
                                    break;
                                default:
                                    break;
                            }
                            return false;
                        };
                        //
                        if (disallowedFinalLayout(finalLayout))
                            return false;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescriptionStencilLayout-stencilFinalLayout-03310
                        if (asset::isDepthOrStencilFormat(format) && !asset::isDepthOnlyFormat(format) && disallowedFinalLayout(finalLayout.actualStencilLayout()))
                            return false;
                        return true;
                    }
                };
                // The arrays pointed to by this array must be terminated by `AttachmentsEnd` value, which implicitly satisfies
                // 
                constexpr static inline SAttachmentDescription AttachmentsEnd = {};
                const SAttachmentDescription* attachments = &AttachmentsEnd;
            

                struct SSubpassDescription
                {
                    constexpr static inline uint32_t AttachmentUnused = 0xffFFffFFu;
                    template<typename layout_t>
                    struct SAttachmentRef
                    {
                        public:
                            // If you leave the `attachmentIndex` as default then it means its not being used
                            uint32_t attachmentIndex = AttachmentUnused;
                            layout_t layout = {};

                            inline bool used() const {return attachmentIndex!=AttachmentUnused;}

                            auto operator<=>(const SAttachmentRef<layout_t>&) const = default;
                        
                        protected:
                            inline bool invalidLayout(const IImage::E_LAYOUT _layout)
                            {
                                switch (_layout)
                                {
                                    case IImage::EL_UNDEFINED: [[fallthrough]];
                                    case IImage::EL_PREINITIALIZED: [[fallthrough]];
                                    case IImage::EL_PRESENT_SRC:
                                        return true;
                                        break;
                                    default:
                                        break;
                                }
                                return false;
                            }
                            inline bool invalid() const
                            {
                                if (!used())
                                    return false;

                                //https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentReference2.html#VUID-VkAttachmentReference2-layout-03077
                                const IImage::E_LAYOUT usageAllOrJustDepth = layout;
                                if (invalidLayout(usageAllOrJustDepth))
                                    return true;
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentReferenceStencilLayout.html#VUID-VkAttachmentReferenceStencilLayout-stencilLayout-03318
                                if constexpr (!std::is_base_of_v<SDepthStencilLayout,layout_t>)
                                if (invalidLayout(layout.actualStencilLayout())
                                    return true;

                                return false;
                            }
                    };
                    struct SInputAttachmentRef : SAttachmentRef<Layout>
                    {
                        core::bitflag<IImage::E_ASPECT_FLAGS> aspectMask = IImage::E_ASPECT_FLAGS::EAF_NONE;

                        auto operator<=>(const SInputAttachmentRef&) const = default;

                        inline bool valid() const
                        {
                            if (invalid())
                                return false;
                            if (used())
                            {
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02800
                                if (!aspectMask.value)
                                    return false;
                                // TODO: synchronization2 will wholly replace this with https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06921
                                switch (layout)
                                {
                                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06912
                                    case IImage::EL_COLOR_ATTACHMENT_OPTIMAL: [[fallthrough]];
                                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06918
                                    case IImage::EL_DEPTH_ATTACHMENT_OPTIMAL: [[fallthrough]];
                                    case IImage::EL_STENCIL_ATTACHMENT_OPTIMAL:
                                        return false;
                                        break;
                                    default:
                                        break;
                                }
                            }
                            return true;
                        }
                    };
                    template<typename layout_t>
                    struct SRenderAttachmentRef
                    {
                        constexpr static inline bool IsDepth = std::is_same_v<layout_t,SDepthStencilLayout>;

                        SAttachmentRef<layout_t> render;
                        SAttachmentRef<layout_t> resolve;

                        auto operator<=>(const SRenderAttachmentRef<layout_t>&) const = default;

                        inline bool valid() const
                        {
                            if (render.invalid() || resolve.invalid())
                                return false;
                            const bool renderUsed = render.used();
                            const bool resolveUsed = resolve.used();
                            if (renderUsed)
                            {
                                // TODO: synchronization2 will replace all this by just 2 things
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06922
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06923
                                switch (render.layout)
                                {
                                    case IImage::EL_COLOR_ATTACHMENT_OPTIMAL:
                                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06915
                                        if constexpr(IsDepth)
                                            return false;
                                        break;
                                    case IImage::EL_SHADER_READ_ONLY_OPTIMAL:
                                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06913
                                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06914
                                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06915
                                        return false;
                                        break;
                                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06919
                                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06920
                                    case IImage::EL_DEPTH_ATTACHMENT_OPTIMAL: [[fallthrough]];
                                    case IImage::EL_DEPTH_READ_ONLY_OPTIMAL:
                                        if constexpr (!IsDepth)
                                            return false;
                                        break;
                                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-06251
                                    case IImage::EL_STENCIL_ATTACHMENT_OPTIMAL: [[fallthrough]];
                                    case IImage::EL_STENCIL_READ_ONLY_OPTIMAL:
                                        return false;
                                        break;
                                    default:
                                        break;
                                }
                                /*
                                if constexpr (IsDepth)
                                switch (render.layout.actualStencilLayout())
                                {
                                        return false;
                                        break;
                                    default:
                                        break;
                                }*/
                            }
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03065
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03177
                            else if (resolveUsed)
                                return false;
                            return true;
                        }
                    };
                    struct SDepthStencilAttachmentRef : SRenderAttachmentRef<SDepthStencilLayout>
                    {
                        enum class E_RESOLVE_MODE : uint8_t
                        {
                            NONE = 0,
                            SAMPLE_ZERO_BIT = 0x00000001,
                            AVERAGE_BIT = 0x00000002,
                            MIN_BIT = 0x00000004,
                            MAX_BIT = 0x00000008
                        };
                        struct ResolveMode
                        {
                            E_RESOLVE_MODE depth : 4 = NONE;
                            E_RESOLVE_MODE stencil : 4 = NONE;
                        } resolveMode;
                    };
                    using SColorAttachmentRef = SRenderAttachmentRef<IImage::E_LAYOUT>;


                    auto operator<=>(const SSubpassDescription&) const = default;

                    inline bool valid() const
                    {
                        if (!depthStencilAttachment.valid())
                            return false;
                        for (auto i=0u; i<MaxColorAttachments; i++)
                        {
                            if (!colorAttachments[i].valid())
                                return false;
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pDepthStencilAttachment-04440
                            if (depthStencilAttachment.render.used() && depthStencilAttachment.render.attachmentIndex==colorAttachments[i].render.attachmentIndex)
                                return false;
                        }
                        bool invalid = false;
                        auto attachmentValidVisitor = [&invalid](const auto& ref)->bool
                        {
                            if (!ref.valid())
                            {
                                invalid = true;
                                return false;
                            }
                            return true;
                        };
                        visitTokenTerminatedArray(inputAttachments,InputAttachmentsEnd,attachmentValidVisitor);
                        if (invalid)
                            return false;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-flags-03076
                        if (flags.hasFlags(E_SUBPASS_DESCRIPTION_FLAGS::PER_VIEW_POSITION_X_ONLY_BIT) && !flags.hasFlags(E_SUBPASS_DESCRIPTION_FLAGS::PER_VIEW_ATTRIBUTES_BIT))
                            return false;
                        return true;
                    }


                    //! Field ordering prioritizes ergonomics
                    static inline constexpr auto MaxColorAttachments = 8u;
                    SColorAttachmentRef colorAttachments[MaxColorAttachments] = {};

                    SDepthStencilAttachmentRef depthStencilAttachment = {};

                    // The arrays pointed to by this array must be terminated by `InputAttachmentsEnd` value
                    constexpr static inline SInputAttachmentRef InputAttachmentsEnd = {};
                    const SInputAttachmentRef* inputAttachments = &InputAttachmentsEnd;

                    // The arrays pointed to by this array must be terminated by `AttachmentUnused` value, which implicitly satisfies:
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-03073
                    const uint32_t* preserveAttachments = &AttachmentUnused;

                    // TODO: shading rate attachment

                    uint32_t viewMask = 0u;
                    core::bitflag<E_SUBPASS_DESCRIPTION_FLAGS> flags = E_SUBPASS_DESCRIPTION_FLAGS::NONE;
                    // Do not expose because we don't support Subpass Shading
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pipelineBindPoint-04953
                    //E_PIPELINE_BIND_POINT pipelineBindPoint : 2 = EPBP_GRAPHICS;
                };
                constexpr static inline SSubpassDescription SubpassesEnd = {};
                const SSubpassDescription* subpasses = &SubpassesEnd;

                struct SSubpassDependency
                {
                    public:
                        constexpr static inline uint32_t External = ~0u;
                        enum class E_DEPENDENCY_FLAGS : uint8_t
                        {
                            BY_REGION = 0x01u,
                            VIEW_LOCAL = 0x02u,
                            DEVICE_GROUP = 0x04u,
                            FEEDBACK_LOOP = 0x08u
                        };

                        uint32_t srcSubpass = External;
                        uint32_t dstSubpass = External;
                        int32_t viewOffset = 0;
                        // TODO: Redo with synchronization2
                        // If a VkMemoryBarrier2 is included in the pNext chain, srcStageMask, dstStageMask, srcAccessMask, and dstAccessMask parameters are ignored. The synchronization and access scopes instead are defined by the parameters of VkMemoryBarrier2.
                        core::bitflag<E_PIPELINE_STAGE_FLAGS> srcStageMask = EPSF_BOTTOM_OF_PIPE_BIT;
                        core::bitflag<E_PIPELINE_STAGE_FLAGS> dstStageMask = EPSF_TOP_OF_PIPE_BIT;
                        core::bitflag<E_ACCESS_FLAGS> srcAccessMask = EAF_NONE;
                        core::bitflag<E_ACCESS_FLAGS> dstAccessMask = EAF_NONE;
                        core::bitflag<E_DEPENDENCY_FLAGS> dependencyFlags = EDF_NONE;

                        auto operator<=>(const SSubpassDependency&) const = default;

                        inline bool valid() const
                        {
                            if (srcSubpass!=External)
                            {
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-03084
                                if (srcSubpass>dstSubpass)
                                    return false;
                                else if (srcSubpass==dstSubpass)
                                {
                                    if (srcStageMask.value&EPSF_FRAMEBUFFER_SPACE_BIT_DEVSH)
                                    {
                                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-06810
                                        if (dstAccessMask.value&(~EPSF_FRAMEBUFFER_SPACE_BIT_DEVSH))
                                            return false;
                                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-02245
                                    }
                                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-viewOffset-02530
                                    if (viewOffset!=0u)
                                        return false;
                                }
                            }
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcSubpass-03085
                            else if (dstSubpass==External)
                                return false;

                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-dependencyFlags-03092
                            if (!dependencyFlags.hasFlags(E_DEPENDENCY_FLAGS::VIEW_LOCAL) && viewOffset!=0u)
                                return false;
                            
                            return validMemoryBarrier(srcSubpass,srcStageMask,srcAccessMask) && validMemoryBarrier(dstSubpass,dstStageMask,dstAccessMask);
                        }

                    private:
                        inline bool validMemoryBarrier(const uint32_t subpassIx, const core::bitflag<E_PIPELINE_STAGE_FLAGS> stageMask, const core::bitflag<E_ACCESS_FLAGS> accessMask) const
                        {
                            // will be redone with sync2
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcStageMask-03937
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-dstStageMask-03937
                            if (stageMask.value==0u)
                                return false;
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-access-types-supported
                            if (subpassIx!=SCreationParams::SSubpassDependency::External)
                            {
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03054
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03055
                                constexpr E_PIPELINE_STAGE_FLAGS kDisallowedFlags = EPSF_HOST_BIT|EPSF_COMPUTE_SHADER_BIT|EPSF_TRANSFER_BIT|EPSF_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|EPSF_COMMAND_PREPROCESS_BIT_NV;
                                if (stageMask.value&kDisallowedFlags)
                                    return false;
                            }
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-dependencyFlags-03090
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-dependencyFlags-03091
                            else if (dependencyFlags.hasFlags(E_DEPENDENCY_FLAGS::VIEW_LOCAL))
                                return false;
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcAccessMask-03088
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDependency2-srcAccessMask-03089
                            if (allAccessesFromStages(stageMask).hasFlags(accessMask))
                                return false;
                            return true;
                        }
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

        struct CreationParamValidationResult
        {
            uint32_t attachmentCount = 0u;
            uint32_t subpassCount = 0u;
            uint32_t dependencyCount = 0u;

            inline operator bool() const {return subpassCount;}
        };
        inline virtual CreationParamValidationResult validateCreationParams(const SCreationParams& params)
        {
            CreationParamValidationResult retval = {};
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pSubpasses-parameter
            if (!params.subpasses)
                return retval;
            
            visitTokenTerminatedArray(params.attachments,SCreationParams::AttachmentsEnd,[&params,&retval](const SCreationParams::SAttachmentDescription& subpass)->bool
            {
                retval.attachmentCount++;
            });

            const bool subpassesHaveViewMasks = params.subpasses[0].viewMask;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-viewMask-03057
            if (!subpassesHaveViewMasks)
            for (auto i=0; i<SCreationParams::MaxMultiviewViewCount; i++)
            if (params.viewCorrelationGroup[i]!=SCreationParams::vcg_init)
                return retval;

            auto setRetvalFalse = [&retval]()->bool{retval.subpassCount = 0; return false;};
            auto invalidAttachmentFormat = [](const auto& attachment, const bool isDepth) -> bool
            {
                if (asset::isDepthOrStencilFormat(attachment.format)!= isDepth)
                    return true;
                return false;
            };
            auto invalidRenderAttachmentRef = [&params,&retval,invalidAttachmentFormat](const auto& renderAttachmentRef) -> bool
            {
                constexpr bool IsDepth = std::remove_reference_t<decltype(renderAttachmentRef)>::IsDepth;
                if (renderAttachmentRef.render.used())
                {
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-03051
                    if (renderAttachmentRef.render.attachmentIndex>=retval.attachmentCount)
                        return true;
                    const auto& renderAttachment = params.attachments[renderAttachmentRef.render.attachmentIndex];
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pColorAttachments-02898
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pDepthStencilAttachment-02900
                    if (invalidAttachmentFormat(renderAttachment,IsDepth))
                        return true;

                    if constexpr (IsDepth)
                    {
                        auto disallowedStencilLayouts = [](const IImage::E_LAYOUT layout) -> bool
                        {
                            switch (layout)
                            {
                                case IImage::EL_COLOR_ATTACHMENT_OPTIMAL: [[fallthrough]];
                                case IImage::EL_DEPTH_ATTACHMENT_OPTIMAL: [[fallthrough]];
                                case IImage::EL_DEPTH_READ_ONLY_OPTIMAL:
                                    return true;
                                    break;
                                default:
                                    break;
                            }
                            return false;
                        };
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescriptionStencilLayout-stencilInitialLayout-03308
                        if (disallowedStencilLayouts(renderAttachment.initialLayout.actualStencilLayout()))
                            return true;
                        const auto stencilFinalLayout = renderAttachment.finalLayout.actualStencilLayout();
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescriptionStencilLayout-stencilFinalLayout-03309
                        if (disallowedStencilLayouts(stencilFinalLayout))
                            return true;
                    }

                    if (renderAttachmentRef.resolve.used())
                    {
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-03051
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pSubpasses-06473
                        if (renderAttachmentRef.resolve.attachmentIndex>=retval.attachmentCount)
                            return true;
                        const auto& resolveAttachment = params.attachments[renderAttachmentRef.resolve.attachmentIndex];
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03066
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03179
                        if (renderAttachment.samples!=IImage::ESCF_1_BIT)
                            return true;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03067
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03180
                        if (resolveAttachment.samples==IImage::ESCF_1_BIT)
                            return true;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03068
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03181
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03182
                        if (resolveAttachment.format!=renderAttachment.format)
                            return true;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-02899
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-02651
                        if (invalidAttachmentFormat(resolveAttachment,IsDepth))
                            return true;
                        
                        if constexpr(IsDepth)
                        {
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03178
                            if (renderAttachmentRef.resolveMode.depth==NONE && renderAttachmentRef.resolveMode.stencil==NONE)
                                return true;
                        }
                    }
                }
                return false;
            };
            visitTokenTerminatedArray(params.subpasses,SCreationParams::SubpassesEnd,[&params,setRetvalFalse,&retval,subpassesHaveViewMasks,invalidRenderAttachmentRef](const SCreationParams::SSubpassDescription& subpass)->bool
            {
                if (!subpass.valid())
                    return setRetvalFalse();

                // TODO: validate references
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-03051
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pSubpasses-06473

                // can't validate:
                // - without allocating unbounded additional memory
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pAttachments-02522
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pAttachments-02523
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-loadOp-03064
                // - without doing silly O(n^2) searches or allocating scratch for a hashmap
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pPreserveAttachments-03074
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-layout-02528

                if (invalidRenderAttachmentRef(subpass.depthStencilAttachment))
                    return setRetvalFalse();

                for (auto i=0u; i<SCreationParams::SSubpassDescription::MaxColorAttachments; i++)
                if (invalidRenderAttachmentRef(subpass.colorAttachments[i]))
                    return setRetvalFalse();

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-viewMask-03058
                if (bool(subpass.viewMask)!=subpassesHaveViewMasks)
                    return setRetvalFalse();

                retval.subpassCount++;
                visitTokenTerminatedArray(subpass.inputAttachments,SCreationParams::SSubpassDescription::InputAttachmentsEnd,[&](const SCreationParams::SSubpassDescription::SInputAttachmentRef& inputAttachmentRef)->bool
                {
                    if (inputAttachmentRef.attachmentIndex!=SCreationParams::SSubpassDescription::AttachmentUnused)
                    {
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-03051
                        if (inputAttachmentRef.attachmentIndex>=retval.attachmentCount)
                            return setRetvalFalse();
                        const auto& inputAttachment = params.attachments[inputAttachmentRef.attachmentIndex];
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pInputAttachments-02897
                        if (asset::isPlanarFormat(inputAttachment.format)) // or some other check?
                            return setRetvalFalse();
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-02525
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02799
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02801
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-04563
                        if (asset::isDepthOrStencilFormat(inputAttachment.format))
                        {
                            if (asset::isStencilOnlyFormat(inputAttachment.format))
                            {
                                if (inputAttachmentRef.aspectMask.value!=IImage::EAF_STENCIL_BIT)
                                    return setRetvalFalse();
                            }
                            else if (inputAttachmentRef.aspectMask.hasFlags(~(core::bitflag(IImage::EAF_DEPTH_BIT)|IImage::EAF_STENCIL_BIT)))
                                return setRetvalFalse();
                        }
                        else if (inputAttachmentRef.aspectMask.value!=IImage::EAF_COLOR_BIT)
                            return setRetvalFalse();
                    }
                    return true;
                });
                return retval.subpassCount;
            });
        
            visitTokenTerminatedArray(params.dependencies,SCreationParams::DependenciesEnd,[&params,setRetvalFalse,&retval,subpassesHaveViewMasks](const SCreationParams::SSubpassDependency& dependency)->bool
            {
                if (!dependency.valid())
                    return setRetvalFalse();

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-srcSubpass-02526
                if (dependency.srcSubpass!=SCreationParams::SSubpassDependency::External && dependency.srcSubpass>=retval.subpassCount)
                    return setRetvalFalse();
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-srcSubpass-02527
                if (dependency.dstSubpass!=SCreationParams::SSubpassDependency::External && dependency.dstSubpass>=retval.subpassCount)
                    return setRetvalFalse();

                const bool hasViewLocalFlag = dependency.dependencyFlags.hasFlags(EDF_VIEW_LOCAL_BIT);
                if (dependency.srcSubpass==dependency.dstSubpass)
                {
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03060
                    if (params.subpasses[dependency.srcSubpass].viewMask && !hasViewLocalFlag) // because of validation rule 03085 we're sure src is valid
                        return setRetvalFalse();
                }

                if (subpassesHaveViewMasks)
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

    explicit IRenderpass(const SCreationParams& params, const uint32_t attachmentCount, const uint32_t subpassCount, const uint32_t dependencyCount) :
        m_params(params),
        m_attachments(attachmentCount ? core::make_refctd_dynamic_array<attachments_array_t>(attachmentCount):nullptr),
        m_subpasses(subpassCount ? core::make_refctd_dynamic_array<subpasses_array_t>(subpassCount):nullptr),
        m_dependencies(dependencyCount ? core::make_refctd_dynamic_array<subpass_deps_array_t>(dependencyCount):nullptr)
    {
        auto attachments = core::SRange<const SCreationParams::SAttachmentDescription>{params.attachments, params.attachments+params.attachmentCount};
        std::copy(attachments.begin(), attachments.end(), m_attachments->begin());
        m_params.attachments = m_attachments->data();

        auto subpasses = core::SRange<const SCreationParams::SSubpassDescription>{params.subpasses, params.subpasses+params.subpassCount};
        std::copy(subpasses.begin(), subpasses.end(), m_subpasses->begin());
        m_params.subpasses = m_subpasses->data();

        uint32_t attRefCnt = 0u;
        uint32_t preservedAttRefCnt = 0u;
        for (const auto& sb : (*m_subpasses))
        {
            attRefCnt += sb.colorAttachmentCount;
            attRefCnt += sb.inputAttachmentCount;
            if (sb.resolveAttachments)
                attRefCnt += sb.colorAttachmentCount;
            if (sb.depthStencilAttachment.attachment!=ATTACHMENT_UNUSED)
                ++attRefCnt;

            if (sb.preserveAttachments)
                preservedAttRefCnt += sb.preserveAttachmentCount;
        }
        if (attRefCnt)
            m_attachmentRefs = core::make_refctd_dynamic_array<attachment_refs_array_t>(attRefCnt);
        if (preservedAttRefCnt)
            m_preservedAttachmentRefs = core::make_refctd_dynamic_array<preserved_attachment_refs_array_t>(preservedAttRefCnt);

        uint32_t refOffset = 0u;
        uint32_t preservedRefOffset = 0u;
        auto* refs = m_attachmentRefs->data();
        auto* preservedRefs = m_preservedAttachmentRefs->data();
        for (auto& sb : (*m_subpasses))
        {
            if (m_attachmentRefs)
            {
#define _COPY_ATTACHMENT_REFS(_array,_count)\
                std::copy(sb._array, sb._array+sb._count, refs+refOffset);\
                sb._array = refs+refOffset;\
                refOffset += sb._count;

                _COPY_ATTACHMENT_REFS(colorAttachments, colorAttachmentCount);
                if (sb.inputAttachments)
                {
                    _COPY_ATTACHMENT_REFS(inputAttachments, inputAttachmentCount);
                }
                if (sb.resolveAttachments)
                {
                    _COPY_ATTACHMENT_REFS(resolveAttachments, colorAttachmentCount);
                }
                if (sb.depthStencilAttachment)
                {
                    refs[refOffset] = sb.depthStencilAttachment[0];
                    sb.depthStencilAttachment = refs + refOffset;
                    ++refOffset;
                }
#undef _COPY_ATTACHMENT_REFS
            }

            if (m_preservedAttachmentRefs)
            {
                std::copy(sb.preserveAttachments, sb.preserveAttachments+sb.preserveAttachmentCount, preservedRefs+preservedRefOffset);
                sb.preserveAttachments = preservedRefs+preservedRefOffset;
                preservedRefOffset += sb.preserveAttachmentCount;
            }
        }

        if (!params.dependencies)
            return;

        auto deps = core::SRange<const SCreationParams::SSubpassDependency>{params.dependencies, params.dependencies+params.dependencyCount};
        std::copy(deps.begin(), deps.end(), m_dependencies->begin());
        m_params.dependencies = m_dependencies->data();
    }

    inline core::SRange<const SCreationParams::SAttachmentDescription> getAttachments() const
    {
        if (!m_attachments)
            return { nullptr, nullptr };
        return { m_attachments->cbegin(), m_attachments->cend() };
    }

    inline core::SRange<const SCreationParams::SSubpassDescription> getSubpasses() const
    {
        if (!m_subpasses)
            return { nullptr, nullptr };
        return { m_subpasses->cbegin(), m_subpasses->cend() };
    }

    inline core::SRange<const SCreationParams::SSubpassDependency> getSubpassDependencies() const
    {
        if (!m_dependencies)
            return { nullptr, nullptr };
        return { m_dependencies->cbegin(), m_dependencies->cend() };
    }

    const SCreationParams& getCreationParameters() const { return m_params; }

protected:
    virtual ~IRenderpass() {}

    SCreationParams m_params;
    using attachments_array_t = core::smart_refctd_dynamic_array<SCreationParams::SAttachmentDescription>;
    // storage for m_params.attachments
    attachments_array_t m_attachments;
    using subpasses_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDescription>;
    // storage for m_params.subpasses
    subpasses_array_t m_subpasses;
    using subpass_deps_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDependency>;
    // storage for m_params.dependencies
    subpass_deps_array_t m_dependencies;
    using attachment_refs_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDescription::SAttachmentRef>;
    attachment_refs_array_t m_attachmentRefs;
    using preserved_attachment_refs_array_t = core::smart_refctd_dynamic_array<uint32_t>;
    preserved_attachment_refs_array_t m_preservedAttachmentRefs;
};

}

#endif
