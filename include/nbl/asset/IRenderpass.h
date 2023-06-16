#ifndef _NBL_I_RENDERPASS_H_INCLUDED_
#define _NBL_I_RENDERPASS_H_INCLUDED_

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
                struct DepthStencilOp
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
                // a depth/stencil attachment in place of a colour one.
                template<typename Layout, template<typename> class op_t>
                struct SAttachmentDescription
                {
                    public:
                        E_FORMAT format = EF_UNKNOWN;
                        IImage::E_SAMPLE_COUNT_FLAGS samples : 6 = IImage::ESCF_1_BIT;
                        uint8_t mayAlias : 1 = false;
                        op_t<LOAD_OP> loadOp = {};
                        op_t<STORE_OP> storeOp = {};
                        Layout initialLayout = {};
                        Layout finalLayout = {};

                        auto operator<=>(const SAttachmentDescription&) const = default;

                    protected:
                        inline bool disallowedFinalLayout(const IImage::LAYOUT& layout) const
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
                };
                struct SDepthStencilAttachmentDescription : SAttachmentDescription<IImage::SDepthStencilLayout,DepthStencilOp>
                {
                    inline bool valid() const
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
                        if (hasDepth && initialLayout.depth==IImage::LAYOUT::UNDEFINED)
                            return false;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-pNext-06705
                        if (hasStencil && initialLayout.actualStencilLayout()==IImage::LAYOUT::UNDEFINED)
                            return false;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescriptionStencilLayout-stencilFinalLayout-03310
                        if (hasStencil && disallowedFinalLayout(finalLayout.actualStencilLayout()))
                            return false;
                        return true;
                    }
                };
                struct SColorAttachmentDescription : SAttachmentDescription<IImage::LAYOUT,Op>
                {
                    inline bool valid() const
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
                };
                // The arrays pointed to by this array must be terminated by `DepthStencilAttachmentsEnd` value, which implicitly satisfies a few VUIDs
                constexpr static inline SDepthStencilAttachmentDescription DepthStencilAttachmentsEnd = {};
                const SDepthStencilAttachmentDescription* depthStencilAttachments = &DepthStencilAttachmentsEnd;
                // The arrays pointed to by this array must be terminated by `ColorAttachmentsEnd` value, which implicitly satisfies a few VUIDs
                constexpr static inline SColorAttachmentDescription ColorAttachmentsEnd = {};
                const SColorAttachmentDescription* colorAttachments = &ColorAttachmentsEnd;

                struct SSubpassDescription
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
                    struct SAttachmentRef
                    {
                        public:
                            // If you leave the `attachmentIndex` as default then it means its not being used
                            uint32_t attachmentIndex = AttachmentUnused;
                            layout_t layout = {};

                            inline bool used() const {return attachmentIndex!=AttachmentUnused;}

                            auto operator<=>(const SAttachmentRef<layout_t>&) const = default;
                        
                        protected:
                            friend struct SInputAttachmentRef;
                            template<bool InputAttachment>
                            inline bool invalidLayout(const IImage::LAYOUT _layout)
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
                            template<bool InputAttachment>
                            inline bool invalid() const
                            {
                                if (!used())
                                    return false;

                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentReference2.html#VUID-VkAttachmentReference2-layout-03077
                                if constexpr (!std::is_base_of_v<IImage::SDepthStencilLayout,layout_t>)
                                {
                                    if (invalidLayout<InputAttachment>(layout.depth))
                                        return true;
                                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentReferenceStencilLayout.html#VUID-VkAttachmentReferenceStencilLayout-stencilLayout-03318
                                    if (invalidLayout<InputAttachment>(layout.actualStencilLayout())
                                        return true;
                                }
                                else if (invalidLayout<InputAttachment>(layout))
                                    return true;

                                return false;
                            }
                    };
                    using SDepthStencilAttachmentRef = SAttachmentRef<IImage::SDepthStencilLayout>;
                    using SColorAttachmentRef = SAttachmentRef<IImage::LAYOUT>;
                    struct SInputAttachmentRef
                    {
                        // we can tell which one you meant by the aspectMask
                        union
                        {
                            SDepthStencilAttachmentRef asDepthStencil;
                            SColorAttachmentRef asColor;
                        };
                        core::bitflag<IImage::E_ASPECT_FLAGS> aspectMask = IImage::E_ASPECT_FLAGS::EAF_NONE;

                        auto operator<=>(const SInputAttachmentRef&) const = default;

                        inline bool valid() const
                        {
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02800
                            if (!aspectMask.value)
                                return false;
                            
                            if (aspectMask.hasFlags(IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT))
                            {
                                if (asColor.invalid<true>())
                                    return false;
                            }
                            else if (aspectMask.value&(IImage::E_ASPECT_FLAGS::EAF_DEPTH_BIT|IImage::E_ASPECT_FLAGS::EAF_STENCIL_BIT))
                            {
                                if (asDepthStencil.invalid<true>())
                                    return false;
                            }
                            else
                                return false;
                            return true;
                        }
                    };
                    template<class attachment_ref_t>
                    struct SRenderAttachmentsRef
                    {
                        constexpr static inline bool IsDepthStencil = std::is_same_v<attachment_ref_t,SDepthStencilAttachmentRef>;

                        attachment_ref_t render;
                        attachment_ref_t resolve;

                        auto operator<=>(const SRenderAttachmentsRef<attachment_ref_t>&) const = default;

                        inline bool valid() const
                        {
                            if (render.invalid<false>() || resolve.invalid<false>())
                                return false;
                            const bool renderUsed = render.used();
                            const bool resolveUsed = resolve.used();
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pResolveAttachments-03065
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03177
                            if (resolveUsed && !renderUsed)
                                return false;
                            return true;
                        }
                    };
                    struct SDepthStencilAttachmentsRef : SRenderAttachmentsRef<SDepthStencilAttachmentRef>
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
                            RESOLVE_MODE depth : 4 = NONE;
                            RESOLVE_MODE stencil : 4 = NONE;
                        } resolveMode;

                        inline bool valid() const
                        {
                            if (resolve.used())
                            {
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03178
                                if (resolveMode.depth==NONE && resolveMode.stencil==NONE)
                                    return false;
                            }
                            return SRenderAttachmentsRef<SDepthStencilAttachmentRef>::valid();
                        }
                    };
                    using SColorAttachmentsRef = SRenderAttachmentsRef<SColorAttachmentRef>;

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
                        core::visit_token_terminated_array(inputAttachments,InputAttachmentsEnd,[&invalid](const auto& ref)->bool
                        {
                            if (!ref.valid())
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


                    //! Field ordering prioritizes ergonomics
                    SDepthStencilAttachmentsRef depthStencilAttachment = {};

                    static inline constexpr auto MaxColorAttachments = 8u;
                    SColorAttachmentsRef colorAttachments[MaxColorAttachments] = {};

                    // The arrays pointed to by this array must be terminated by `InputAttachmentsEnd` value
                    constexpr static inline SInputAttachmentRef InputAttachmentsEnd = {};
                    const SInputAttachmentRef* inputAttachments = &InputAttachmentsEnd;

                    // The arrays pointed to by this array must be terminated by `AttachmentUnused` value, which implicitly satisfies:
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-03073
                    const uint32_t* preserveAttachments = &AttachmentUnused;

                    // TODO: shading rate attachment

                    uint32_t viewMask = 0u;
                    core::bitflag<FLAGS> flags = FLAGS::NONE;
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

                        inline bool valid() const
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

                    private:
#if 0
                        inline bool validMemoryBarrier(const uint32_t subpassIx, const core::bitflag<E_PIPELINE_STAGE_FLAGS> stageMask, const core::bitflag<E_ACCESS_FLAGS> accessMask) const
                        {
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-access-types-supported
                            if (subpassIx!=SCreationParams::SSubpassDependency::External)
                            {
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03054
                                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03055
                                constexpr E_PIPELINE_STAGE_FLAGS kDisallowedFlags = EPSF_HOST_BIT|EPSF_COMPUTE_SHADER_BIT|EPSF_TRANSFER_BIT|EPSF_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|EPSF_COMMAND_PREPROCESS_BIT_NV;
                                if (stageMask.value&kDisallowedFlags)
                                    return false;
                            }
                            return true;
                        }
#endif
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
            uint32_t depthStencilAttachmentCount = 0u;
            uint32_t colorAttachmentCount = 0u;
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
            
            core::visit_token_terminated_array(params.depthStencilAttachments,SCreationParams::DepthStencilAttachmentsEnd,[&params,&retval](const SCreationParams::SDepthStencilAttachmentDescription& subpass)->bool
            {
                retval.colorAttachmentCount++;
            });
            core::visit_token_terminated_array(params.colorAttachments,SCreationParams::ColorAttachmentsEnd,[&params,&retval](const SCreationParams::SColorAttachmentDescription& subpass)->bool
            {
                retval.colorAttachmentCount++;
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
                constexpr bool IsDepth = std::remove_reference_t<decltype(renderAttachmentRef)>::IsDepthStencil;
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
                    }
                }
                return false;
            };
            core::visit_token_terminated_array(params.subpasses,SCreationParams::SubpassesEnd,[&params,setRetvalFalse,&retval,subpassesHaveViewMasks,invalidRenderAttachmentRef](const SCreationParams::SSubpassDescription& subpass)->bool
            {
                if (!subpass.valid())
                    return setRetvalFalse();

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
                core::visit_token_terminated_array(subpass.inputAttachments,SCreationParams::SSubpassDescription::InputAttachmentsEnd,[&](const SCreationParams::SSubpassDescription::SInputAttachmentRef& inputAttachmentRef)->bool
                {
                    if (inputAttachmentRef.attachmentIndex!=SCreationParams::SSubpassDescription::AttachmentUnused)
                    {
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-03051
                        if (inputAttachmentRef.attachmentIndex>=retval.attachmentCount)
                            return setRetvalFalse();
                        const auto& inputAttachment = params.attachments[inputAttachmentRef.attachmentIndex];
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-pInputAttachments-02897
                        if (isPlanarFormat(inputAttachment.format)) // or some other check?
                            return setRetvalFalse();
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-attachment-02525
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02799
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-02801
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription2.html#VUID-VkSubpassDescription2-attachment-04563
                        if (isDepthOrStencilFormat(inputAttachment.format))
                        {
                            if (isStencilOnlyFormat(inputAttachment.format))
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
        
            core::visit_token_terminated_array(params.dependencies,SCreationParams::DependenciesEnd,[&params,setRetvalFalse,&retval,subpassesHaveViewMasks](const SCreationParams::SSubpassDependency& dependency)->bool
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

        const SCreationParams& getCreationParameters() const { return m_params; }

    protected:
        IRenderpass(const SCreationParams& params, const CreationParamValidationResult& counts);
        virtual ~IRenderpass() {}

        SCreationParams m_params;
        // store for pointers in `m_params`
        using depth_stencil_attachments_array_t = core::smart_refctd_dynamic_array<SCreationParams::SDepthStencilAttachmentDescription>;
        using color_attachments_array_t = core::smart_refctd_dynamic_array<SCreationParams::SColorAttachmentDescription>;
        using subpass_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDescription>;
        using input_attachment_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDescription::SInputAttachmentRef>;
        using preserved_attachment_refs_array_t = core::smart_refctd_dynamic_array<uint32_t>;
        using subpass_deps_array_t = core::smart_refctd_dynamic_array<SCreationParams::SSubpassDependency>;
        depth_stencil_attachments_array_t m_depthStencilAttachments;
        color_attachments_array_t m_colorAttachments;
        subpass_array_t m_subpasses;
        input_attachment_array_t m_inputAttachments;
        preserved_attachment_refs_array_t m_preserveAttachments;
        subpass_deps_array_t m_subpassDependencies;
};

}

#endif
