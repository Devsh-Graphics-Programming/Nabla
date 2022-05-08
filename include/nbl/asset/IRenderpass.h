#ifndef __NBL_I_RENDERPASS_H_INCLUDED__
#define __NBL_I_RENDERPASS_H_INCLUDED__

#include "nbl/core/SRange.h"
#include "nbl/core/containers/refctd_dynamic_array.h"
#include "nbl/core/math/glslFunctions.tcc"

#include "nbl/asset/IImage.h"
#include "nbl/asset/EImageLayout.h"
#include "nbl/asset/ECommonEnums.h"

namespace nbl::asset
{

class NBL_API IRenderpass
{
public:
    static constexpr inline uint32_t ATTACHMENT_UNUSED = 0xffFFffFFu;

    enum E_LOAD_OP : uint8_t
    {
        ELO_LOAD = 0,
        ELO_CLEAR,
        ELO_DONT_CARE
    };
    enum E_STORE_OP : uint8_t
    {
        ESO_STORE = 0,
        ESO_DONT_CARE
    };
    enum E_SUBPASS_DESCRIPTION_FLAGS
    {
        ESDF_NONE = 0x00,
        ESDF_PER_VIEW_ATTRIBUTES_BIT = 0x01,
        ESDF_PER_VIEW_POSITION_X_ONLY_BIT = 0x02
    };

    struct SCreationParams
    {
        struct SAttachmentDescription
        {
            E_FORMAT format = EF_UNKNOWN;
            IImage::E_SAMPLE_COUNT_FLAGS samples = IImage::ESCF_1_BIT;
            E_LOAD_OP loadOp = ELO_DONT_CARE;
            E_STORE_OP storeOp = ESO_DONT_CARE;
            E_IMAGE_LAYOUT initialLayout = EIL_UNDEFINED;
            E_IMAGE_LAYOUT finalLayout = EIL_UNDEFINED;
        };

        struct SSubpassDescription
        {
            struct SAttachmentRef
            {
                uint32_t attachment = ATTACHMENT_UNUSED;
                E_IMAGE_LAYOUT layout = EIL_UNDEFINED;
            };

            E_SUBPASS_DESCRIPTION_FLAGS flags = ESDF_NONE;
            E_PIPELINE_BIND_POINT pipelineBindPoint;
            const SAttachmentRef* depthStencilAttachment;
            uint32_t inputAttachmentCount;
            const SAttachmentRef* inputAttachments;
            //! denotes resolve attachment count as well
            uint32_t colorAttachmentCount;
            const SAttachmentRef* colorAttachments;
            const SAttachmentRef* resolveAttachments;
            uint32_t preserveAttachmentCount;
            const uint32_t* preserveAttachments;
        };

        struct SSubpassDependency
        {
            uint32_t srcSubpass;
            uint32_t dstSubpass;
            E_PIPELINE_STAGE_FLAGS srcStageMask;
            E_PIPELINE_STAGE_FLAGS dstStageMask;
            E_ACCESS_FLAGS srcAccessMask;
            E_ACCESS_FLAGS dstAccessMask;
            E_DEPENDENCY_FLAGS dependencyFlags;
        };

        static inline constexpr auto MaxColorAttachments = 8u;

        uint32_t attachmentCount;
        const SAttachmentDescription* attachments;
        
        uint32_t subpassCount = 0u;
        const SSubpassDescription* subpasses = nullptr;

        uint32_t dependencyCount = 0u;
        const SSubpassDependency* dependencies = nullptr;
    };

    explicit IRenderpass(const SCreationParams& params) : 
        m_params(params),
        m_attachments(params.attachmentCount ? core::make_refctd_dynamic_array<attachments_array_t>(params.attachmentCount):nullptr),
        m_subpasses(params.subpassCount ? core::make_refctd_dynamic_array<subpasses_array_t>(params.subpassCount):nullptr),
        m_dependencies(params.dependencyCount ? core::make_refctd_dynamic_array<subpass_deps_array_t>(params.dependencyCount):nullptr)
    {
        if (!params.subpasses)
            return;

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
            if (sb.depthStencilAttachment)
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

                // Todo(achal): It is probably wise to do the existence check on colorAttachements
                // as well since it could be NULL according to the Vulkan spec
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
