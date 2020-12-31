#ifndef __NBL_I_RENDERPASS_H_INCLUDED__
#define __NBL_I_RENDERPASS_H_INCLUDED__

#include "nbl/asset/IImage.h"
#include "nbl/asset/IDescriptorSet.h" // for E_IMAGE_LAYOUT only
#include "nbl/core/math/glslFunctions.h"

namespace nbl {
namespace asset
{

// TODO probably should be moved to separate header
enum E_PIPELINE_STAGE_FLAGS : uint32_t
{
    EPSF_TOP_OF_PIPE_BIT = 0x00000001,
    EPSF_DRAW_INDIRECT_BIT = 0x00000002,
    EPSF_VERTEX_INPUT_BIT = 0x00000004,
    EPSF_VERTEX_SHADER_BIT = 0x00000008,
    EPSF_TESSELLATION_CONTROL_SHADER_BIT = 0x00000010,
    EPSF_TESSELLATION_EVALUATION_SHADER_BIT = 0x00000020,
    EPSF_GEOMETRY_SHADER_BIT = 0x00000040,
    EPSF_FRAGMENT_SHADER_BIT = 0x00000080,
    EPSF_EARLY_FRAGMENT_TESTS_BIT = 0x00000100,
    EPSF_LATE_FRAGMENT_TESTS_BIT = 0x00000200,
    EPSF_COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400,
    EPSF_COMPUTE_SHADER_BIT = 0x00000800,
    EPSF_TRANSFER_BIT = 0x00001000,
    EPSF_BOTTOM_OF_PIPE_BIT = 0x00002000,
    EPSF_HOST_BIT = 0x00004000,
    EPSF_ALL_GRAPHICS_BIT = 0x00008000,
    EPSF_ALL_COMMANDS_BIT = 0x00010000,
    EPSF_TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
    EPSF_CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
    EPSF_RAY_TRACING_SHADER_BIT_KHR = 0x00200000,
    EPSF_ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000,
    EPSF_SHADING_RATE_IMAGE_BIT_NV = 0x00400000,
    EPSF_TASK_SHADER_BIT_NV = 0x00080000,
    EPSF_MESH_SHADER_BIT_NV = 0x00100000,
    EPSF_FRAGMENT_DENSITY_PROCESS_BIT_EXT = 0x00800000,
    EPSF_COMMAND_PREPROCESS_BIT_NV = 0x00020000
};
// TODO probably should be moved to separate header
enum E_ACCESS_FLAGS : uint32_t
{
    EAF_INDIRECT_COMMAND_READ_BIT = 0x00000001,
    EAF_INDEX_READ_BIT = 0x00000002,
    EAF_VERTEX_ATTRIBUTE_READ_BIT = 0x00000004,
    EAF_UNIFORM_READ_BIT = 0x00000008,
    EAF_INPUT_ATTACHMENT_READ_BIT = 0x00000010,
    EAF_SHADER_READ_BIT = 0x00000020,
    EAF_SHADER_WRITE_BIT = 0x00000040,
    EAF_COLOR_ATTACHMENT_READ_BIT = 0x00000080,
    EAF_COLOR_ATTACHMENT_WRITE_BIT = 0x00000100,
    EAF_DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x00000200,
    EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400,
    EAF_TRANSFER_READ_BIT = 0x00000800,
    EAF_TRANSFER_WRITE_BIT = 0x00001000,
    EAF_HOST_READ_BIT = 0x00002000,
    EAF_HOST_WRITE_BIT = 0x00004000,
    EAF_MEMORY_READ_BIT = 0x00008000,
    EAF_MEMORY_WRITE_BIT = 0x00010000,
    EAF_TRANSFORM_FEEDBACK_WRITE_BIT_EXT = 0x02000000,
    EAF_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT = 0x04000000,
    EAF_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT = 0x08000000,
    EAF_CONDITIONAL_RENDERING_READ_BIT_EXT = 0x00100000,
    EAF_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT = 0x00080000,
    EAF_ACCELERATION_STRUCTURE_READ_BIT_KHR = 0x00200000,
    EAF_ACCELERATION_STRUCTURE_WRITE_BIT_KHR = 0x00400000,
    EAF_SHADING_RATE_IMAGE_READ_BIT_NV = 0x00800000,
    EAF_FRAGMENT_DENSITY_MAP_READ_BIT_EXT = 0x01000000,
    EAF_COMMAND_PREPROCESS_READ_BIT_NV = 0x00020000,
    EAF_COMMAND_PREPROCESS_WRITE_BIT_NV = 0x00040000
};
// TODO probably should be moved to separate header
enum E_DEPENDENCY_FLAGS
{
    EDF_BY_REGION_BIT = 0x01,
    EDF_VIEW_LOCAL_BIT = 0x02,
    EDF_DEVICE_GROUP_BIT = 0x04
};

class IRenderpass
{
public:
    enum E_LOAD_OP : uint8_t
    {
        ELO_LOAD,
        ELO_CLEAR,
        ELO_DONT_CARE
    };
    enum E_STORE_OP : uint8_t
    {
        ESO_STORE,
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
                uint32_t attachment;
                E_IMAGE_LAYOUT layout;
            };

            E_SUBPASS_DESCRIPTION_FLAGS flags = ESDF_NONE;
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

        _NBL_STATIC_INLINE_CONSTEXPR auto MaxColorAttachments = 8u;

        uint32_t attachmentCount;
        const SAttachmentDescription* attachments;
        
        uint32_t subpassCount = 0u;
        const SSubpassDescription* subpasses = nullptr;

        uint32_t dependencyCount = 0u;
        const SSubpassDependency* dependencies = nullptr;
    };

    explicit IRenderpass(const SCreationParams& params) : 
        m_params(params),
        m_attachments(m_params.attachmentCount ? core::make_refctd_dynamic_array<attachments_array_t>(params.attachmentCount):nullptr),
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
            m_attachmentRefs = core::make_smart_refctd_ptr<attachment_refs_array_t>(attRefCnt);
        if (preservedAttRefCnt)
            m_preservedAttachmentRefs = core::make_smart_refctd_ptr<preserved_attachment_refs_array_t>(preservedAttRefCnt);

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
                _COPY_ATTACHMENT_REFS(inputAttachments, inputAttachmentCount);
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

protected:
    virtual ~IRenderpass() = 0;

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
}

#endif