#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

IPhysicalDevice::IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc) :
    m_system(std::move(s)), m_GLSLCompiler(std::move(glslc))
{
}

void IPhysicalDevice::addCommonGLSLDefines(std::ostringstream& pool, const bool runningInRenderdoc)
{
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_UBO_SIZE",m_limits.maxUBOSize);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_SSBO_SIZE",m_limits.maxSSBOSize);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_BUFFER_VIEW_TEXELS",m_limits.maxBufferViewSizeTexels);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_BUFFER_SIZE",m_limits.maxBufferSize);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_IMAGE_ARRAY_LAYERS",m_limits.maxImageArrayLayers);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_PER_STAGE_SSBO_COUNT",m_limits.maxPerStageSSBOs);
    
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_SSBO_COUNT",m_limits.maxSSBOs);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_UBO_COUNT",m_limits.maxUBOs);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_TEXTURE_COUNT",m_limits.maxTextures);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_STORAGE_IMAGE_COUNT",m_limits.maxStorageImages);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_DRAW_INDIRECT_COUNT",m_limits.maxDrawIndirectCount);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MIN_POINT_SIZE",m_limits.pointSizeRange[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_POINT_SIZE",m_limits.pointSizeRange[1]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MIN_LINE_WIDTH",m_limits.lineWidthRange[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_LINE_WIDTH",m_limits.lineWidthRange[1]);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_VIEWPORTS",m_limits.maxViewports);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_VIEWPORT_WIDTH",m_limits.maxViewportDims[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_VIEWPORT_HEIGHT",m_limits.maxViewportDims[1]);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_WORKGROUP_SIZE_X",m_limits.maxWorkgroupSize[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_WORKGROUP_SIZE_Y",m_limits.maxWorkgroupSize[1]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_WORKGROUP_SIZE_Z",m_limits.maxWorkgroupSize[2]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS",m_limits.maxOptimallyResidentWorkgroupInvocations);

    // TODO: Need upper and lower bounds on workgroup sizes!
    // TODO: Need to know if subgroup size is constant/known
    addGLSLDefineToPool(pool,"NBL_LIMIT_SUBGROUP_SIZE",m_limits.subgroupSize);
    // TODO: @achal test examples 14 and 48 on all APIs and GPUs
    
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_RESIDENT_INVOCATIONS",m_limits.maxResidentInvocations);


    // TODO: Add feature defines


    if (runningInRenderdoc)
        addGLSLDefineToPool(pool,"NBL_RUNNING_IN_RENDERDOC");
}

bool IPhysicalDevice::validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const
{
    using range_t = core::SRange<const ILogicalDevice::SQueueCreationParams>;
    range_t qcis(params.queueParams, params.queueParams+params.queueParamsCount);

    for (const auto& qci : qcis)
    {
        if (qci.familyIndex >= m_qfamProperties->size())
            return false;

        const auto& qfam = (*m_qfamProperties)[qci.familyIndex];
        if (qci.count == 0u)
            return false;
        if (qci.count > qfam.queueCount)
            return false;

        for (uint32_t i = 0u; i < qci.count; ++i)
        {
            const float priority = qci.priorities[i];
            if (priority < 0.f)
                return false;
            if (priority > 1.f)
                return false;
        }
    }

    return true;
}

}