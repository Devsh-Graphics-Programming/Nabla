#ifndef __NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/system/declarations.h"

#include <type_traits>

#include "nbl/asset/IImage.h" //for VkExtent3D only
#include "nbl/asset/ISpecializedShader.h"
#include "nbl/asset/utils/IGLSLCompiler.h"

#include "nbl/system/ISystem.h"

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"
#include "nbl/video/ILogicalDevice.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class IPhysicalDevice : public core::Interface, public core::Unmovable
{
public:
    struct SLimits
    {
        uint32_t UBOAlignment;
        uint32_t SSBOAlignment;
        uint32_t bufferViewAlignment;

        uint32_t maxUBOSize;
        uint32_t maxSSBOSize;
        uint32_t maxBufferViewSizeTexels;
        uint32_t maxBufferSize;

        uint32_t maxImageArrayLayers;

        uint32_t maxPerStageSSBOs;
        //uint32_t maxPerStageUBOs;
        //uint32_t maxPerStageTextures;
        //uint32_t maxPerStageStorageImages;

        uint32_t maxSSBOs;
        uint32_t maxUBOs;
        uint32_t maxTextures;
        uint32_t maxStorageImages;

        uint32_t maxDrawIndirectCount;

        float pointSizeRange[2];
        float lineWidthRange[2];

        uint32_t maxViewports;
        uint32_t maxViewportDims[2];

        uint32_t maxWorkgroupSize[3];

        uint32_t subgroupSize;
        core::bitflag<asset::ISpecializedShader::E_SHADER_STAGE> subgroupOpsShaderStages;
    };

    struct SFeatures
    {
        bool robustBufferAccess = false;
        bool imageCubeArray = false;
        bool logicOp = false;
        bool multiDrawIndirect = false;
        bool multiViewport = false;
        bool vertexAttributeDouble = false;
        bool dispatchBase = false;
        bool shaderSubgroupBasic = false;
        bool shaderSubgroupVote = false;
        bool shaderSubgroupArithmetic = false;
        bool shaderSubgroupBallot = false;
        bool shaderSubgroupShuffle = false;
        bool shaderSubgroupShuffleRelative = false;
        bool shaderSubgroupClustered = false;
        bool shaderSubgroupQuad = false;
        // Whether `shaderSubgroupQuad` flag refer to all stages where subgroup ops are reported to be supported.
        // See SLimit::subgroupOpsShaderStages.
        bool shaderSubgroupQuadAllStages = false;
        bool drawIndirectCount = false;
        bool rayQuery = false;
        bool accelerationStructure = false;
        bool accelerationStructureCaptureReplay = false;
        bool accelerationStructureIndirectBuild = false;
        bool accelerationStructureHostCommands = false;
        bool descriptorBindingAccelerationStructureUpdateAfterBind = false;
        bool allowCommandBufferQueryCopies = false;
    };

    struct SMemoryProperties
    {
        uint32_t        memoryTypeCount;
        VkMemoryType    memoryTypes[VK_MAX_MEMORY_TYPES];
        uint32_t        memoryHeapCount;
        VkMemoryHeap    memoryHeaps[VK_MAX_MEMORY_HEAPS];
    };

    enum E_QUEUE_FLAGS : uint32_t
    {
        EQF_GRAPHICS_BIT = 0x01,
        EQF_COMPUTE_BIT = 0x02,
        EQF_TRANSFER_BIT = 0x04,
        EQF_SPARSE_BINDING_BIT = 0x08,
        EQF_PROTECTED_BIT = 0x10
    };
    struct SQueueFamilyProperties
    {
        core::bitflag<E_QUEUE_FLAGS> queueFlags;
        uint32_t queueCount;
        uint32_t timestampValidBits;
        asset::VkExtent3D minImageTransferGranularity;
    };

    const SLimits& getLimits() const { return m_limits; }
    const SFeatures& getFeatures() const { return m_features; }
    const SMemoryProperties& getMemoryProperties() const { return m_memoryProperties; }

    auto getQueueFamilyProperties() const 
    {
        using citer_t = qfam_props_array_t::pointee::const_iterator;
        return core::SRange<const SQueueFamilyProperties, citer_t, citer_t>(
            m_qfamProperties->cbegin(),
            m_qfamProperties->cend()
        );
    }

    inline system::ISystem* getSystem() const {return m_system.get();}
    inline asset::IGLSLCompiler* getGLSLCompiler() const {return m_GLSLCompiler.get();}

    virtual IDebugCallback* getDebugCallback() = 0;

    virtual bool isSwapchainSupported() const = 0;

    core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice(const ILogicalDevice::SCreationParams& params)
    {
        if (!validateLogicalDeviceCreation(params))
            return nullptr;

        return createLogicalDevice_impl(params);
    }

    virtual E_API_TYPE getAPIType() const = 0;

protected:
    IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc);

    virtual core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) = 0;

    bool validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const;


    core::smart_refctd_ptr<system::ISystem> m_system;
    core::smart_refctd_ptr<asset::IGLSLCompiler> m_GLSLCompiler;

    SLimits m_limits;
    SFeatures m_features;
    SMemoryProperties m_memoryProperties;
    using qfam_props_array_t = core::smart_refctd_dynamic_array<SQueueFamilyProperties>;
    qfam_props_array_t m_qfamProperties;
};

}

#endif
