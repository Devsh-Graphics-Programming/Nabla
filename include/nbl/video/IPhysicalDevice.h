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
            // its 1D because multidimensional workgroups are an illusion
            uint32_t maxOptimallyResidentWorkgroupInvocations;

            uint32_t subgroupSize;

            // These are maximum number of invocations you could expect to execute simultaneously on this device.
            uint32_t maxResidentInvocations;

            // TODO: move the subgroupOps bitflag to `SFeatures`
            // Also isn't there a separate bitflag per subgroup op type?
            core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages;

            uint64_t nonCoherentAtomSize;

            // AccelerationStructure
            uint64_t           maxGeometryCount;
            uint64_t           maxInstanceCount;
            uint64_t           maxPrimitiveCount;
            uint32_t           maxPerStageDescriptorAccelerationStructures;
            uint32_t           maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
            uint32_t           maxDescriptorSetAccelerationStructures;
            uint32_t           maxDescriptorSetUpdateAfterBindAccelerationStructures;
            uint32_t           minAccelerationStructureScratchOffsetAlignment;

            // RayTracingPipeline
            uint32_t           shaderGroupHandleSize;
            uint32_t           maxRayRecursionDepth;
            uint32_t           maxShaderGroupStride;
            uint32_t           shaderGroupBaseAlignment;
            uint32_t           shaderGroupHandleCaptureReplaySize;
            uint32_t           maxRayDispatchInvocationCount;
            uint32_t           shaderGroupHandleAlignment;
            uint32_t           maxRayHitAttributeSize;
        };

        struct SFeatures
        {
            bool robustBufferAccess = false;
            bool imageCubeArray = false;
            bool logicOp = false;
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
            bool multiDrawIndirect = false;

            // RayQuery
            bool rayQuery = false;

            // AccelerationStructure
            bool accelerationStructure = false;
            bool accelerationStructureCaptureReplay = false;
            bool accelerationStructureIndirectBuild = false;
            bool accelerationStructureHostCommands = false;
            bool descriptorBindingAccelerationStructureUpdateAfterBind = false;

            // RayTracingPipeline
            bool rayTracingPipeline = false;
            bool rayTracingPipelineShaderGroupHandleCaptureReplay = false;
            bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false;
            bool rayTracingPipelineTraceRaysIndirect = false;
            bool rayTraversalPrimitiveCulling = false;

            // Fragment Shader Interlock
            bool fragmentShaderSampleInterlock = false;
            bool fragmentShaderPixelInterlock = false;
            bool fragmentShaderShadingRateInterlock = false;
        };

        struct SMemoryProperties
        {
            uint32_t        memoryTypeCount;
            VkMemoryType    memoryTypes[VK_MAX_MEMORY_TYPES];
            uint32_t        memoryHeapCount;
            VkMemoryHeap    memoryHeaps[VK_MAX_MEMORY_HEAPS];
        };

        struct SFormatProperties
        {
            core::bitflag<asset::E_FORMAT_FEATURE> linearTilingFeatures;
            core::bitflag<asset::E_FORMAT_FEATURE> optimalTilingFeatures;
            core::bitflag<asset::E_FORMAT_FEATURE> bufferFeatures;
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

        // these are the defines which shall be added to any IGPUShader which has its source as GLSL
        inline core::SRange<const char* const> getExtraGLSLDefines() const
        {
            const char* const* begin = m_extraGLSLDefines.data();
            return {begin,begin+m_extraGLSLDefines.size()};
        }

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

        virtual SFormatProperties getFormatProperties(asset::E_FORMAT format) const = 0;

    protected:
        IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc);

        virtual core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) = 0;

        bool validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const;

        void addCommonGLSLDefines(std::ostringstream& pool, const bool runningInRenderDoc);

        template<typename... Args>
        inline void addGLSLDefineToPool(std::ostringstream& pool, const char* define, Args&&... args)
        {
            const ptrdiff_t pos = pool.tellp();
            m_extraGLSLDefines.push_back(reinterpret_cast<const char*>(pos));
            pool << define << " ";
            ((pool << std::forward<Args>(args)), ...);
        }
        inline void finalizeGLSLDefinePool(std::ostringstream&& pool)
        {
            m_GLSLDefineStringPool.resize(static_cast<size_t>(pool.tellp())+m_extraGLSLDefines.size());
            const auto data = ptrdiff_t(m_GLSLDefineStringPool.data());

            const auto str = pool.str();
            size_t nullCharsWritten = 0u;
            for (auto i=0u; i<m_extraGLSLDefines.size(); i++)
            {
                auto& dst = m_extraGLSLDefines[i];
                const auto len = (i!=(m_extraGLSLDefines.size()-1u) ? ptrdiff_t(m_extraGLSLDefines[i+1]):str.length())-ptrdiff_t(dst);
                const char* src = str.data()+ptrdiff_t(dst);
                dst += data+(nullCharsWritten++);
                memcpy(const_cast<char*>(dst),src,len);
                const_cast<char*>(dst)[len] = 0;
            }
        }


        core::smart_refctd_ptr<system::ISystem> m_system;
        core::smart_refctd_ptr<asset::IGLSLCompiler> m_GLSLCompiler;

        SLimits m_limits;
        SFeatures m_features;
        SMemoryProperties m_memoryProperties;
        using qfam_props_array_t = core::smart_refctd_dynamic_array<SQueueFamilyProperties>;
        qfam_props_array_t m_qfamProperties;

        core::vector<char> m_GLSLDefineStringPool;
        core::vector<const char*> m_extraGLSLDefines;
};

}

#endif
