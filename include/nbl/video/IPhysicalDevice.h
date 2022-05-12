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

class NBL_API2 IPhysicalDevice : public core::Interface, public core::Unmovable
{
    public:
        //
        virtual E_API_TYPE getAPIType() const = 0;
        
        // TODO: fold into SLimits
        struct APIVersion
        {
            uint32_t major : 5;
            uint32_t minor : 5;
            uint32_t patch : 22;
        };
        const APIVersion& getAPIVersion() const { return m_apiVersion; }
        
        //
        struct SLimits
        {
            uint8_t deviceUUID[VK_UUID_SIZE] = {}; // TODO: implement on Vulkan with VkPhysicalDeviceIDProperties

            //
            uint32_t UBOAlignment;
            uint32_t SSBOAlignment;
            uint32_t bufferViewAlignment;
            float    maxSamplerAnisotropyLog2;
            float    timestampPeriodInNanoSeconds; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future

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
            uint32_t maxDynamicOffsetSSBOs;
            uint32_t maxDynamicOffsetUBOs;
            uint32_t maxTextures;
            uint32_t maxStorageImages;

            uint64_t maxTextureSize;

            uint32_t maxDrawIndirectCount;

            float pointSizeRange[2];
            float lineWidthRange[2];

            uint32_t maxViewports;
            uint32_t maxViewportDims[2];

            uint32_t maxWorkgroupSize[3];
            // its 1D because multidimensional workgroups are an illusion
            uint32_t maxOptimallyResidentWorkgroupInvocations = 0u;

            uint32_t subgroupSize;

            // These are maximum number of invocations you could expect to execute simultaneously on this device.
            uint32_t maxResidentInvocations = 0u;

            // TODO: move the subgroupOps bitflag to `SFeatures`
            // Also isn't there a separate bitflag per subgroup op type?
            core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages;

            uint64_t nonCoherentAtomSize;

            asset::IGLSLCompiler::E_SPIRV_VERSION spirvVersion;

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

            // utility functions
            // In the cases where the workgroups synchronise with each other such as work DAGs (i.e. `CScanner`),
            // `workgroupSpinningProtection` is meant to protect against launching a dispatch so wide that
            // a workgroup of the next cut of the DAG spins for an extended time to wait on a workgroup from a previous one.
            inline uint32_t computeOptimalPersistentWorkgroupDispatchSize(const uint64_t elementCount, const uint32_t workgroupSize, const uint32_t workgroupSpinningProtection=1u) const
            {
                assert(elementCount!=0ull && "Input element count can't be 0!");
                const uint64_t infinitelyWideDeviceWGCount = (elementCount-1ull)/(static_cast<uint64_t>(workgroupSize)*static_cast<uint64_t>(workgroupSpinningProtection))+1ull;
                const uint32_t maxResidentWorkgroups = maxResidentInvocations/workgroupSize;
                return static_cast<uint32_t>(core::min<uint64_t>(infinitelyWideDeviceWGCount,maxResidentWorkgroups));
            }
        };
        const SLimits& getLimits() const { return m_limits; }

        //
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
            bool samplerAnisotropy = false;
            bool geometryShader    = false;

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

            // Queries
            bool allowCommandBufferQueryCopies = false;
            bool inheritedQueries = false;

            // Buffer Device Address
            bool bufferDeviceAddress = false;
        };
        const SFeatures& getFeatures() const { return m_features; }

        //
        struct SMemoryProperties
        {
            uint32_t        memoryTypeCount = 0u;
            VkMemoryType    memoryTypes[VK_MAX_MEMORY_TYPES];
            uint32_t        memoryHeapCount = 0u;
            VkMemoryHeap    memoryHeaps[VK_MAX_MEMORY_HEAPS];
        };
        const SMemoryProperties& getMemoryProperties() const { return m_memoryProperties; }

        //
        struct SFormatBufferUsage
        {
            uint8_t isInitialized : 1u;

            uint8_t vertexAttribute : 1u; // vertexAtrtibute binding
            uint8_t bufferView : 1u; // samplerBuffer
            uint8_t storageBufferView : 1u; // imageBuffer
            uint8_t storageBufferViewAtomic : 1u; // imageBuffer
            uint8_t accelerationStructureVertex : 1u;

            inline SFormatBufferUsage operator & (const SFormatBufferUsage& other) const
            {
                SFormatBufferUsage result;
                result.vertexAttribute = vertexAttribute & other.vertexAttribute;
                result.bufferView = bufferView & other.bufferView;
                result.storageBufferView = storageBufferView & other.storageBufferView;
                result.storageBufferViewAtomic = storageBufferViewAtomic & other.storageBufferViewAtomic;
                result.accelerationStructureVertex = accelerationStructureVertex & other.accelerationStructureVertex;
                return result;
            }

            inline SFormatBufferUsage operator | (const SFormatBufferUsage& other) const
            {
                SFormatBufferUsage result;
                result.vertexAttribute = vertexAttribute | other.vertexAttribute;
                result.bufferView = bufferView | other.bufferView;
                result.storageBufferView = storageBufferView | other.storageBufferView;
                result.storageBufferViewAtomic = storageBufferViewAtomic | other.storageBufferViewAtomic;
                result.accelerationStructureVertex = accelerationStructureVertex | other.accelerationStructureVertex;
                return result;
            }

            inline SFormatBufferUsage operator ^ (const SFormatBufferUsage& other) const
            {
                SFormatBufferUsage result;
                result.vertexAttribute = vertexAttribute ^ other.vertexAttribute;
                result.bufferView = bufferView ^ other.bufferView;
                result.storageBufferView = storageBufferView ^ other.storageBufferView;
                result.storageBufferViewAtomic = storageBufferViewAtomic ^ other.storageBufferViewAtomic;
                result.accelerationStructureVertex = accelerationStructureVertex ^ other.accelerationStructureVertex;
                return result;
            }

            inline bool operator == (const SFormatBufferUsage& other) const
            {
                return
                    (vertexAttribute == other.vertexAttribute) &&
                    (bufferView == other.bufferView) &&
                    (storageBufferView == other.storageBufferView) &&
                    (storageBufferViewAtomic == other.storageBufferViewAtomic) &&
                    (accelerationStructureVertex == other.accelerationStructureVertex);
            }
        };
        virtual const SFormatBufferUsage& getBufferFormatUsages(const asset::E_FORMAT format) = 0;

        //
        struct SFormatImageUsage
        {
            uint8_t isInitialized : 1u;

            uint16_t sampledImage : 1u; // samplerND
            uint16_t storageImage : 1u; // imageND
            uint16_t storageImageAtomic : 1u;
            uint16_t attachment : 1u; // color, depth, stencil can be infferred from the format itself
            uint16_t attachmentBlend : 1u;
            uint16_t blitSrc : 1u;
            uint16_t blitDst : 1u;
            uint16_t transferSrc : 1u;
            uint16_t transferDst : 1u;
            uint16_t log2MaxSamples : 3u; // 0 means cant use as a multisample image format

            inline SFormatImageUsage operator & (const SFormatImageUsage& other) const
            {
                SFormatImageUsage result;
                result.sampledImage = sampledImage & other.sampledImage;
                result.storageImage = storageImage & other.storageImage;
                result.storageImageAtomic = storageImageAtomic & other.storageImageAtomic;
                result.attachment = attachment & other.attachment;
                result.attachmentBlend = attachmentBlend & other.attachmentBlend;
                result.blitSrc = blitSrc & other.blitSrc;
                result.blitDst = blitDst & other.blitDst;
                result.transferSrc = transferSrc & other.transferSrc;
                result.transferDst = transferDst & other.transferDst;
                result.log2MaxSamples = log2MaxSamples & other.log2MaxSamples;
                return result;
            }

            inline SFormatImageUsage operator | (const SFormatImageUsage& other) const
            {
                SFormatImageUsage result;
                result.sampledImage = sampledImage | other.sampledImage;
                result.storageImage = storageImage | other.storageImage;
                result.storageImageAtomic = storageImageAtomic | other.storageImageAtomic;
                result.attachment = attachment | other.attachment;
                result.attachmentBlend = attachmentBlend | other.attachmentBlend;
                result.blitSrc = blitSrc | other.blitSrc;
                result.blitDst = blitDst | other.blitDst;
                result.transferSrc = transferSrc | other.transferSrc;
                result.transferDst = transferDst | other.transferDst;
                result.log2MaxSamples = log2MaxSamples | other.log2MaxSamples;
                return result;
            }

            inline SFormatImageUsage operator ^ (const SFormatImageUsage& other) const
            {
                SFormatImageUsage result;
                result.sampledImage = sampledImage ^ other.sampledImage;
                result.storageImage = storageImage ^ other.storageImage;
                result.storageImageAtomic = storageImageAtomic ^ other.storageImageAtomic;
                result.attachment = attachment ^ other.attachment;
                result.attachmentBlend = attachmentBlend ^ other.attachmentBlend;
                result.blitSrc = blitSrc ^ other.blitSrc;
                result.blitDst = blitDst ^ other.blitDst;
                result.transferSrc = transferSrc ^ other.transferSrc;
                result.transferDst = transferDst ^ other.transferDst;
                result.log2MaxSamples = log2MaxSamples ^ other.log2MaxSamples;
                return result;
            }

            inline bool operator == (const SFormatImageUsage& other) const
            {
                return
                    (sampledImage == other.sampledImage) &&
                    (storageImage == other.storageImage) &&
                    (storageImageAtomic == other.storageImageAtomic) &&
                    (attachment == other.attachment) &&
                    (attachmentBlend == other.attachmentBlend) &&
                    (blitSrc == other.blitSrc) &&
                    (blitDst == other.blitDst) &&
                    (transferSrc == other.transferSrc) &&
                    (transferDst == other.transferDst) &&
                    (log2MaxSamples == other.log2MaxSamples);
            }
        };
        virtual const SFormatImageUsage& getImageFormatUsagesLinear(const asset::E_FORMAT format) = 0;
        virtual const SFormatImageUsage& getImageFormatUsagesOptimal(const asset::E_FORMAT format) = 0;

        //
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

            inline bool operator!=(const SQueueFamilyProperties& other) const
            {
                return queueFlags.value != other.queueFlags.value || queueCount != other.queueCount || timestampValidBits != other.timestampValidBits || minImageTransferGranularity != other.minImageTransferGranularity;
            }
        };
        auto getQueueFamilyProperties() const 
        {
            using citer_t = qfam_props_array_t::pointee::const_iterator;
            return core::SRange<const SQueueFamilyProperties, citer_t, citer_t>(
                m_qfamProperties->cbegin(),
                m_qfamProperties->cend()
            );
        }

        // these are the defines which shall be added to any IGPUShader which has its source as GLSL
        inline core::SRange<const char* const> getExtraGLSLDefines() const
        {
            const char* const* begin = m_extraGLSLDefines.data();
            return {begin,begin+m_extraGLSLDefines.size()};
        }

        //
        inline system::ISystem* getSystem() const {return m_system.get();}
        inline asset::IGLSLCompiler* getGLSLCompiler() const {return m_GLSLCompiler.get();}

        virtual IDebugCallback* getDebugCallback() = 0;

        // TODO: shouldn't this be in SFeatures?
        virtual bool isSwapchainSupported() const = 0;

        core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice(const ILogicalDevice::SCreationParams& params)
        {
            if (!validateLogicalDeviceCreation(params))
                return nullptr;

            return createLogicalDevice_impl(params);
        }

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
        APIVersion m_apiVersion;
        using qfam_props_array_t = core::smart_refctd_dynamic_array<SQueueFamilyProperties>;
        qfam_props_array_t m_qfamProperties;

        SFormatImageUsage m_linearTilingUsages[asset::EF_UNKNOWN] = {};
        SFormatImageUsage m_optimalTilingUsages[asset::EF_UNKNOWN] = {};
        SFormatBufferUsage m_bufferUsages[asset::EF_UNKNOWN] = {};

        core::vector<char> m_GLSLDefineStringPool;
        core::vector<const char*> m_extraGLSLDefines;
};

}

#endif
