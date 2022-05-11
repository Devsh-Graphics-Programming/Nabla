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
        //
        virtual E_API_TYPE getAPIType() const = 0;

        enum E_TYPE : uint8_t {
            ET_OTHER = 0,
            ET_INTEGRATED_GPU = 1,
            ET_DISCRETE_GPU = 2,
            ET_VIRTUAL_GPU = 3,
            ET_CPU = 4,
        };

        enum E_DRIVER_ID : uint8_t
        {
            EDI_OTHER = 0,
            EDI_AMD_PROPRIETARY = 1,
            EDI_AMD_OPEN_SOURCE = 2,
            EDI_MESA_RADV = 3,
            EDI_NVIDIA_PROPRIETARY = 4,
            EDI_INTEL_PROPRIETARY_WINDOWS = 5,
            EDI_INTEL_OPEN_SOURCE_MESA = 6,
            EDI_IMAGINATION_PROPRIETARY = 7,
            EDI_QUALCOMM_PROPRIETARY = 8,
            EDI_ARM_PROPRIETARY = 9,
            EDI_GOOGLE_SWIFTSHADER = 10,
            EDI_GGP_PROPRIETARY = 11,
            EDI_BROADCOM_PROPRIETARY = 12,
            EDI_MESA_LLVMPIPE = 13,
            EDI_MOLTENVK = 14,
            EDI_COREAVI_PROPRIETARY = 15,
            EDI_JUICE_PROPRIETARY = 16,
            EDI_VERISILICON_PROPRIETARY = 17,
            EDI_MESA_TURNIP = 18,
            EDI_MESA_V3DV = 19,
            EDI_MESA_PANVK = 20,
            EDI_SAMSUNG_PROPRIETARY = 21,
            EDI_MESA_VENUS = 22,
        };

        //
        struct APIVersion
        {
            uint32_t major : 5;
            uint32_t minor : 5;
            uint32_t patch : 22;
        };

        //
        struct SLimits
        {
            //uint32_t              maxImageDimension1D;
            //uint32_t              maxImageDimension2D;
            //uint32_t              maxImageDimension3D;
            //uint32_t              maxImageDimensionCube;
            uint32_t maxImageArrayLayers;
            uint32_t maxBufferViewSizeTexels;
            uint32_t maxUBOSize;
            uint32_t maxSSBOSize;
            //uint32_t              maxPushConstantsSize;
            //uint32_t              maxMemoryAllocationCount;
            //uint32_t              maxSamplerAllocationCount;
            //VkDeviceSize          bufferImageGranularity;
            //VkDeviceSize          sparseAddressSpaceSize;
            //uint32_t              maxBoundDescriptorSets;
            //uint32_t              maxPerStageDescriptorSamplers;
            //uint32_t              maxPerStageDescriptorUniformBuffers;
            uint32_t maxPerStageDescriptorSSBOs;
            //uint32_t              maxPerStageDescriptorSampledImages;
            //uint32_t              maxPerStageDescriptorStorageImages;
            //uint32_t              maxPerStageDescriptorInputAttachments;
            //uint32_t              maxPerStageResources;
            //uint32_t              maxDescriptorSetSamplers;
            uint32_t maxDescriptorSetUBOs;
            uint32_t maxDescriptorSetDynamicOffsetUBOs;
            uint32_t maxDescriptorSetSSBOs;
            uint32_t maxDescriptorSetDynamicOffsetSSBOs;
            uint32_t maxDescriptorSetImages;
            uint32_t maxDescriptorSetStorageImages;
            //uint32_t              maxDescriptorSetInputAttachments;
            //uint32_t              maxVertexInputAttributes;
            //uint32_t              maxVertexInputBindings;
            //uint32_t              maxVertexInputAttributeOffset;
            //uint32_t              maxVertexInputBindingStride;
            //uint32_t              maxVertexOutputComponents;
            //uint32_t              maxTessellationGenerationLevel;
            //uint32_t              maxTessellationPatchSize;
            //uint32_t              maxTessellationControlPerVertexInputComponents;
            //uint32_t              maxTessellationControlPerVertexOutputComponents;
            //uint32_t              maxTessellationControlPerPatchOutputComponents;
            //uint32_t              maxTessellationControlTotalOutputComponents;
            //uint32_t              maxTessellationEvaluationInputComponents;
            //uint32_t              maxTessellationEvaluationOutputComponents;
            //uint32_t              maxGeometryShaderInvocations;
            //uint32_t              maxGeometryInputComponents;
            //uint32_t              maxGeometryOutputComponents;
            //uint32_t              maxGeometryOutputVertices;
            //uint32_t              maxGeometryTotalOutputComponents;
            //uint32_t              maxFragmentInputComponents;
            //uint32_t              maxFragmentOutputAttachments;
            //uint32_t              maxFragmentDualSrcAttachments;
            //uint32_t              maxFragmentCombinedOutputResources;
            uint32_t maxComputeSharedMemorySize;
            //uint32_t              maxComputeWorkGroupCount[3];
            //uint32_t              maxComputeWorkGroupInvocations;
            uint32_t maxWorkgroupSize[3];
            //uint32_t              subPixelPrecisionBits;
            //uint32_t              subTexelPrecisionBits;
            //uint32_t              mipmapPrecisionBits;
            //uint32_t              maxDrawIndexedIndexValue;
            uint32_t maxDrawIndirectCount;
            //float                 maxSamplerLodBias;
            float    maxSamplerAnisotropyLog2;
            uint32_t maxViewports;
            uint32_t maxViewportDims[2];
            //float                 viewportBoundsRange[2];
            //uint32_t              viewportSubPixelBits;
            //size_t                minMemoryMapAlignment;
            uint32_t bufferViewAlignment;
            uint32_t UBOAlignment;
            uint32_t SSBOAlignment;
            //int32_t               minTexelOffset;
            //uint32_t              maxTexelOffset;
            //int32_t               minTexelGatherOffset;
            //uint32_t              maxTexelGatherOffset;
            //float                 minInterpolationOffset;
            //float                 maxInterpolationOffset;
            //uint32_t              subPixelInterpolationOffsetBits;
            //uint32_t              maxFramebufferWidth;
            //uint32_t              maxFramebufferHeight;
            //uint32_t              maxFramebufferLayers;
            //VkSampleCountFlags    framebufferColorSampleCounts;
            //VkSampleCountFlags    framebufferDepthSampleCounts;
            //VkSampleCountFlags    framebufferStencilSampleCounts;
            //VkSampleCountFlags    framebufferNoAttachmentsSampleCounts;
            //uint32_t              maxColorAttachments;
            //VkSampleCountFlags    sampledImageColorSampleCounts;
            //VkSampleCountFlags    sampledImageIntegerSampleCounts;
            //VkSampleCountFlags    sampledImageDepthSampleCounts;
            //VkSampleCountFlags    sampledImageStencilSampleCounts;
            //VkSampleCountFlags    storageImageSampleCounts;
            //uint32_t              maxSampleMaskWords;
            //VkBool32              timestampComputeAndGraphics;
            float    timestampPeriodInNanoSeconds; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
            //uint32_t              maxClipDistances;
            //uint32_t              maxCullDistances;
            //uint32_t              maxCombinedClipAndCullDistances;
            //uint32_t              discreteQueuePriorities;
            float pointSizeRange[2];
            float lineWidthRange[2];
            //float                 pointSizeGranularity;
            //float                 lineWidthGranularity;
            //VkBool32              strictLines;
            //VkBool32              standardSampleLocations;
            //VkDeviceSize          optimalBufferCopyOffsetAlignment;
            //VkDeviceSize          optimalBufferCopyRowPitchAlignment;
            uint64_t nonCoherentAtomSize;

            //--> VkPhysicalDeviceSubgroupProperties
            uint32_t subgroupSize;
            core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages;
            //VkSubgroupFeatureFlags    supportedOperations; -> in SFeatures as booleans instead of flags
            //VkBool32                  quadOperationsInAllStages;
            
            //--> VkPhysicalDeviceAccelerationStructurePropertiesKHR
            uint64_t           maxGeometryCount;
            uint64_t           maxInstanceCount;
            uint64_t           maxPrimitiveCount;
            uint32_t           maxPerStageDescriptorAccelerationStructures;
            uint32_t           maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
            uint32_t           maxDescriptorSetAccelerationStructures;
            uint32_t           maxDescriptorSetUpdateAfterBindAccelerationStructures;
            uint32_t           minAccelerationStructureScratchOffsetAlignment;

            //--> VkPhysicalDeviceRayTracingPipelinePropertiesKHR
            uint32_t           shaderGroupHandleSize;
            uint32_t           maxRayRecursionDepth;
            uint32_t           maxShaderGroupStride;
            uint32_t           shaderGroupBaseAlignment;
            uint32_t           shaderGroupHandleCaptureReplaySize;
            uint32_t           maxRayDispatchInvocationCount;
            uint32_t           shaderGroupHandleAlignment;
            uint32_t           maxRayHitAttributeSize;

            //--> Nabla:
            uint32_t maxBufferSize;
            uint64_t maxTextureSize; // TODO: Use maxImageDimensions1D/2D/3D/Cube instead for gl and get rid of this
            uint32_t maxOptimallyResidentWorkgroupInvocations = 0u; //  its 1D because multidimensional workgroups are an illusion
            uint32_t maxResidentInvocations = 0u; //  These are maximum number of invocations you could expect to execute simultaneously on this device.
            asset::IGLSLCompiler::E_SPIRV_VERSION spirvVersion;

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
        
        struct SProperties
        {
            //--> VkPhysicalDeviceProperties:
            APIVersion  apiVersion;
            // uint32_t driverVersion;
            // uint32_t vendorID;
            // uint32_t deviceID;
            E_TYPE      deviceType;
            // char     deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
            // uint8_t  pipelineCacheUUID[VK_UUID_SIZE];
            SLimits     limits;
            // VkPhysicalDeviceSparseProperties    sparseProperties;

            //--> VkPhysicalDeviceDriverProperties
            E_DRIVER_ID driverID;
            // char driverName[VK_MAX_DRIVER_NAME_SIZE];
            // char driverInfo[VK_MAX_DRIVER_INFO_SIZE];
            // VkConformanceVersion conformanceVersion;

            //--> VkPhysicalDeviceIDProperties
            uint8_t deviceUUID[VK_UUID_SIZE];
            // uint8_t driverUUID[VK_UUID_SIZE];
            // uint8_t deviceLUID[VK_LUID_SIZE];
            // uint32_t deviceNodeMask;
            // VkBool32 deviceLUIDValid;
        };

        const SProperties& getProperties() const { return m_properties; }
        const SLimits& getLimits() const { return m_properties.limits; }
        APIVersion getAPIVersion() const { return m_properties.apiVersion; }

        //
        struct SFeatures
        {
            bool robustBufferAccess = false;
            //VkBool32    fullDrawIndexUint32;
            bool imageCubeArray = false;
            //VkBool32    independentBlend;
            bool geometryShader    = false;
            //VkBool32    tessellationShader;
            //VkBool32    sampleRateShading;
            //VkBool32    dualSrcBlend;
            bool logicOp = false;
            bool multiDrawIndirect = false;
            //VkBool32    drawIndirectFirstInstance;
            //VkBool32    depthClamp;
            //VkBool32    depthBiasClamp;
            //VkBool32    fillModeNonSolid;
            //VkBool32    depthBounds;
            //VkBool32    wideLines;
            //VkBool32    largePoints;
            //VkBool32    alphaToOne;
            bool multiViewport = false;
            bool samplerAnisotropy = false;
            //VkBool32    textureCompressionETC2;
            //VkBool32    textureCompressionASTC_LDR;
            //VkBool32    textureCompressionBC;
            //VkBool32    occlusionQueryPrecise;
            //VkBool32    pipelineStatisticsQuery;
            //VkBool32    vertexPipelineStoresAndAtomics;
            //VkBool32    fragmentStoresAndAtomics;
            //VkBool32    shaderTessellationAndGeometryPointSize;
            //VkBool32    shaderImageGatherExtended;
            //VkBool32    shaderStorageImageExtendedFormats;
            //VkBool32    shaderStorageImageMultisample;
            //VkBool32    shaderStorageImageReadWithoutFormat;
            //VkBool32    shaderStorageImageWriteWithoutFormat;
            //VkBool32    shaderUniformBufferArrayDynamicIndexing;
            //VkBool32    shaderSampledImageArrayDynamicIndexing;
            //VkBool32    shaderStorageBufferArrayDynamicIndexing;
            //VkBool32    shaderStorageImageArrayDynamicIndexing;
            //VkBool32    shaderClipDistance;
            //VkBool32    shaderCullDistance;
            bool vertexAttributeDouble = false; // shaderFloat64
            //VkBool32    shaderInt64;
            //VkBool32    shaderInt16;
            //VkBool32    shaderResourceResidency;
            //VkBool32    shaderResourceMinLod;
            //VkBool32    sparseBinding;
            //VkBool32    sparseResidencyBuffer;
            //VkBool32    sparseResidencyImage2D;
            //VkBool32    sparseResidencyImage3D;
            //VkBool32    sparseResidency2Samples;
            //VkBool32    sparseResidency4Samples;
            //VkBool32    sparseResidency8Samples;
            //VkBool32    sparseResidency16Samples;
            //VkBool32    sparseResidencyAliased;
            //VkBool32    variableMultisampleRate;
            bool inheritedQueries = false;

            //--> VkPhysicalDeviceSubgroupProperties: // TODO(Erfan): I think we should move these into SProperties::SLimits since it's part of properties and not features
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

            //--> VkPhysicalDeviceRayQueryFeaturesKHR
            bool rayQuery = false;

            //--> VkPhysicalDeviceAccelerationStructureFeaturesKHR
            bool accelerationStructure = false;
            bool accelerationStructureCaptureReplay = false;
            bool accelerationStructureIndirectBuild = false;
            bool accelerationStructureHostCommands = false;
            bool descriptorBindingAccelerationStructureUpdateAfterBind = false;

            //--> VkPhysicalDeviceRayTracingPipelineFeaturesKHR
            bool rayTracingPipeline = false;
            bool rayTracingPipelineShaderGroupHandleCaptureReplay = false;
            bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false;
            bool rayTracingPipelineTraceRaysIndirect = false;
            bool rayTraversalPrimitiveCulling = false;

            //--> VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT
            bool fragmentShaderSampleInterlock = false;
            bool fragmentShaderPixelInterlock = false;
            bool fragmentShaderShadingRateInterlock = false;


            //--> VkPhysicalDeviceBufferDeviceAddressFeaturesKHR
            bool bufferDeviceAddress = false;
            //VkBool32           bufferDeviceAddressCaptureReplay;
            //VkBool32           bufferDeviceAddressMultiDevice;
            
            //--> Nabla:
            bool dispatchBase = false;
            bool drawIndirectCount = false;
            bool allowCommandBufferQueryCopies = false;
        };
        const SFeatures& getFeatures() const { return m_features; }

        enum E_MEMORY_PROPERTY_FLAGS : uint32_t
        {
            EMPF_DEVICE_LOCAL_BIT = 0x00000001,
            EMPF_HOST_READABLE_BIT = 0x00000002, 
            EMPF_HOST_WRITABLE_BIT = 0x00000004, 
            EMPF_HOST_COHERENT_BIT = 0x00000008,
            EMPF_HOST_CACHED_BIT = 0x000000010,
            //EMPF_LAZILY_ALLOCATED_BIT = 0x00000020,
            //EMPF_PROTECTED_BIT = 0x00000040,
            //EMPF_DEVICE_COHERENT_BIT_AMD = 0x00000080,
            //EMPF_DEVICE_UNCACHED_BIT_AMD = 0x00000100,
            //EMPF_RDMA_CAPABLE_BIT_NV = 0x00000200,
        };

        struct MemoryType
        {
            core::bitflag<E_MEMORY_PROPERTY_FLAGS> propertyFlags;
            uint32_t heapIndex;
        };

        enum E_MEMORY_HEAP_FLAGS : uint32_t
        {
            EMHF_DEVICE_LOCAL_BIT = 0x00000001,
            EMHF_MULTI_INSTANCE_BIT = 0x00000002,
        };

        struct MemoryHeap
        {
            size_t size;
            core::bitflag<E_MEMORY_HEAP_FLAGS> flags;
        };

        //
        struct SMemoryProperties
        {
            uint32_t        memoryTypeCount = 0u;
            MemoryType      memoryTypes[VK_MAX_MEMORY_TYPES];
            uint32_t        memoryHeapCount = 0u;
            MemoryHeap      memoryHeaps[VK_MAX_MEMORY_HEAPS];
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

        SProperties m_properties;
        SFeatures m_features;
        SMemoryProperties m_memoryProperties;
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
