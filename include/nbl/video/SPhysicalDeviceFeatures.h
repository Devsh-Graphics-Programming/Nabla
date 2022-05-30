#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_

namespace nbl::video
{

struct SPhysicalDeviceFeatures
{
    /* Vulkan 1.0 Core  */
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

    /* Vulkan 1.1 Core */
    //VkBool32           storageBuffer16BitAccess;
    //VkBool32           uniformAndStorageBuffer16BitAccess;
    //VkBool32           storagePushConstant16;
    //VkBool32           storageInputOutput16;
    //VkBool32           multiview;
    //VkBool32           multiviewGeometryShader;
    //VkBool32           multiviewTessellationShader;
    //VkBool32           variablePointersStorageBuffer;
    //VkBool32           variablePointers;
    //VkBool32           protectedMemory;
    //VkBool32           samplerYcbcrConversion;
    //VkBool32           shaderDrawParameters;

    /* Vulkan 1.2 Core */
    //VkBool32           samplerMirrorClampToEdge;
    //VkBool32           drawIndirectCount;
    //VkBool32           storageBuffer8BitAccess;
    //VkBool32           uniformAndStorageBuffer8BitAccess;
    //VkBool32           storagePushConstant8;
    //VkBool32           shaderBufferInt64Atomics;
    //VkBool32           shaderSharedInt64Atomics;
    //VkBool32           shaderFloat16;
    //VkBool32           shaderInt8;
    //VkBool32           descriptorIndexing;
    //VkBool32           shaderInputAttachmentArrayDynamicIndexing;
    //VkBool32           shaderUniformTexelBufferArrayDynamicIndexing;
    //VkBool32           shaderStorageTexelBufferArrayDynamicIndexing;
    //VkBool32           shaderUniformBufferArrayNonUniformIndexing;
    //VkBool32           shaderSampledImageArrayNonUniformIndexing;
    //VkBool32           shaderStorageBufferArrayNonUniformIndexing;
    //VkBool32           shaderStorageImageArrayNonUniformIndexing;
    //VkBool32           shaderInputAttachmentArrayNonUniformIndexing;
    //VkBool32           shaderUniformTexelBufferArrayNonUniformIndexing;
    //VkBool32           shaderStorageTexelBufferArrayNonUniformIndexing;
    //VkBool32           descriptorBindingUniformBufferUpdateAfterBind;
    //VkBool32           descriptorBindingSampledImageUpdateAfterBind;
    //VkBool32           descriptorBindingStorageImageUpdateAfterBind;
    //VkBool32           descriptorBindingStorageBufferUpdateAfterBind;
    //VkBool32           descriptorBindingUniformTexelBufferUpdateAfterBind;
    //VkBool32           descriptorBindingStorageTexelBufferUpdateAfterBind;
    //VkBool32           descriptorBindingUpdateUnusedWhilePending;
    //VkBool32           descriptorBindingPartiallyBound;
    //VkBool32           descriptorBindingVariableDescriptorCount;
    //VkBool32           runtimeDescriptorArray;
    //VkBool32           samplerFilterMinmax;
    //VkBool32           scalarBlockLayout;
    //VkBool32           imagelessFramebuffer;
    //VkBool32           uniformBufferStandardLayout;
    //VkBool32           shaderSubgroupExtendedTypes;
    //VkBool32           separateDepthStencilLayouts;
    //VkBool32           hostQueryReset;
    //VkBool32           timelineSemaphore;
    //VkBool32           bufferDeviceAddress;
    //VkBool32           bufferDeviceAddressCaptureReplay;
    //VkBool32           bufferDeviceAddressMultiDevice;
    //VkBool32           vulkanMemoryModel;
    //VkBool32           vulkanMemoryModelDeviceScope;
    //VkBool32           vulkanMemoryModelAvailabilityVisibilityChains;
    //VkBool32           shaderOutputViewportIndex;
    //VkBool32           shaderOutputLayer;
    //VkBool32           subgroupBroadcastDynamicId;

    /* Vulkan 1.3 Core */
    //VkBool32           robustImageAccess;
    //VkBool32           inlineUniformBlock;
    //VkBool32           descriptorBindingInlineUniformBlockUpdateAfterBind;
    //VkBool32           pipelineCreationCacheControl;
    //VkBool32           privateData;
    //VkBool32           shaderDemoteToHelperInvocation;
    //VkBool32           shaderTerminateInvocation;
    //VkBool32           subgroupSizeControl;
    //VkBool32           computeFullSubgroups;
    //VkBool32           synchronization2;
    //VkBool32           textureCompressionASTC_HDR;
    //VkBool32           shaderZeroInitializeWorkgroupMemory;
    //VkBool32           dynamicRendering;
    //VkBool32           shaderIntegerDotProduct;
    //VkBool32           maintenance4; -> Doesn't make sense to expose, too vulkan specific

    /* FragmentDensityMapFeaturesEXT */
    //VkBool32           fragmentDensityMap;
    //VkBool32           fragmentDensityMapDynamic;
    //VkBool32           fragmentDensityMapNonSubsampledImages;

    /* FragmentDensityMap2FeaturesEXT */
    //VkBool32           fragmentDensityMapDeferred;

    /* RayQueryFeaturesKHR */
    bool rayQuery = false;
            
    /* AccelerationStructureFeaturesKHR */
    bool accelerationStructure = false;
    bool accelerationStructureCaptureReplay = false;
    bool accelerationStructureIndirectBuild = false;
    bool accelerationStructureHostCommands = false;
    bool descriptorBindingAccelerationStructureUpdateAfterBind = false;
            
    /* RayTracingPipelineFeaturesKHR */
    bool rayTracingPipeline = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplay = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false;
    bool rayTracingPipelineTraceRaysIndirect = false;
    bool rayTraversalPrimitiveCulling = false;
            
    /* FragmentShaderInterlockFeaturesEXT */
    bool fragmentShaderSampleInterlock = false;
    bool fragmentShaderPixelInterlock = false;
    bool fragmentShaderShadingRateInterlock = false;
            
    /* BufferDeviceAddressFeaturesKHR */
    bool bufferDeviceAddress = false;
    //VkBool32           bufferDeviceAddressCaptureReplay;
    //VkBool32           bufferDeviceAddressMultiDevice;
            
    bool drawIndirectCount = false; // TODO(Erfan): Move in 1.2 features
            
    /* Nabla */
    bool dispatchBase = false; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = false;
};

} // nbl::video

#endif
