// constexprs
NBL_CONSTEXPR_STATIC_INLINE uint32_t MinMaxImageDimension2D = 1 << 14;
NBL_CONSTEXPR_STATIC_INLINE uint32_t MinMaxSSBOSize = (0x1u << 30u) - 4;
NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxMaxPushConstantsSize = 256;
NBL_CONSTEXPR_STATIC_INLINE uint32_t MinMaxWorkgroupCount = 1 << 12;
NBL_CONSTEXPR_STATIC_INLINE uint32_t MinMaxWorkgroupInvocations = 256;
NBL_CONSTEXPR_STATIC_INLINE int32_t MinSubPixelInterpolationOffsetBits = 4;
NBL_CONSTEXPR_STATIC_INLINE uint16_t MinMaxColorAttachments = 8;

// Limits Defaults
// VK 1.0
NBL_GENERATE_GET_OR_DEFAULT(maxImageDimension1D, uint32_t, MinMaxImageDimension2D);
NBL_GENERATE_GET_OR_DEFAULT(maxImageDimension2D, uint32_t, MinMaxImageDimension2D);
NBL_GENERATE_GET_OR_DEFAULT(maxImageDimension3D, uint32_t, 1 << 11);
NBL_GENERATE_GET_OR_DEFAULT(maxImageDimensionCube, uint32_t, MinMaxImageDimension2D);
NBL_GENERATE_GET_OR_DEFAULT(maxImageArrayLayers, uint32_t, 1 << 11);
NBL_GENERATE_GET_OR_DEFAULT(maxBufferViewTexels, uint32_t, 1u << 25);
NBL_GENERATE_GET_OR_DEFAULT(maxUBOSize, uint32_t, 1u << 16);
NBL_GENERATE_GET_OR_DEFAULT(maxSSBOSize, uint32_t, MinMaxSSBOSize);
NBL_GENERATE_GET_OR_DEFAULT(maxPushConstantsSize, uint16_t, 128);
NBL_GENERATE_GET_OR_DEFAULT(maxMemoryAllocationCount, uint32_t, 1 << 12);
NBL_GENERATE_GET_OR_DEFAULT(maxSamplerAllocationCount, uint32_t, 4000);
NBL_GENERATE_GET_OR_DEFAULT(bufferImageGranularity, uint32_t, 1u << 16);

NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorSamplers, uint32_t, 16);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUBOs, uint32_t, 15);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorSSBOs, uint32_t, 31);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorImages, uint32_t, 96);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorStorageImages, uint32_t, 8);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorInputAttachments, uint32_t, 7);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageResources, uint32_t, 127);

NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetSamplers, uint32_t, 80);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUBOs, uint32_t, 90);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetDynamicOffsetUBOs, uint32_t, 8);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetSSBOs, uint32_t, 155);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetDynamicOffsetSSBOs, uint32_t, 8);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetImages, uint32_t, 480);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetStorageImages, uint32_t, 40);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetInputAttachments, uint32_t, 7);

NBL_GENERATE_GET_OR_DEFAULT(maxVertexOutputComponents, uint16_t, 124);

NBL_GENERATE_GET_OR_DEFAULT(maxTessellationGenerationLevel, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxTessellationPatchSize, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxTessellationControlPerVertexInputComponents, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxTessellationControlPerVertexOutputComponents, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxTessellationControlPerPatchOutputComponents, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxTessellationControlTotalOutputComponents, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxTessellationEvaluationInputComponents, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxTessellationEvaluationOutputComponents, uint16_t, 0);

NBL_GENERATE_GET_OR_DEFAULT(maxGeometryShaderInvocations, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxGeometryInputComponents, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxGeometryOutputComponents, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxGeometryOutputVertices, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxGeometryTotalOutputComponents, uint16_t, 0);

NBL_GENERATE_GET_OR_DEFAULT(maxFragmentInputComponents, uint32_t, 116);
NBL_GENERATE_GET_OR_DEFAULT(maxFragmentOutputAttachments, uint32_t, 8);
NBL_GENERATE_GET_OR_DEFAULT(maxFragmentDualSrcAttachments, uint32_t, 1);
NBL_GENERATE_GET_OR_DEFAULT(maxFragmentCombinedOutputResources, uint32_t, 16);

NBL_GENERATE_GET_OR_DEFAULT(maxComputeSharedMemorySize, uint32_t, 1 << 15);
NBL_GENERATE_GET_OR_DEFAULT(maxComputeWorkGroupCountX, uint32_t, MinMaxWorkgroupCount);
NBL_GENERATE_GET_OR_DEFAULT(maxComputeWorkGroupCountY, uint32_t, MinMaxWorkgroupCount);
NBL_GENERATE_GET_OR_DEFAULT(maxComputeWorkGroupCountZ, uint32_t, MinMaxWorkgroupCount);
NBL_GENERATE_GET_OR_DEFAULT(maxComputeWorkGroupInvocations, uint16_t, MinMaxWorkgroupInvocations);
NBL_GENERATE_GET_OR_DEFAULT(maxWorkgroupSizeX, uint16_t, MinMaxWorkgroupInvocations);
NBL_GENERATE_GET_OR_DEFAULT(maxWorkgroupSizeY, uint16_t, MinMaxWorkgroupInvocations);
NBL_GENERATE_GET_OR_DEFAULT(maxWorkgroupSizeZ, uint16_t, 64u);

NBL_GENERATE_GET_OR_DEFAULT(subPixelPrecisionBits, uint16_t, 4);
NBL_GENERATE_GET_OR_DEFAULT(subTexelPrecisionBits, uint16_t, 4);
NBL_GENERATE_GET_OR_DEFAULT(mipmapPrecisionBits, uint16_t, 4);

NBL_GENERATE_GET_OR_DEFAULT(maxDrawIndirectCount, uint32_t, 0x1u << 30);

NBL_GENERATE_GET_OR_DEFAULT(maxSamplerLodBiasBitPattern, uint32_t, asuint(4.f));
NBL_GENERATE_GET_OR_DEFAULT(maxSamplerAnisotropyLog2, uint16_t, 4);

NBL_GENERATE_GET_OR_DEFAULT(maxViewports, uint16_t, 16);
NBL_GENERATE_GET_OR_DEFAULT(maxViewportDimsX, uint16_t, MinMaxImageDimension2D);
NBL_GENERATE_GET_OR_DEFAULT(maxViewportDimsY, uint16_t, MinMaxImageDimension2D);
NBL_GENERATE_GET_OR_DEFAULT(viewportBoundsRangeBitPatternMin, uint32_t, asuint(-MinMaxImageDimension2D*2u));
NBL_GENERATE_GET_OR_DEFAULT(viewportBoundsRangeBitPatternMax, uint32_t, asuint(MinMaxImageDimension2D*2u-1));
NBL_GENERATE_GET_OR_DEFAULT(viewportSubPixelBits, uint32_t, 0);

NBL_GENERATE_GET_OR_DEFAULT(minMemoryMapAlignment, uint16_t, 64);
NBL_GENERATE_GET_OR_DEFAULT(bufferViewAlignment, uint16_t, 64);
NBL_GENERATE_GET_OR_DEFAULT(minUBOAlignment, uint16_t, 256);
NBL_GENERATE_GET_OR_DEFAULT(minSSBOAlignment, uint16_t, 64);

NBL_GENERATE_GET_OR_DEFAULT(minTexelOffset, int16_t, -8);
NBL_GENERATE_GET_OR_DEFAULT(maxTexelOffset, uint16_t, 7);
NBL_GENERATE_GET_OR_DEFAULT(minTexelGatherOffset, int16_t, -8);
NBL_GENERATE_GET_OR_DEFAULT(maxTexelGatherOffset, uint16_t, 7);

NBL_GENERATE_GET_OR_DEFAULT(minInterpolationOffsetBitPattern, uint32_t, asuint(-0.5f));
NBL_GENERATE_GET_OR_DEFAULT(maxInterpolationOffsetBitPattern, uint32_t, asuint(0.4375));
NBL_GENERATE_GET_OR_DEFAULT(subPixelInterpolationOffsetBits, uint16_t, MinSubPixelInterpolationOffsetBits);

NBL_GENERATE_GET_OR_DEFAULT(maxFramebufferWidth, uint32_t, MinMaxImageDimension2D);
NBL_GENERATE_GET_OR_DEFAULT(maxFramebufferHeight, uint32_t, MinMaxImageDimension2D);
NBL_GENERATE_GET_OR_DEFAULT(maxFramebufferLayers, uint32_t, 1 << 10);

NBL_GENERATE_GET_OR_DEFAULT(maxColorAttachments, uint16_t, MinMaxColorAttachments);

NBL_GENERATE_GET_OR_DEFAULT(maxSampleMaskWords, uint16_t, 1);

NBL_GENERATE_GET_OR_DEFAULT(timestampPeriodInNanoSecondsBitPattern, uint32_t, asuint(83.334f));

NBL_GENERATE_GET_OR_DEFAULT(maxClipDistances, uint16_t, 8);
NBL_GENERATE_GET_OR_DEFAULT(maxCullDistances, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxCombinedClipAndCullDistances, uint16_t, 8);

NBL_GENERATE_GET_OR_DEFAULT(discreteQueuePriorities, uint32_t, 2);

NBL_GENERATE_GET_OR_DEFAULT(pointSizeRangeBitPatternMin, uint32_t, asuint(1.f));
NBL_GENERATE_GET_OR_DEFAULT(pointSizeRangeBitPatternMax, uint32_t, asuint(64.f));
NBL_GENERATE_GET_OR_DEFAULT(lineWidthRangeBitPatternMin, uint32_t, asuint(1.f));
NBL_GENERATE_GET_OR_DEFAULT(lineWidthRangeBitPatternMax, uint32_t, asuint(1.f));
NBL_GENERATE_GET_OR_DEFAULT(pointSizeGranularityBitPattern, uint32_t, asuint(1.f));
NBL_GENERATE_GET_OR_DEFAULT(lineWidthGranularityBitPattern, uint32_t, asuint(1.f));

NBL_GENERATE_GET_OR_DEFAULT(strictLines, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(standardSampleLocations, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(optimalBufferCopyOffsetAlignment, uint16_t, 256);
NBL_GENERATE_GET_OR_DEFAULT(optimalBufferCopyRowPitchAlignment, uint16_t, 128);
NBL_GENERATE_GET_OR_DEFAULT(nonCoherentAtomSize, uint16_t, 256);

// VK 1.1
NBL_GENERATE_GET_OR_DEFAULT(subgroupSize, uint16_t, 4);
NBL_GENERATE_GET_OR_DEFAULT(subgroupOpsShaderStagesBitPattern, uint64_t, nbl::hlsl::ShaderStage::ESS_COMPUTE | nbl::hlsl::ShaderStage::ESS_ALL_GRAPHICS);
NBL_GENERATE_GET_OR_DEFAULT(shaderSubgroupClustered, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSubgroupArithmetic, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSubgroupQuad, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSubgroupQuadAllStages, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(pointClippingBehaviorBitPattern, uint64_t, nbl::hlsl::PointClippingBehavior::EPCB_USER_CLIP_PLANES_ONLY);

NBL_GENERATE_GET_OR_DEFAULT(maxMultiviewViewCount, uint16_t, 6);
NBL_GENERATE_GET_OR_DEFAULT(maxMultiviewInstanceIndex, uint32_t, (1u << 27) - 1);

NBL_GENERATE_GET_OR_DEFAULT(maxPerSetDescriptors, uint32_t, 572);
NBL_GENERATE_GET_OR_DEFAULT(maxMemoryAllocationSize, uint64_t, MinMaxSSBOSize);

// VK 1.2
NBL_GENERATE_GET_OR_DEFAULT(shaderSignedZeroInfNanPreserveFloat64, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDenormPreserveFloat16, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDenormPreserveFloat32, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDenormPreserveFloat64, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDenormFlushToZeroFloat16, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDenormFlushToZeroFloat32, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDenormFlushToZeroFloat64, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderRoundingModeRTEFloat16, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderRoundingModeRTEFloat32, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderRoundingModeRTEFloat64, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderRoundingModeRTZFloat16, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderRoundingModeRTZFloat32, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderRoundingModeRTZFloat64, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(maxUpdateAfterBindDescriptorsInAllPools, uint32_t, 0x1u << 20);
NBL_GENERATE_GET_OR_DEFAULT(shaderUniformBufferArrayNonUniformIndexingNative, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSampledImageArrayNonUniformIndexingNative, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderStorageBufferArrayNonUniformIndexingNative, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderStorageImageArrayNonUniformIndexingNative, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderInputAttachmentArrayNonUniformIndexingNative, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(robustBufferAccessUpdateAfterBind, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(quadDivergentImplicitLod, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUpdateAfterBindSamplers, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUpdateAfterBindUBOs, uint32_t, 15);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUpdateAfterBindSSBOs, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUpdateAfterBindImages, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUpdateAfterBindStorageImages, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUpdateAfterBindInputAttachments, uint32_t, MinMaxColorAttachments);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageUpdateAfterBindResources, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindSamplers, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindUBOs, uint32_t, 72);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs, uint32_t, 8);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindSSBOs, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs, uint32_t, 4);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindImages, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindStorageImages, uint32_t, 500000);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindInputAttachments, uint32_t, MinMaxColorAttachments);

NBL_GENERATE_GET_OR_DEFAULT(supportedDepthResolveModesBitPattern, uint64_t, nbl::hlsl::ResolveModeFlags::SAMPLE_ZERO_BIT);
NBL_GENERATE_GET_OR_DEFAULT(supportedStencilResolveModesBitPattern, uint64_t, nbl::hlsl::ResolveModeFlags::SAMPLE_ZERO_BIT);
NBL_GENERATE_GET_OR_DEFAULT(independentResolveNone, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(independentResolve, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(filterMinmaxImageComponentMapping, bool, false);

// VK 1.3
NBL_GENERATE_GET_OR_DEFAULT(minSubgroupSize, uint16_t, 64);
NBL_GENERATE_GET_OR_DEFAULT(maxSubgroupSize, uint16_t, 4);
NBL_GENERATE_GET_OR_DEFAULT(maxComputeWorkgroupSubgroups, uint32_t, 16);
NBL_GENERATE_GET_OR_DEFAULT(requiredSubgroupSizeStagesBitPattern, uint64_t, nbl::hlsl::ShaderStage::ESS_UNKNOWN);

NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct8BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct8BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct8BitMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct4x8BitPackedUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct4x8BitPackedSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct4x8BitPackedMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct16BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct16BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct16BitMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct32BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct32BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct32BitMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct64BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct64BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProduct64BitMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating8BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating8BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating16BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating16BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating32BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating32BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating64BitUnsignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating64BitSignedAccelerated, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(storageTexelBufferOffsetAlignmentBytes, uint64_t, nbl::hlsl::numeric_limits<uint64_t>::max);
NBL_GENERATE_GET_OR_DEFAULT(uniformTexelBufferOffsetAlignmentBytes, uint64_t, nbl::hlsl::numeric_limits<uint64_t>::max);

NBL_GENERATE_GET_OR_DEFAULT(maxBufferSize, uint64_t, MinMaxSSBOSize);

// Nabla Core Extensions
NBL_GENERATE_GET_OR_DEFAULT(minImportedHostPointerAlignment, uint32_t, 0x1u << 31);

NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat32AtomicAdd, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat64Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat64AtomicAdd, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat32AtomicAdd, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat64Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat64AtomicAdd, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderImageFloat32AtomicAdd, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(sparseImageFloat32Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(sparseImageFloat32AtomicAdd, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(robustStorageBufferAccessSizeAlignment, uint64_t, 0x1ull << 63);
NBL_GENERATE_GET_OR_DEFAULT(robustUniformBufferAccessSizeAlignment, uint64_t, 0x1ull << 63);

// Extensions
NBL_GENERATE_GET_OR_DEFAULT(shaderTrinaryMinmax, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderExplicitVertexParameter, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(gpuShaderHalfFloatAMD, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderImageLoadStoreLod, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(displayTiming, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(maxDiscardRectangles, uint32_t, 0);

NBL_GENERATE_GET_OR_DEFAULT(primitiveOverestimationSizeBitPattern, uint32_t, asuint(0.f));
NBL_GENERATE_GET_OR_DEFAULT(maxExtraPrimitiveOverestimationSizeBitPattern, uint32_t, asuint(0.f));
NBL_GENERATE_GET_OR_DEFAULT(extraPrimitiveOverestimationSizeGranularityBitPattern, uint32_t, asuint(nbl::hlsl::numeric_limits<float>::infinity));
NBL_GENERATE_GET_OR_DEFAULT(primitiveUnderestimation, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(conservativePointAndLineRasterization, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(degenerateTrianglesRasterized, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(degenerateLinesRasterized, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(fullyCoveredFragmentShaderInputVariable, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(conservativeRasterizationPostDepthCoverage, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(queueFamilyForeign, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderStencilExport, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(variableSampleLocations, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(sampleLocationSubPixelBits, uint16_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(sampleLocationSampleCountsBitPattern, uint64_t, (nbl::hlsl::SampleCountFlags)(0u));
NBL_GENERATE_GET_OR_DEFAULT(maxSampleLocationGridSizeX, uint32_t, 0u);
NBL_GENERATE_GET_OR_DEFAULT(maxSampleLocationGridSizeY, uint32_t, 0u);
NBL_GENERATE_GET_OR_DEFAULT(sampleLocationCoordinateRangeBitPatternMin, uint32_t, asuint(1.f));
NBL_GENERATE_GET_OR_DEFAULT(sampleLocationCoordinateRangeBitPatternMax, uint32_t, asuint(0.f));

NBL_GENERATE_GET_OR_DEFAULT(maxAccelerationStructureGeometryCount, uint64_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxAccelerationStructureInstanceCount, uint64_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxAccelerationStructurePrimitiveCount, uint64_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorAccelerationStructures, uint64_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxPerStageDescriptorUpdateAfterBindAccelerationStructures, uint64_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetAccelerationStructures, uint64_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetUpdateAfterBindAccelerationStructures, uint64_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(minAccelerationStructureScratchOffsetAlignment, uint64_t, 0x1u << 31u);

NBL_GENERATE_GET_OR_DEFAULT(maxRayRecursionDepth, uint32_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxShaderGroupStride, uint32_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(shaderGroupBaseAlignment, uint32_t, 0x1u << 31u);
NBL_GENERATE_GET_OR_DEFAULT(maxRayDispatchInvocationCount, uint32_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(shaderGroupHandleAlignment, uint32_t, 0x1u << 31u);
NBL_GENERATE_GET_OR_DEFAULT(maxRayHitAttributeSize, uint32_t, 0);

NBL_GENERATE_GET_OR_DEFAULT(shaderSMBuiltins, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(postDepthCoverage, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(computeDerivativeGroupQuads, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(computeDerivativeGroupLinear, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(imageFootprint, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(pciDomain, uint32_t, ~0u);
NBL_GENERATE_GET_OR_DEFAULT(pciBus, uint32_t, ~0u);
NBL_GENERATE_GET_OR_DEFAULT(pciDevice, uint32_t, ~0u);
NBL_GENERATE_GET_OR_DEFAULT(pciFunction, uint32_t, ~0u);

NBL_GENERATE_GET_OR_DEFAULT(minFragmentDensityTexelSizeX, uint32_t, ~0u);
NBL_GENERATE_GET_OR_DEFAULT(minFragmentDensityTexelSizeY, uint32_t, ~0u);
NBL_GENERATE_GET_OR_DEFAULT(maxFragmentDensityTexelSizeX, uint32_t, 0u);
NBL_GENERATE_GET_OR_DEFAULT(maxFragmentDensityTexelSizeY, uint32_t, 0u);
NBL_GENERATE_GET_OR_DEFAULT(fragmentDensityInvocations, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(decorateString, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderImageInt64Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(sparseImageInt64Atomics, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(lineSubPixelPrecisionBits, uint32_t, 0);

NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat16Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat16AtomicAdd, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat16AtomicMinMax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat32AtomicMinMax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderBufferFloat64AtomicMinMax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat16Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat16AtomicAdd, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat16AtomicMinMax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat32AtomicMinMax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedFloat64AtomicMinMax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderImageFloat32AtomicMinMax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(sparseImageFloat32AtomicMinMax, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(deviceMemoryReport, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderNonSemanticInfo, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(fragmentShaderBarycentric, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderSubgroupUniformControlFlow, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(subsampledLoads, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(subsampledCoarseReconstructionEarlyAccess, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(maxSubsampledArrayLayers, uint32_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxDescriptorSetSubsampledSamplers, uint32_t, 0);

NBL_GENERATE_GET_OR_DEFAULT(workgroupMemoryExplicitLayout, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(workgroupMemoryExplicitLayoutScalarBlockLayout, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(workgroupMemoryExplicitLayout8BitAccess, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(workgroupMemoryExplicitLayout16BitAccess, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(colorWriteEnable, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(cooperativeMatrixSupportedStagesBitPattern, uint64_t, nbl::hlsl::ShaderStage::ESS_UNKNOWN);

// Nabla
NBL_GENERATE_GET_OR_DEFAULT(computeUnits, uint32_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(dispatchBase, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(allowCommandBufferQueryCopies, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(maxOptimallyResidentWorkgroupInvocations, uint32_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(maxResidentInvocations, uint32_t, 0);
NBL_GENERATE_GET_OR_DEFAULT(spirvVersionBitPattern, uint64_t, nbl::hlsl::SpirvVersion::ESV_1_6);

NBL_GENERATE_GET_OR_DEFAULT(logicOp, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(vertexPipelineStoresAndAtomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(fragmentStoresAndAtomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderTessellationAndGeometryPointSize, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderStorageImageMultisample, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderStorageImageReadWithoutFormat, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderStorageImageArrayDynamicIndexing, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderFloat64, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(variableMultisampleRate, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(storagePushConstant16, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(storageInputOutput16, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(multiviewGeometryShader, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(multiviewTessellationShader, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(drawIndirectCount, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(storagePushConstant8, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderBufferInt64Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSharedInt64Atomics, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderFloat16, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderInputAttachmentArrayDynamicIndexing, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderUniformBufferArrayNonUniformIndexing, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderInputAttachmentArrayNonUniformIndexing, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(descriptorBindingUniformBufferUpdateAfterBind, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(samplerFilterMinmax, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(vulkanMemoryModelAvailabilityVisibilityChains, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderOutputViewportIndex, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderOutputLayer, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDemoteToHelperInvocation, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderTerminateInvocation, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderZeroInitializeWorkgroupMemory, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderDeviceClock, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderSubgroupClock, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(imageFootPrint, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderIntegerFunctions2, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderEarlyAndLateFragmentTests, bool, false);

// Features Defaults
// VK 1.0
NBL_GENERATE_GET_OR_DEFAULT(robustBufferAccess, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(geometryShader, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(tessellationShader, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(depthBounds, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(wideLines, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(largePoints, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(alphaToOne, bool, true);

NBL_GENERATE_GET_OR_DEFAULT(pipelineStatisticsQuery, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderCullDistance, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(shaderResourceResidency, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(shaderResourceMinLod, bool, false);

// VK 1.1
// VK 1.2
NBL_GENERATE_GET_OR_DEFAULT(bufferDeviceAddressMultiDevice, bool, false);

// VK 1.3
NBL_GENERATE_GET_OR_DEFAULT(robustImageAccess, bool, false);

// Nabla Core Extensions
NBL_GENERATE_GET_OR_DEFAULT(robustBufferAccess2, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(robustImageAccess2, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(nullDescriptor, bool, false);

// Extensions
NBL_GENERATE_GET_OR_DEFAULT(swapchainModeBitPattern, uint64_t, nbl::hlsl::SwapchainMode::ESM_NONE);

NBL_GENERATE_GET_OR_DEFAULT(shaderInfoAMD, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(conditionalRendering, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(inheritedConditionalRendering, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(geometryShaderPassthrough, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(hdrMetadata, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(performanceCounterQueryPools, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(performanceCounterMultipleQueryPools, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(mixedAttachmentSamples, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(accelerationStructure, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(accelerationStructureIndirectBuild, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(accelerationStructureHostCommands, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(rayTracingPipeline, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(rayTraversalPrimitiveCulling, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(rayQuery, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(representativeFragmentTest, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(bufferMarkerAMD, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(fragmentDensityMap, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(fragmentDensityMapDynamic, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(fragmentDensityMapNonSubsampledImages, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(deviceCoherentMemory, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(memoryPriority, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(fragmentShaderSampleInterlock, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(fragmentShaderPixelInterlock, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(fragmentShaderShadingRateInterlock, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(rectangularLines, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(bresenhamLines, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(smoothLines, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(stippledRectangularLines, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(stippledBresenhamLines, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(stippledSmoothLines, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(indexTypeUint8, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(deferredHostOperations, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(pipelineExecutableInfo, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(deviceGeneratedCommands, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(rayTracingMotionBlur, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(rayTracingMotionBlurPipelineTraceRaysIndirect, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(fragmentDensityMapDeferred, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(rasterizationOrderColorAttachmentAccess, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(rasterizationOrderDepthAttachmentAccess, bool, false);
NBL_GENERATE_GET_OR_DEFAULT(rasterizationOrderStencilAttachmentAccess, bool, false);

NBL_GENERATE_GET_OR_DEFAULT(cooperativeMatrixRobustBufferAccess, bool, false);

// Nabla
