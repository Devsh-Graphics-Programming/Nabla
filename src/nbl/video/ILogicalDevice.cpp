#include "nbl/video/IPhysicalDevice.h"

using namespace nbl;
using namespace nbl::video;


ILogicalDevice::ILogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, const IPhysicalDevice* const physicalDevice, const SCreationParams& params, const bool runningInRenderdoc)
    : m_api(api), m_physicalDevice(physicalDevice), m_enabledFeatures(params.featuresToEnable), m_compilerSet(params.compilerSet),
    m_logger(m_physicalDevice->getDebugCallback() ? m_physicalDevice->getDebugCallback()->getLogger():nullptr)
{
    uint32_t qcnt = 0u;
    uint32_t greatestFamNum = 0u;
    for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
    {
        greatestFamNum = core::max(greatestFamNum,params.queueParams[i].familyIndex);
        qcnt += params.queueParams[i].count;
    }

    m_queues = core::make_refctd_dynamic_array<queues_array_t>(qcnt);
    m_queueFamilyInfos = core::make_refctd_dynamic_array<q_family_info_array_t>(greatestFamNum+1u);

    for (uint32_t i=0; i<params.queueParamsCount; i++)
    {
        const auto& qci = params.queueParams[i];
        auto& info = const_cast<QueueFamilyInfo&>(m_queueFamilyInfos->operator[](qci.familyIndex));
        {
            using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
            info.supportedStages = stage_flags_t::HOST_BIT;

            const auto transferStages = stage_flags_t::COPY_BIT|stage_flags_t::CLEAR_BIT|(m_enabledFeatures.accelerationStructure ? stage_flags_t::ACCELERATION_STRUCTURE_COPY_BIT:stage_flags_t::NONE)|stage_flags_t::RESOLVE_BIT|stage_flags_t::BLIT_BIT;
            const core::bitflag<stage_flags_t> computeAndGraphicsStages = (m_enabledFeatures.deviceGeneratedCommands ? stage_flags_t::COMMAND_PREPROCESS_BIT:stage_flags_t::NONE)|
                (m_enabledFeatures.conditionalRendering ? stage_flags_t::CONDITIONAL_RENDERING_BIT:stage_flags_t::NONE)|transferStages|stage_flags_t::DISPATCH_INDIRECT_COMMAND_BIT;

            const auto familyFlags = m_physicalDevice->getQueueFamilyProperties()[qci.familyIndex].queueFlags;
            if (familyFlags.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT))
            {
                info.supportedStages |= computeAndGraphicsStages|stage_flags_t::COMPUTE_SHADER_BIT;
                if (m_enabledFeatures.accelerationStructure)
                    info.supportedStages |= stage_flags_t::ACCELERATION_STRUCTURE_COPY_BIT|stage_flags_t::ACCELERATION_STRUCTURE_BUILD_BIT;
                if (m_enabledFeatures.rayTracingPipeline)
                    info.supportedStages |= stage_flags_t::RAY_TRACING_SHADER_BIT;
            }
            if (familyFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT))
            {
                info.supportedStages |= computeAndGraphicsStages|stage_flags_t::VERTEX_INPUT_BITS|stage_flags_t::VERTEX_SHADER_BIT;

                if (m_enabledFeatures.tessellationShader)
                    info.supportedStages |= stage_flags_t::TESSELLATION_CONTROL_SHADER_BIT|stage_flags_t::TESSELLATION_EVALUATION_SHADER_BIT;
                if (m_enabledFeatures.geometryShader)
                    info.supportedStages |= stage_flags_t::GEOMETRY_SHADER_BIT;
                // we don't do transform feedback
                //if (m_enabledFeatures.meshShader)
                //    info.supportedStages |= stage_flags_t::;
                //if (m_enabledFeatures.taskShader)
                //    info.supportedStages |= stage_flags_t::;
                if (m_enabledFeatures.fragmentDensityMap)
                    info.supportedStages |= stage_flags_t::FRAGMENT_DENSITY_PROCESS_BIT;
                //if (m_enabledFeatures.????)
                //    info.supportedStages |= stage_flags_t::SHADING_RATE_ATTACHMENT_BIT;

                info.supportedStages |= stage_flags_t::FRAMEBUFFER_SPACE_BITS;
            }
            if (familyFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT))
                info.supportedStages |= transferStages;
        }
        {
            using access_flags_t = asset::ACCESS_FLAGS;
            info.supportedAccesses = access_flags_t::HOST_READ_BIT|access_flags_t::HOST_WRITE_BIT;
        }
        info.firstQueueIndex = qci.count;
    }
    // bothering with an `std::exclusive_scan` is a bit too cumbersome here
    uint32_t sum = 0u;
    for (auto i=0u; i<m_queueFamilyInfos->size(); i++)
    {
        auto& x = m_queueFamilyInfos->operator[](i).firstQueueIndex;
        auto tmp = sum+x;
        const_cast<uint32_t&>(x) = sum;
        sum = tmp;
    }

    if (auto hlslCompiler = m_compilerSet ? m_compilerSet->getShaderCompiler(asset::IShader::E_CONTENT_TYPE::ECT_HLSL):nullptr)
        hlslCompiler->getDefaultIncludeFinder()->addSearchPath("nbl/builtin/hlsl/jit",core::make_smart_refctd_ptr<CJITIncludeLoader>(m_physicalDevice->getLimits(),m_physicalDevice->getFeatures()));
}

E_API_TYPE ILogicalDevice::getAPIType() const { return m_physicalDevice->getAPIType(); }

const SPhysicalDeviceLimits& ILogicalDevice::getPhysicalDeviceLimits() const
{
    return m_physicalDevice->getLimits();
}

bool ILogicalDevice::supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::PIPELINE_STAGE_FLAGS> stageMask) const
{
    if (queueFamilyIndex>m_queueFamilyInfos->size())
        return false;
    using q_family_flags_t = IQueue::FAMILY_FLAGS;
    const auto& familyProps = m_physicalDevice->getQueueFamilyProperties()[queueFamilyIndex].queueFlags;
    // strip special values
    if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS))
        return true;
    if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS) && bool(familyProps&(q_family_flags_t::COMPUTE_BIT|q_family_flags_t::GRAPHICS_BIT|q_family_flags_t::TRANSFER_BIT)))
        stageMask ^= asset::PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
    if (familyProps.hasFlags(q_family_flags_t::GRAPHICS_BIT))
    {
        if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS))
            stageMask ^= asset::PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS;
        if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::PRE_RASTERIZATION_SHADERS_BITS))
            stageMask ^= asset::PIPELINE_STAGE_FLAGS::PRE_RASTERIZATION_SHADERS_BITS;
    }
    return getSupportedStageMask(queueFamilyIndex).hasFlags(stageMask);
}

bool ILogicalDevice::supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::ACCESS_FLAGS> stageMask) const
{
    if (queueFamilyIndex>m_queueFamilyInfos->size())
        return false;
    using q_family_flags_t = IQueue::FAMILY_FLAGS;
    const auto& familyProps = m_physicalDevice->getQueueFamilyProperties()[queueFamilyIndex].queueFlags;
    const bool shaderCapableFamily = bool(familyProps&(q_family_flags_t::COMPUTE_BIT|q_family_flags_t::GRAPHICS_BIT));
    // strip special values
    if (stageMask.hasFlags(asset::ACCESS_FLAGS::MEMORY_READ_BITS))
        stageMask ^= asset::ACCESS_FLAGS::MEMORY_READ_BITS;
    else if (stageMask.hasFlags(asset::ACCESS_FLAGS::SHADER_READ_BITS) && shaderCapableFamily)
        stageMask ^= asset::ACCESS_FLAGS::SHADER_READ_BITS;
    if (stageMask.hasFlags(asset::ACCESS_FLAGS::MEMORY_WRITE_BITS))
        stageMask ^= asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
    else if (stageMask.hasFlags(asset::ACCESS_FLAGS::SHADER_WRITE_BITS) && shaderCapableFamily)
        stageMask ^= asset::ACCESS_FLAGS::SHADER_WRITE_BITS;
    return getSupportedAccessMask(queueFamilyIndex).hasFlags(stageMask);
}

bool ILogicalDevice::validateMemoryBarrier(const uint32_t queueFamilyIndex, asset::SMemoryBarrier barrier) const
{
    if (!supportsMask(queueFamilyIndex,barrier.srcStageMask) || !supportsMask(queueFamilyIndex,barrier.dstStageMask))
        return false;
    if (!supportsMask(queueFamilyIndex,barrier.srcAccessMask) || !supportsMask(queueFamilyIndex,barrier.dstAccessMask))
        return false;

    using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
    const core::bitflag<stage_flags_t> supportedStageMask = getSupportedStageMask(queueFamilyIndex);
    using access_flags_t = asset::ACCESS_FLAGS;
    const core::bitflag<access_flags_t> supportedAccessMask = getSupportedAccessMask(queueFamilyIndex);
    auto validAccess = [supportedStageMask,supportedAccessMask](core::bitflag<stage_flags_t>& stageMask, core::bitflag<access_flags_t>& accessMask) -> bool
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03916
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03917
        if (bool(accessMask&(access_flags_t::HOST_READ_BIT|access_flags_t::HOST_WRITE_BIT)) && !stageMask.hasFlags(stage_flags_t::HOST_BIT))
            return false;
        // this takes care of all stuff below
        if (stageMask.hasFlags(stage_flags_t::ALL_COMMANDS_BITS))
            return true;
        // first strip unsupported bits
        stageMask &= supportedStageMask;
        accessMask &= supportedAccessMask;
        // TODO: finish this stuff
        if (stageMask.hasFlags(stage_flags_t::ALL_GRAPHICS_BITS))
        {
            if (stageMask.hasFlags(stage_flags_t::ALL_TRANSFER_BITS))
            {
            }
            else
            {
            }
        }
        else
        {
            if (stageMask.hasFlags(stage_flags_t::ALL_TRANSFER_BITS))
            {
            }
            else
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03914
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03915
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03927
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03928
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-06256
            }
            // this is basic valid usage stuff
            #ifdef _NBL_DEBUG
            // TODO:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03900
            if (accessMask.hasFlags(access_flags_t::INDIRECT_COMMAND_READ_BIT) && !bool(stageMask&(stage_flags_t::DISPATCH_INDIRECT_COMMAND_BIT|stage_flags_t::ACCELERATION_STRUCTURE_BUILD_BIT)))
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03901
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03902
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03903
            //constexpr core::bitflag<stage_flags_t> ShaderStages = stage_flags_t::PRE_RASTERIZATION_SHADERS;
            //const bool noShaderStages = stageMask&ShaderStages;
            // TODO:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03904
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03905
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03906
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03907
            // IMPLICIT: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-07454
            // IMPLICIT: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03909
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-07272
            // TODO:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03910
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03911
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03912
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03913
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03918
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03919
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03924
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03925
            #endif
        }
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-07457
        return true;
    };

    return true;
}

core::smart_refctd_ptr<IGPUBufferView> ILogicalDevice::createBufferView(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt)
{
    if (!underlying.isValid() || !underlying.buffer->wasCreatedBy(this))
        return nullptr;
    if (!getPhysicalDevice()->getBufferFormatUsages()[_fmt].bufferView)
        return nullptr;
    return createBufferView_impl(underlying,_fmt);
}

core::smart_refctd_ptr<IGPUShader> ILogicalDevice::createShader(const asset::ICPUShader* cpushader, const asset::ISPIRVOptimizer* optimizer)
{
    if (!cpushader)
        return nullptr;

    const char* entryPoint = "main"; // every compiler seems to be handicapped this way?
    const asset::IShader::E_SHADER_STAGE shaderStage = cpushader->getStage();

    core::smart_refctd_ptr<const asset::ICPUShader> spirvShader;
    if (cpushader->getContentType()==asset::ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
        spirvShader = core::smart_refctd_ptr<const asset::ICPUShader>(cpushader);
    else
    {
        auto compiler = m_compilerSet->getShaderCompiler(cpushader->getContentType());

        asset::IShaderCompiler::SCompilerOptions commonCompileOptions = {};

        commonCompileOptions.preprocessorOptions.logger = m_physicalDevice->getDebugCallback() ? m_physicalDevice->getDebugCallback()->getLogger():nullptr;
        commonCompileOptions.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder(); // to resolve includes before compilation
        commonCompileOptions.preprocessorOptions.sourceIdentifier = cpushader->getFilepathHint().c_str();
        commonCompileOptions.preprocessorOptions.extraDefines = {};

        commonCompileOptions.stage = shaderStage;
        commonCompileOptions.debugInfoFlags =
            asset::IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT |
            asset::IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT;
        commonCompileOptions.spirvOptimizer = optimizer;
        commonCompileOptions.targetSpirvVersion = m_physicalDevice->getLimits().spirvVersion;

        if (cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_HLSL)
        {
            // TODO: add specific HLSLCompiler::SOption params
            spirvShader = m_compilerSet->compileToSPIRV(cpushader,commonCompileOptions);
        }
        else if (cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
        {
            spirvShader = m_compilerSet->compileToSPIRV(cpushader,commonCompileOptions);
        }
        else
            spirvShader = m_compilerSet->compileToSPIRV(cpushader,commonCompileOptions);

        if (!spirvShader)
            return nullptr;
    }

    auto spirv = spirvShader->getContent();
    if (!spirv)
        return nullptr;

    // for debugging 
    if constexpr (true)
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        m_physicalDevice->getSystem()->createFile(future,system::path(cpushader->getFilepathHint()).parent_path()/"compiled.spv",system::IFileBase::ECF_WRITE);
        if (auto file=future.acquire(); file&&bool(*file))
        {
            system::IFile::success_t succ;
            (*file)->write(succ,spirv->getPointer(),0,spirv->getSize());
            succ.getBytesProcessed(true);
        }
    }

    return createShader_impl(spirvShader.get());
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> ILogicalDevice::createDescriptorSetLayout(const core::SRange<const IGPUDescriptorSetLayout::SBinding>& bindings)
{
    // TODO: MORE VALIDATION, but after descriptor indexing.
    uint32_t maxSamplersCount = 0u;
    uint32_t dynamicSSBOCount=0u,dynamicUBOCount=0u;
    for (auto& binding : bindings)
    {
        if (binding.type==asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)
            dynamicSSBOCount++;
        else if (binding.type==asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)
            dynamicUBOCount++;
        else if (binding.type==asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER && binding.samplers)
        {
            auto* samplers = binding.samplers;
            for (uint32_t i=0u; i<binding.count; ++i)
            if (!samplers[i]->wasCreatedBy(this))
                return nullptr;
            maxSamplersCount += binding.count;
        }
    }

    const auto& limits = m_physicalDevice->getLimits();
    if (dynamicSSBOCount>limits.maxDescriptorSetDynamicOffsetSSBOs || dynamicUBOCount>limits.maxDescriptorSetDynamicOffsetUBOs)
        return nullptr;

    return createDescriptorSetLayout_impl(bindings,maxSamplersCount);
}


bool ILogicalDevice::updateDescriptorSets(const std::span<const IGPUDescriptorSet::SWriteDescriptorSet>& descriptorWrites, const std::span<const IGPUDescriptorSet::SCopyDescriptorSet>& descriptorCopies)
{
    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        const auto& write = pDescriptorWrites[i];
        auto* ds = static_cast<IGPUDescriptorSet*>(write.dstSet);

        assert(ds->getLayout()->isCompatibleDevicewise(ds));

        if (!ds->validateWrite(write))
            return false;
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto& copy = pDescriptorCopies[i];
        const auto* srcDS = static_cast<const IGPUDescriptorSet*>(copy.srcSet);
        const auto* dstDS = static_cast<IGPUDescriptorSet*>(copy.dstSet);

        if (!dstDS->isCompatibleDevicewise(srcDS))
            return false;

        if (!dstDS->validateCopy(copy))
            return false;
    }

    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        auto& write = pDescriptorWrites[i];
        auto* ds = static_cast<IGPUDescriptorSet*>(write.dstSet);
        ds->processWrite(write);
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto& copy = pDescriptorCopies[i];
        auto* dstDS = static_cast<IGPUDescriptorSet*>(pDescriptorCopies[i].dstSet);
        dstDS->processCopy(copy);
    }

    updateDescriptorSets_impl(descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);

    return true;
}

core::smart_refctd_ptr<IGPURenderpass> ILogicalDevice::createRenderpass(const IGPURenderpass::SCreationParams& params)
{
    IGPURenderpass::SCreationParamValidationResult validation = IGPURenderpass::validateCreationParams(params);
    if (!validation)
        return nullptr;
            
    const auto& optimalTilingUsages = getPhysicalDevice()->getImageFormatUsagesOptimalTiling();
    auto invalidAttachment = [this,&optimalTilingUsages]<typename Layout, template<typename> class op_t>(const IGPURenderpass::SCreationParams::SAttachmentDescription<Layout,op_t>& desc) -> bool
    {
        // We won't support linear attachments, so implicitly satisfy
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-linearColorAttachment-06499
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-linearColorAttachment-06500
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-linearColorAttachment-06501
        const auto& usages = optimalTilingUsages[desc.format];
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAttachmentDescription2-samples-08745
        //if (!usages.sampleCounts.hasFlags(desc.samples))
            //return true;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pInputAttachments-02897
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pColorAttachments-02898
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pResolveAttachments-09343
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-pDepthStencilAttachment-02900
        if (!usages.attachment)
            return true;
        return false;
    };
    for (uint32_t i=0u; i<validation.depthStencilAttachmentCount; i++)
    if (invalidAttachment(params.depthStencilAttachments[i]))
        return nullptr;
    for (uint32_t i=0u; i<validation.colorAttachmentCount; i++)
    if (invalidAttachment(params.colorAttachments[i]))
        return nullptr;

    const auto mixedAttachmentSamples = getEnabledFeatures().mixedAttachmentSamples;
    const auto supportedDepthResolveModes = getPhysicalDeviceLimits().supportedDepthResolveModes;
    const auto supportedStencilResolveModes = getPhysicalDeviceLimits().supportedStencilResolveModes;
    const auto independentResolve = getPhysicalDeviceLimits().independentResolve;
    const auto independentResolveNone = getPhysicalDeviceLimits().independentResolveNone;
    const auto maxColorAttachments = getPhysicalDeviceLimits().maxColorAttachments;
    const int32_t maxMultiviewViewCount = getPhysicalDeviceLimits().maxMultiviewViewCount;
    for (auto i=0u; i<validation.subpassCount; i++)
    {
        using subpass_desc_t = IGPURenderpass::SCreationParams::SSubpassDescription;
        const subpass_desc_t& subpass = params.subpasses[i];

        auto depthSamples = static_cast<IGPUImage::E_SAMPLE_COUNT_FLAGS>(128);
        if (subpass.depthStencilAttachment.render.used())
        {
            depthSamples = params.depthStencilAttachments[subpass.depthStencilAttachment.render.attachmentIndex].samples;

            using resolve_flag_t = IGPURenderpass::SCreationParams::SSubpassDescription::SDepthStencilAttachmentsRef::RESOLVE_MODE;
            // TODO: seems like `multisampledRenderToSingleSampledEnable` needs resolve modes but not necessarily a resolve attachmen
            const resolve_flag_t depthResolve = subpass.depthStencilAttachment.resolveMode.depth;
            const resolve_flag_t stencilResolve = subpass.depthStencilAttachment.resolveMode.stencil;
            if (subpass.depthStencilAttachment.resolve.used() || /*multisampledToSingleSampledUsed*/false)
            {
                const auto& attachment = params.depthStencilAttachments[(subpass.depthStencilAttachment.resolve.used() ? subpass.depthStencilAttachment.resolve:subpass.depthStencilAttachment.render).attachmentIndex];

                const bool hasDepth = !asset::isStencilOnlyFormat(attachment.format);
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-depthResolveMode-03183
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06874
                if (hasDepth && !supportedDepthResolveModes.hasFlags(depthResolve))
                    return nullptr;
;
                const bool hasStencil = !asset::isDepthOnlyFormat(attachment.format);
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-stencilResolveMode-03184
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06875
                if (hasStencil && !supportedStencilResolveModes.hasFlags(stencilResolve))
                    return nullptr;

                if (hasDepth && hasStencil)
                {
                    if (!independentResolve && depthResolve!=stencilResolve)
                    {
                        if (independentResolveNone)
                        {
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03186
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06877
                            if (depthResolve!=resolve_flag_t::NONE && stencilResolve!=resolve_flag_t::NONE)
                                return nullptr;
                        }
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03185
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06876
                        else
                            return nullptr;
                    }
                }

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06873
                if (/*multisampledToSingleSampledUsed*/false && depthResolve==resolve_flag_t::NONE && stencilResolve==resolve_flag_t::NONE)
                    return nullptr;
            }
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-colorAttachmentCount-03063
        for (auto j=maxColorAttachments; j<subpass_desc_t::MaxColorAttachments; j++)
        if (subpass.colorAttachments[j].render.used())
            return nullptr;
        // TODO: support `VK_EXT_multisampled_render_to_single_sampled`
        auto samplesForAllColor = (depthSamples>IGPUImage::ESCF_64_BIT||mixedAttachmentSamples/*||multisampledRenderToSingleSampled*/) ? static_cast<IGPUImage::E_SAMPLE_COUNT_FLAGS>(0):depthSamples;
        for (auto j=0u; j<maxColorAttachments; j++)
        {
            const auto& ref = subpass.colorAttachments[j].render;
            if (!ref.used())
                continue;

            const auto samples = params.colorAttachments[ref.attachmentIndex].samples;
            // initialize if everything till now was unused
            if (!samplesForAllColor)
                samplesForAllColor = samples;

            if (mixedAttachmentSamples)
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-None-09456
                if (samples>depthSamples)
                    return nullptr;
            }
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-multisampledRenderToSingleSampled-06869
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-multisampledRenderToSingleSampled-06872
            else if (!false/*multisampledRenderToSingleSampled*/ && samples!=samplesForAllColor)
                return nullptr;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-viewMask-06706
        if (core::findMSB(subpass.viewMask)>=maxMultiviewViewCount)
            return nullptr;
    }

    for (auto i=0u; i<validation.dependencyCount; i++)
    {
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03054
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03055
    }

    return createRenderpass_impl(params,std::move(validation));
}