#include "nbl/video/IPhysicalDevice.h"

#include "git_info.h"
#define NBL_LOG_FUNCTION m_logger.log
#include "nbl/logging_macros.h"

using namespace nbl;
using namespace nbl::video;


ILogicalDevice::ILogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, const IPhysicalDevice* const physicalDevice, const SCreationParams& params, const bool runningInRenderdoc)
    : m_api(api), m_physicalDevice(physicalDevice), m_enabledFeatures(params.featuresToEnable), m_compilerSet(params.compilerSet),
    m_logger(m_physicalDevice->getDebugCallback() ? m_physicalDevice->getDebugCallback()->getLogger() : nullptr)
{
    {
        uint32_t qcnt = 0u;
        for (uint32_t i = 0; i < MaxQueueFamilies; i++)
            qcnt += params.queueParams[i].count;
        m_queues = core::make_refctd_dynamic_array<queues_array_t>(qcnt);
    }

    for (uint32_t i = 0; i < MaxQueueFamilies; i++)
    {
        const auto& qci = params.queueParams[i];
        auto& info = const_cast<QueueFamilyInfo&>(m_queueFamilyInfos[i]);
        if (qci.count)
        {
            using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
            using access_flags_t = asset::ACCESS_FLAGS;
            info.supportedStages = stage_flags_t::HOST_BIT;
            info.supportedAccesses = access_flags_t::HOST_READ_BIT | access_flags_t::HOST_WRITE_BIT;

            const auto transferStages = stage_flags_t::COPY_BIT | stage_flags_t::CLEAR_BIT | stage_flags_t::RESOLVE_BIT | stage_flags_t::BLIT_BIT;
            const auto transferAccesses = access_flags_t::TRANSFER_READ_BIT | access_flags_t::TRANSFER_WRITE_BIT;

            core::bitflag<stage_flags_t> computeAndGraphicsStages = transferStages | stage_flags_t::DISPATCH_INDIRECT_COMMAND_BIT;
            core::bitflag<access_flags_t> computeAndGraphicsAccesses = transferAccesses | access_flags_t::INDIRECT_COMMAND_READ_BIT | access_flags_t::UNIFORM_READ_BIT | access_flags_t::STORAGE_READ_BIT | access_flags_t::STORAGE_WRITE_BIT | access_flags_t::SAMPLED_READ_BIT;
            if (m_enabledFeatures.deviceGeneratedCommands)
            {
                computeAndGraphicsStages |= stage_flags_t::COMMAND_PREPROCESS_BIT;
                computeAndGraphicsAccesses |= access_flags_t::COMMAND_PREPROCESS_READ_BIT | access_flags_t::COMMAND_PREPROCESS_WRITE_BIT;
            }
            if (m_enabledFeatures.conditionalRendering)
            {
                computeAndGraphicsStages |= stage_flags_t::CONDITIONAL_RENDERING_BIT;
                computeAndGraphicsAccesses |= access_flags_t::CONDITIONAL_RENDERING_READ_BIT;
            }

            const auto familyFlags = m_physicalDevice->getQueueFamilyProperties()[i].queueFlags;
            if (familyFlags.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT))
            {
                info.supportedStages |= computeAndGraphicsStages | stage_flags_t::COMPUTE_SHADER_BIT;
                info.supportedAccesses |= computeAndGraphicsAccesses;
                if (m_enabledFeatures.accelerationStructure)
                {
                    info.supportedStages |= stage_flags_t::ACCELERATION_STRUCTURE_COPY_BIT | stage_flags_t::ACCELERATION_STRUCTURE_BUILD_BIT;
                    info.supportedAccesses |= access_flags_t::ACCELERATION_STRUCTURE_READ_BIT | access_flags_t::ACCELERATION_STRUCTURE_WRITE_BIT;
                }
                if (m_enabledFeatures.rayTracingPipeline)
                {
                    info.supportedStages |= stage_flags_t::RAY_TRACING_SHADER_BIT;
                    info.supportedAccesses |= access_flags_t::SHADER_BINDING_TABLE_READ_BIT;
                }
            }
            if (familyFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT))
            {
                info.supportedStages |= computeAndGraphicsStages | stage_flags_t::VERTEX_INPUT_BITS | stage_flags_t::VERTEX_SHADER_BIT;
                info.supportedAccesses |= computeAndGraphicsAccesses | access_flags_t::INDEX_READ_BIT | access_flags_t::VERTEX_ATTRIBUTE_READ_BIT;
                info.supportedAccesses |= access_flags_t::DEPTH_STENCIL_ATTACHMENT_READ_BIT | access_flags_t::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                info.supportedAccesses |= access_flags_t::INPUT_ATTACHMENT_READ_BIT | access_flags_t::COLOR_ATTACHMENT_READ_BIT | access_flags_t::COLOR_ATTACHMENT_WRITE_BIT;

                if (m_enabledFeatures.tessellationShader)
                    info.supportedStages |= stage_flags_t::TESSELLATION_CONTROL_SHADER_BIT | stage_flags_t::TESSELLATION_EVALUATION_SHADER_BIT;
                if (m_enabledFeatures.geometryShader)
                    info.supportedStages |= stage_flags_t::GEOMETRY_SHADER_BIT;
                // we don't do transform feedback
                //if (m_enabledFeatures.meshShader)
                //    info.supportedStages |= stage_flags_t::;
                //if (m_enabledFeatures.taskShader)
                //    info.supportedStages |= stage_flags_t::;
                if (m_enabledFeatures.fragmentDensityMap)
                {
                    info.supportedStages |= stage_flags_t::FRAGMENT_DENSITY_PROCESS_BIT;
                    info.supportedAccesses |= access_flags_t::FRAGMENT_DENSITY_MAP_READ_BIT;
                }
                //if (m_enabledFeatures.????)
                //    info.supportedStages |= stage_flags_t::SHADING_RATE_ATTACHMENT_BIT;
                //    info.supportedAccesses |= access_flags_t::SHADING_RATE_ATTACHMENT_READ_BIT;

                info.supportedStages |= stage_flags_t::FRAMEBUFFER_SPACE_BITS;
            }
            if (familyFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT))
            {
                info.supportedStages |= transferStages;
                info.supportedAccesses |= transferAccesses;
                if (m_enabledFeatures.accelerationStructure)
                {
                    info.supportedStages |= stage_flags_t::ACCELERATION_STRUCTURE_COPY_BIT;
                    info.supportedAccesses |= access_flags_t::ACCELERATION_STRUCTURE_READ_BIT | access_flags_t::ACCELERATION_STRUCTURE_WRITE_BIT;
                }
            }
        }
        info.queueCount = qci.count;
        if (i)
            info.firstQueueIndex = m_queueFamilyInfos[i - 1].firstQueueIndex + m_queueFamilyInfos[i - 1].queueCount;
        else
            info.firstQueueIndex = 0;
    }

    if (auto hlslCompiler = m_compilerSet ? m_compilerSet->getShaderCompiler(asset::IShader::E_CONTENT_TYPE::ECT_HLSL) : nullptr)
        hlslCompiler->getDefaultIncludeFinder()->addSearchPath("nbl/builtin/hlsl/jit", core::make_smart_refctd_ptr<CJITIncludeLoader>(m_physicalDevice->getLimits(), m_enabledFeatures));
}

E_API_TYPE ILogicalDevice::getAPIType() const { return m_physicalDevice->getAPIType(); }

const SPhysicalDeviceLimits& ILogicalDevice::getPhysicalDeviceLimits() const
{
    return m_physicalDevice->getLimits();
}

bool ILogicalDevice::supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::PIPELINE_STAGE_FLAGS> stageMask) const
{
    if (getQueueCount(queueFamilyIndex) == 0)
        return false;
    using q_family_flags_t = IQueue::FAMILY_FLAGS;
    const auto& familyProps = m_physicalDevice->getQueueFamilyProperties()[queueFamilyIndex].queueFlags;
    // strip special values
    if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS))
        return true;
    if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS) && bool(familyProps & (q_family_flags_t::COMPUTE_BIT | q_family_flags_t::GRAPHICS_BIT | q_family_flags_t::TRANSFER_BIT)))
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

bool ILogicalDevice::supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::ACCESS_FLAGS> accessMask) const
{
    if (getQueueCount(queueFamilyIndex) == 0)
        return false;
    using q_family_flags_t = IQueue::FAMILY_FLAGS;
    const auto& familyProps = m_physicalDevice->getQueueFamilyProperties()[queueFamilyIndex].queueFlags;
    const bool shaderCapableFamily = bool(familyProps & (q_family_flags_t::COMPUTE_BIT | q_family_flags_t::GRAPHICS_BIT));
    // strip special values
    if (accessMask.hasFlags(asset::ACCESS_FLAGS::MEMORY_READ_BITS))
        accessMask ^= asset::ACCESS_FLAGS::MEMORY_READ_BITS;
    else if (accessMask.hasFlags(asset::ACCESS_FLAGS::SHADER_READ_BITS) && shaderCapableFamily)
        accessMask ^= asset::ACCESS_FLAGS::SHADER_READ_BITS;
    if (accessMask.hasFlags(asset::ACCESS_FLAGS::MEMORY_WRITE_BITS))
        accessMask ^= asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
    else if (accessMask.hasFlags(asset::ACCESS_FLAGS::SHADER_WRITE_BITS) && shaderCapableFamily)
        accessMask ^= asset::ACCESS_FLAGS::SHADER_WRITE_BITS;
    return getSupportedAccessMask(queueFamilyIndex).hasFlags(accessMask);
}

bool ILogicalDevice::validateMemoryBarrier(const uint32_t queueFamilyIndex, asset::SMemoryBarrier barrier) const
{
    if (!supportsMask(queueFamilyIndex, barrier.srcStageMask) || !supportsMask(queueFamilyIndex, barrier.dstStageMask))
    {
        NBL_LOG_ERROR("Invalid stage mask");
        return false;
    }
    if (!supportsMask(queueFamilyIndex, barrier.srcAccessMask) || !supportsMask(queueFamilyIndex, barrier.dstAccessMask))
    {
        NBL_LOG_ERROR("Invalid access mask");
        return false;
    }

    using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
    const core::bitflag<stage_flags_t> supportedStageMask = getSupportedStageMask(queueFamilyIndex);
    using access_flags_t = asset::ACCESS_FLAGS;
    const core::bitflag<access_flags_t> supportedAccessMask = getSupportedAccessMask(queueFamilyIndex);
    auto validAccess = [supportedStageMask, supportedAccessMask](core::bitflag<stage_flags_t>& stageMask, core::bitflag<access_flags_t>& accessMask) -> bool
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03916
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03917
            if (bool(accessMask & (access_flags_t::HOST_READ_BIT | access_flags_t::HOST_WRITE_BIT)) && !stageMask.hasFlags(stage_flags_t::HOST_BIT))
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
                if (accessMask.hasFlags(access_flags_t::INDIRECT_COMMAND_READ_BIT) && !bool(stageMask & (stage_flags_t::DISPATCH_INDIRECT_COMMAND_BIT | stage_flags_t::ACCELERATION_STRUCTURE_BUILD_BIT)))
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


IQueue::RESULT ILogicalDevice::waitIdle()
{
    const auto retval = waitIdle_impl();
    // Since this is equivalent to calling waitIdle on all queues, just proceed to releasing tracked resources
    for (auto queue : *m_queues)
        queue->cullResources();
    return retval;
}


core::smart_refctd_ptr<IGPUBufferView> ILogicalDevice::createBufferView(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt)
{
    if (!underlying.isValid() || !underlying.buffer->wasCreatedBy(this))
    {
        NBL_LOG_ERROR("Invalid buffer range");
        return nullptr;
    }
    if (!getPhysicalDevice()->getBufferFormatUsages()[_fmt].bufferView)
    {
        NBL_LOG_ERROR("Invalid buffer format");
        return nullptr;
    }
    return createBufferView_impl(underlying, _fmt);
}


core::smart_refctd_ptr<asset::ICPUShader> ILogicalDevice::compileShader(const SShaderCreationParameters& creationParams)
{
    if (!creationParams.cpushader)
    {
        NBL_LOG_ERROR("No valid CPU Shader supplied");
        return nullptr;
    }

    const asset::IShader::E_SHADER_STAGE shaderStage = creationParams.cpushader->getStage();
    const auto& features = getEnabledFeatures();

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-00704
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-00705
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-02091
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-02092
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-00706
    switch (shaderStage)
    {
    case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION:
        if (!features.tessellationShader)
        {
            NBL_LOG_ERROR("Cannot create IGPUShader for %p, Tessellation Shader feature not enabled!", creationParams.cpushader);
            return nullptr;
        }
        break;
    case IGPUShader::E_SHADER_STAGE::ESS_GEOMETRY:
        if (!features.geometryShader)
        {
            NBL_LOG_ERROR("Cannot create IGPUShader for %p, Geometry Shader feature not enabled!", creationParams.cpushader);
            return nullptr;
        }
        break;
    case IGPUShader::E_SHADER_STAGE::ESS_ALL_OR_LIBRARY: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_VERTEX: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_COMPUTE:
        break;
        // unsupported yet
    case IGPUShader::E_SHADER_STAGE::ESS_TASK: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_MESH:
        NBL_LOG_ERROR("Unsupported (yet) shader stage");
        return nullptr;
        break;
    case IGPUShader::E_SHADER_STAGE::ESS_RAYGEN: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_ANY_HIT: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_MISS: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_INTERSECTION: [[fallthrough]];
    case IGPUShader::E_SHADER_STAGE::ESS_CALLABLE:
        if (!features.rayTracingPipeline)
        {
            NBL_LOG_ERROR("Cannot create IGPUShader for %p, Raytracing Pipeline feature not enabled!", creationParams.cpushader);
            return nullptr;
        }
        break;
    default:
        // Implicit unsupported stages or weird multi-bit stage enum values
        NBL_LOG_ERROR("Unknown Shader Stage %d", shaderStage);
        return nullptr;
        break;
    }

    core::smart_refctd_ptr<asset::ICPUShader> spirvShader;
    if (creationParams.cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
    {
        if (creationParams.optimizer)
        {
            spirvShader = core::make_smart_refctd_ptr<asset::ICPUShader>(
                std::move(creationParams.optimizer->optimize(creationParams.cpushader->getContent(), m_logger)),
                shaderStage, asset::ICPUShader::E_CONTENT_TYPE::ECT_SPIRV,
                std::string(creationParams.cpushader->getFilepathHint()));
        }
        else
        {
            spirvShader = asset::IAsset::castDown<asset::ICPUShader>(creationParams.cpushader->clone());
        }
    }
    else
    {
        auto compiler = m_compilerSet->getShaderCompiler(creationParams.cpushader->getContentType());

        asset::IShaderCompiler::SCompilerOptions commonCompileOptions = {};

        commonCompileOptions.preprocessorOptions.logger = m_physicalDevice->getDebugCallback() ? m_physicalDevice->getDebugCallback()->getLogger() : nullptr;
        commonCompileOptions.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder(); // to resolve includes before compilation
        commonCompileOptions.preprocessorOptions.sourceIdentifier = creationParams.cpushader->getFilepathHint().c_str();
        commonCompileOptions.preprocessorOptions.extraDefines = {};

        commonCompileOptions.stage = shaderStage;
        commonCompileOptions.debugInfoFlags =
            asset::IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT |
            asset::IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT;
        commonCompileOptions.spirvOptimizer = creationParams.optimizer;
        commonCompileOptions.targetSpirvVersion = m_physicalDevice->getLimits().spirvVersion;

        commonCompileOptions.readCache = creationParams.readCache;
        commonCompileOptions.writeCache = creationParams.writeCache;

        if (creationParams.cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_HLSL)
        {
            // TODO: add specific HLSLCompiler::SOption params
            spirvShader = m_compilerSet->compileToSPIRV(creationParams.cpushader, commonCompileOptions);
        }
        else if (creationParams.cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
        {
            spirvShader = m_compilerSet->compileToSPIRV(creationParams.cpushader, commonCompileOptions);
        }
        else
        {
            NBL_LOG_ERROR("Unknown shader content type");
            return nullptr;
        }
    }

    if (!spirvShader)
    {
        NBL_LOG_ERROR("SPIR-V Compilation from non SPIR-V shader %p failed", creationParams.cpushader);
        return nullptr;
    }

    auto spirv = spirvShader->getContent();
    if (!spirv)
    {
        NBL_LOG_ERROR("SPIR-V Compilation from non SPIR-V shader %p failed", creationParams.cpushader);
        return nullptr;
    }

    // for debugging 
    if constexpr (true)
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        m_physicalDevice->getSystem()->createFile(future, system::path(creationParams.cpushader->getFilepathHint()).parent_path() / "compiled.spv", system::IFileBase::ECF_WRITE);
        if (auto file = future.acquire(); file && bool(*file))
        {
            system::IFile::success_t succ;
            (*file)->write(succ, spirv->getPointer(), 0, spirv->getSize());
            succ.getBytesProcessed(true);
        }
    }

    return spirvShader;
}

core::smart_refctd_ptr<IGPUShader> ILogicalDevice::createShader(const SShaderCreationParameters& creationParams)
{
    auto cpuShader = compileShader(creationParams);
    if (!cpuShader) 
        return nullptr;

    auto shader = createShader_impl(cpuShader.get());
    const auto path = creationParams.cpushader->getFilepathHint();
    if (shader && !path.empty())
        shader->setObjectDebugName(path.c_str());
    return shader;
}

core::smart_refctd_ptr<IGPUShader> ILogicalDevice::createShader(const asset::ICPUShader* cpushader, const asset::ISPIRVOptimizer* optimizer)
{
    return ILogicalDevice::createShader({ cpushader, optimizer, nullptr });
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> ILogicalDevice::createDescriptorSetLayout(const std::span<const IGPUDescriptorSetLayout::SBinding> bindings)
{
    // TODO: MORE VALIDATION, but after descriptor indexing.

    bool variableLengthArrayDescriptorFound = false;
    bool updateableAfterBindBindingFound = false;
    uint32_t variableLengthArrayDescriptorBindingNr = 0;
    uint32_t highestBindingNr = 0u;
    uint32_t maxSamplersCount = 0u;
    uint32_t dynamicSSBOCount = 0u, dynamicUBOCount = 0u;
    for (uint32_t i = 0u; i < bindings.size(); ++i)
    {
        const auto& binding = bindings[i];

        if (binding.type == asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)
            dynamicSSBOCount++;
        else if (binding.type == asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)
            dynamicUBOCount++;
        // If binding comes with samplers, we're specifying that this binding corresponds to immutable samplers
        else if ((binding.type == asset::IDescriptor::E_TYPE::ET_SAMPLER or binding.type == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER) and binding.immutableSamplers)
        {
            auto* samplers = binding.immutableSamplers;
            for (uint32_t ii = 0u; ii < binding.count; ++ii)
                if ((not samplers[ii]) or (not samplers[ii]->wasCreatedBy(this)))
                {
                    NBL_LOG_ERROR("Invalid sampler (bindings[%u].immutableSamplers[%u])", i, ii);
                    return nullptr;
                }
            maxSamplersCount += binding.count;
        }

        if (bindings[i].createFlags & IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT)
            updateableAfterBindBindingFound = true;

        // validate if only last binding is run-time sized and there is only one run-time sized binding
        bool isCurrentDescriptorVariableLengthArray = static_cast<bool>(bindings[i].createFlags & IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_VARIABLE_DESCRIPTOR_COUNT_BIT);
        // no 2 run-time sized descriptors allowed
        if (variableLengthArrayDescriptorFound && isCurrentDescriptorVariableLengthArray)
        {
            NBL_LOG_ERROR("Only one variable-sized binding is allowed (bindings[%u])", i);
            return nullptr;
        }

        if (isCurrentDescriptorVariableLengthArray)
        {
            variableLengthArrayDescriptorFound = true;
            variableLengthArrayDescriptorBindingNr = bindings[i].binding;
        }
        highestBindingNr = std::max(highestBindingNr, bindings[i].binding);
    }

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkDescriptorSetLayoutCreateInfo-descriptorType-03001
    if (updateableAfterBindBindingFound and dynamicSSBOCount + dynamicUBOCount != 0)
    {
        NBL_LOG_ERROR("UPDATE_AFTER_BIND bindings are mutually exclusive with DYNAMIC bindings");
        return nullptr;
    }

    // only last binding can be run-time sized
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03004
    if (variableLengthArrayDescriptorFound && variableLengthArrayDescriptorBindingNr != highestBindingNr)
    {
        NBL_LOG_ERROR("Only last binding can be variable-sized");
        return nullptr;
    }

    const auto& limits = m_physicalDevice->getLimits();
    if (dynamicSSBOCount > limits.maxDescriptorSetDynamicOffsetSSBOs || dynamicUBOCount > limits.maxDescriptorSetDynamicOffsetUBOs)
    {
        NBL_LOG_ERROR("Number of dynamic bindings exceeds device limits");
        return nullptr;
    }

    return createDescriptorSetLayout_impl(bindings, maxSamplersCount);
}


bool ILogicalDevice::updateDescriptorSets(const std::span<const IGPUDescriptorSet::SWriteDescriptorSet> descriptorWrites, const std::span<const IGPUDescriptorSet::SCopyDescriptorSet> descriptorCopies)
{
    using redirect_t = IGPUDescriptorSetLayout::CBindingRedirect;
    SUpdateDescriptorSetsParams params = { .writes = descriptorWrites,.copies = descriptorCopies };
    core::vector<asset::IDescriptor::E_TYPE> writeTypes(descriptorWrites.size());
    auto outCategory = writeTypes.data();
    params.pWriteTypes = outCategory;
    core::vector<IGPUDescriptorSet::SWriteValidationResult> writeValidationResults(descriptorWrites.size());
    for (auto i = 0u; i < descriptorWrites.size(); i++)
    {
        const auto& write = descriptorWrites[i];
        auto* ds = write.dstSet;
        if (!ds || !ds->wasCreatedBy(this))
        {
            NBL_LOG_ERROR("Invalid write descriptor set was given (descriptorWrites[%u])", i);
            return false;
        }

        const auto writeCount = write.count;
        writeValidationResults[i] = ds->validateWrite(write);
        switch (asset::IDescriptor::GetTypeCategory(*outCategory = writeValidationResults[i].type))
        {
        case asset::IDescriptor::EC_BUFFER:
            params.bufferCount += writeCount;
            break;
        case asset::IDescriptor::EC_SAMPLER:
        case asset::IDescriptor::EC_IMAGE:
            params.imageCount += writeCount;
            break;
        case asset::IDescriptor::EC_BUFFER_VIEW:
            params.bufferViewCount += writeCount;
            break;
        case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
            params.accelerationStructureCount += writeCount;
            params.accelerationStructureWriteCount++;
            break;
        default: // validation failed
            NBL_LOG_ERROR("Invalid descriptor type (descriptorWrites[%u])", i);
            return false;
        }
        outCategory++;
    }

    core::vector<IGPUDescriptorSet::SCopyValidationResult> copyValidationResults(descriptorCopies.size());
    for (auto i = 0u; i < descriptorCopies.size(); i++)
    {
        const auto& copy = descriptorCopies[i];
        const auto* srcDS = copy.srcSet;
        const auto* dstDS = static_cast<IGPUDescriptorSet*>(copy.dstSet);
        if (!dstDS || !dstDS->wasCreatedBy(this))
        {
            NBL_LOG_ERROR("Invalid copy descriptor set (descriptorCopies[%d])", i);
            return false;
        }
        if (!srcDS || !dstDS->isCompatibleDevicewise(srcDS))
        {
            NBL_LOG_ERROR("Invalid copy descriptor set (descriptorCopies[%d])", i);
            return false;
        }

        copyValidationResults[i] = dstDS->validateCopy(copy);
        if (asset::IDescriptor::E_TYPE::ET_COUNT == copyValidationResults[i].type)
        {
            NBL_LOG_ERROR("Invalid copy descriptor set (descriptorCopies[%u])", i);
            return false;
        }
    }

    for (auto i = 0; i < descriptorWrites.size(); i++)
    {
        const auto& write = descriptorWrites[i];
        write.dstSet->processWrite(write, writeValidationResults[i]);
    }
    for (auto i = 0; i < descriptorCopies.size(); i++)
    {
        const auto& copy = descriptorCopies[i];
        copy.dstSet->processCopy(copy, copyValidationResults[i]);
    }

    updateDescriptorSets_impl(params);

    return true;
}

bool ILogicalDevice::nullifyDescriptors(const std::span<const IGPUDescriptorSet::SDropDescriptorSet> dropDescriptors)
{
    SDropDescriptorSetsParams params = { .drops = dropDescriptors };
    for (auto i = 0u; i < dropDescriptors.size(); i++)
    {
        const auto& drop = dropDescriptors[i];
        auto ds = drop.dstSet;

        if (!ds || !ds->wasCreatedBy(this))
        {
            NBL_LOG_ERROR("Invalid drop description set was given (dropDescriptors[%u])", i);
            return false;
        }

        auto bindingType = ds->getBindingType(IGPUDescriptorSetLayout::CBindingRedirect::binding_number_t(drop.binding));
        auto writeCount = drop.count;
        switch (asset::IDescriptor::GetTypeCategory(bindingType))
        {
        case asset::IDescriptor::EC_BUFFER:
            params.bufferCount += writeCount;
            break;
        case asset::IDescriptor::EC_SAMPLER:
        case asset::IDescriptor::EC_IMAGE:
            params.imageCount += writeCount;
            break;
        case asset::IDescriptor::EC_BUFFER_VIEW:
            params.bufferViewCount += writeCount;
            break;
        case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
            params.accelerationStructureCount += writeCount;
            params.accelerationStructureWriteCount++;
            break;
        default: // validation failed
            NBL_LOG_ERROR("Invalid binding type (dropDescriptors[%u])", i);
            return false;
        }

        // (no binding)
        if (bindingType == asset::IDescriptor::E_TYPE::ET_COUNT)
        {
            NBL_LOG_ERROR("Invalid binding type (dropDescriptors[%u])", i);
            return false;
        }
    }

    for (const auto& drop : dropDescriptors)
    {
        auto ds = drop.dstSet;
        ds->dropDescriptors(drop);
    }

    nullifyDescriptors_impl(params);
    return true;
}

core::smart_refctd_ptr<IGPURenderpass> ILogicalDevice::createRenderpass(const IGPURenderpass::SCreationParams& params)
{
    IGPURenderpass::SCreationParamValidationResult validation = IGPURenderpass::validateCreationParams(params);
    if (!validation)
    {
        NBL_LOG_ERROR("Invalid parameters were given");
        return nullptr;
    }

    const auto& optimalTilingUsages = getPhysicalDevice()->getImageFormatUsagesOptimalTiling();
    auto invalidAttachment = [this, &optimalTilingUsages]<typename Layout, template<typename> class op_t>(const IGPURenderpass::SCreationParams::SAttachmentDescription<Layout, op_t>&desc) -> bool
    {
        // We won't support linear attachments, so implicitly satisfy:
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
    for (uint32_t i = 0u; i < validation.depthStencilAttachmentCount; i++)
        if (invalidAttachment(params.depthStencilAttachments[i]))
        {
            NBL_LOG_ERROR("Invalid depth stencil attachment was given (depthStencilAttachments[%u])", i);
            return nullptr;
        }
    for (uint32_t i = 0u; i < validation.colorAttachmentCount; i++)
        if (invalidAttachment(params.colorAttachments[i]))
        {
            NBL_LOG_ERROR("Invalid color attachment was given (colorAttachments[%u])", i);
            return nullptr;
        }

    const auto mixedAttachmentSamples = getEnabledFeatures().mixedAttachmentSamples;
    const auto supportedDepthResolveModes = getPhysicalDeviceLimits().supportedDepthResolveModes;
    const auto supportedStencilResolveModes = getPhysicalDeviceLimits().supportedStencilResolveModes;
    const auto maxColorAttachments = getPhysicalDeviceLimits().maxColorAttachments;
    const int32_t maxMultiviewViewCount = getPhysicalDeviceLimits().maxMultiviewViewCount;
    for (auto i = 0u; i < validation.subpassCount; i++)
    {
        using subpass_desc_t = IGPURenderpass::SCreationParams::SSubpassDescription;
        const subpass_desc_t& subpass = params.subpasses[i];

        auto depthSamples = static_cast<IGPUImage::E_SAMPLE_COUNT_FLAGS>(128);
        if (subpass.depthStencilAttachment.render.used())
        {
            depthSamples = params.depthStencilAttachments[subpass.depthStencilAttachment.render.attachmentIndex].samples;

            // TODO: seems like `multisampledRenderToSingleSampledEnable` needs resolve modes but not necessarily a resolve attachmen
            const hlsl::ResolveModeFlags depthResolve = static_cast<hlsl::ResolveModeFlags>(subpass.depthStencilAttachment.resolveMode.depth);
            const hlsl::ResolveModeFlags stencilResolve = static_cast<hlsl::ResolveModeFlags>(subpass.depthStencilAttachment.resolveMode.stencil);
            if (subpass.depthStencilAttachment.resolve.used() || /*multisampledToSingleSampledUsed*/false)
            {
                const auto& attachment = params.depthStencilAttachments[(subpass.depthStencilAttachment.resolve.used() ? subpass.depthStencilAttachment.resolve : subpass.depthStencilAttachment.render).attachmentIndex];

                const bool hasDepth = !asset::isStencilOnlyFormat(attachment.format);
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-depthResolveMode-03183
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06874
                if (hasDepth && !supportedDepthResolveModes.hasFlags(depthResolve))
                {
                    NBL_LOG_ERROR("Invalid stencil attachment's resolve mode (subpasses[%u])", i);
                    return nullptr;
                }
                ;
                const bool hasStencil = !asset::isDepthOnlyFormat(attachment.format);
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-stencilResolveMode-03184
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06875
                if (hasStencil && !supportedStencilResolveModes.hasFlags(stencilResolve))
                {
                    NBL_LOG_ERROR("Invalid stencil attachment's resolve mode (subpasses[%u])", i);
                    return nullptr;
                }

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescriptionDepthStencilResolve-pNext-06873
                if (/*multisampledToSingleSampledUsed*/false && depthResolve == hlsl::ResolveModeFlags::NONE && stencilResolve == hlsl::ResolveModeFlags::NONE)
                {
                    NBL_LOG_ERROR("Invalid stencil attachment's resolve mode (subpasses[%u])", i);
                    return nullptr;
                }
            }
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-colorAttachmentCount-03063
        for (auto j = maxColorAttachments; j < subpass_desc_t::MaxColorAttachments; j++)
            if (subpass.colorAttachments[j].render.used())
            {
                NBL_LOG_ERROR("Invalid color attachment (subpasses[%u].colorAttachments[%u])", i, static_cast<uint32_t>(j));
                return nullptr;
            }
        // TODO: support `VK_EXT_multisampled_render_to_single_sampled`
        auto samplesForAllColor = (depthSamples > IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_64_BIT || mixedAttachmentSamples/*||multisampledRenderToSingleSampled*/) ? static_cast<IGPUImage::E_SAMPLE_COUNT_FLAGS>(0) : depthSamples;
        for (auto j = 0u; j < maxColorAttachments; j++)
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
                if (samples > depthSamples)
                {
                    NBL_LOG_ERROR("Invalid color attachment (subpasses[%u].colorAttachments[%u])", i, static_cast<uint32_t>(j));
                    return nullptr;
                }
            }
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-multisampledRenderToSingleSampled-06869
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-multisampledRenderToSingleSampled-06872
            else if (!false/*multisampledRenderToSingleSampled*/ && samples != samplesForAllColor)
            {
                NBL_LOG_ERROR("Invalid color attachment (subpasses[%u].colorAttachments[%u])", i, static_cast<uint32_t>(j));
                return nullptr;
            }
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubpassDescription2-viewMask-06706
        if (hlsl::findMSB(subpass.viewMask) >= maxMultiviewViewCount)
        {
            NBL_LOG_ERROR("Invalid viewMask (subpasses[%u])", i);
            return nullptr;
        }
    }

    for (auto i = 0u; i < validation.dependencyCount; i++)
    {
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03054
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkRenderPassCreateInfo2-pDependencies-03055
    }

    return createRenderpass_impl(params, std::move(validation));
}

asset::ICPUPipelineCache::SCacheKey ILogicalDevice::getPipelineCacheKey() const
{
    const auto& props = m_physicalDevice->getProperties();
    asset::ICPUPipelineCache::SCacheKey key = { .deviceAndDriverUUID = "Nabla_v0.0.0_Vulkan_" }; // TODO: append version to Nabla
    {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (size_t i = 0; i < sizeof(props.pipelineCacheUUID); i++)
            ss << std::hex << std::setw(2) << static_cast<uint32_t>(props.pipelineCacheUUID[i]);
        key.deviceAndDriverUUID += ss.str();
    }
    return key;
}

bool ILogicalDevice::createGraphicsPipelines(
    IGPUPipelineCache* const pipelineCache,
    const std::span<const IGPUGraphicsPipeline::SCreationParams> params,
    core::smart_refctd_ptr<IGPUGraphicsPipeline>* const output
)
{
    std::fill_n(output, params.size(), nullptr);
    IGPUGraphicsPipeline::SCreationParams::SSpecializationValidationResult specConstantValidation = commonCreatePipelines(nullptr, params,
        [this](const IGPUShader::SSpecInfo& info)->bool
        {
            if (!info.shader)
                return false;
            return info.shader->wasCreatedBy(this);
        }
    );
    if (!specConstantValidation)
    {
        NBL_LOG_ERROR("Invalid parameters were given");
        return false;
    }

    const auto& features = getEnabledFeatures();
    const auto& limits = getPhysicalDeviceLimits();
    for (auto ix = 0u; ix < params.size(); ix++)
    {
        const auto& ci = params[ix];
        auto renderpass = ci.renderpass;
        if (!renderpass->wasCreatedBy(this))
        {
            NBL_LOG_ERROR("Invalid renderpass was given (params[%u])", ix);
            return false;
        }

        const auto& rasterParams = ci.cached.rasterization;
        if (rasterParams.alphaToOneEnable && !features.alphaToOne)
        {
            NBL_LOG_ERROR("Feature `alpha to one` is not enabled");
            return false;
        }
        if (rasterParams.depthBoundsTestEnable && !features.depthBounds)
        {
            NBL_LOG_ERROR("Feature `depth bounds` is not enabled");
            return false;
        }

        const auto samples = 0x1u << rasterParams.samplesLog2;

        // TODO: loads more validation on extra parameters here!
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766

        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-subpass-01505
        // baiscally the AMD version must have the rasterization samples equal to the maximum of all attachment samples counts

        const auto& passParams = renderpass->getCreationParameters();
        const auto& subpass = passParams.subpasses[ci.cached.subpassIx];
        if (subpass.viewMask)
        {
            /*
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-renderPass-06047
            if (!limits.multiviewTessellationShader && .test(tesS_contrOL))
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-renderPass-06048
            if (!limits.multiviewGeomtryShader && .test(GEOMETRY))
                return false;
            */
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-renderPass-06578
            //NOTE: index of MSB must be less than maxMultiviewViewCount; wrong negation here, should be >=
            if (hlsl::findMSB(subpass.viewMask) > limits.maxMultiviewViewCount)
            {
                NBL_LOG_ERROR("Invalid viewMask (params[%u])", ix);
                return false;
            }
        }
        if (subpass.depthStencilAttachment.render.used())
        {
            const auto& attachment = passParams.depthStencilAttachments[subpass.depthStencilAttachment.render.attachmentIndex];

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-multisampledRenderToSingleSampled-06853
            bool sampleCountNeedsToMatch = !features.mixedAttachmentSamples /*&& !features.multisampledRenderToSingleSampled*/;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-subpass-01411
            if (/*detect NV version && */(rasterParams.depthTestEnable() || rasterParams.stencilTestEnable() || rasterParams.depthBoundsTestEnable))
                sampleCountNeedsToMatch = true;
            if (sampleCountNeedsToMatch && attachment.samples != samples)
            {
                NBL_LOG_ERROR("Invalid depth stencil attachment (params[%u])", ix);
                return false;
            }
        }
        for (auto i = 0; i < IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments; i++)
        {
            const auto& render = subpass.colorAttachments[i].render;
            if (render.used())
            {
                const auto& attachment = passParams.colorAttachments[render.attachmentIndex];
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-renderPass-06041
                if (ci.cached.blend.blendParams[i].blendEnabled() && !getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[attachment.format].attachmentBlend)
                {
                    NBL_LOG_ERROR("Invalid color attachment (params[%u].colorAttachments[%u])", ix, i);
                    return false;
                }

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-multisampledRenderToSingleSampled-06853
                if (!features.mixedAttachmentSamples /*&& !features.multisampledRenderToSingleSampled*/ && attachment.samples != samples)
                {
                    NBL_LOG_ERROR("Invalid color attachment (params[%u].colorAttachments[%u])", ix, i);
                    return false;
                }
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-subpass-01412
                if (/*detect NV version && */(attachment.samples > samples))
                {
                    NBL_LOG_ERROR("Invalid color attachment (params[%u].colorAttachments[%u])", ix, i);
                    return false;
                }
            }
        }
    }
    createGraphicsPipelines_impl(pipelineCache, params, output, specConstantValidation);

    for (auto i = 0u; i < params.size(); i++)
        if (!output[i])
        {
            NBL_LOG_ERROR("GraphicPipeline was not created (params[%u])", i);
            return false;
        }
    return true;
}

bool ILogicalDevice::createRayTracingPipelines(IGPUPipelineCache* const pipelineCache, 
  const std::span<const IGPURayTracingPipeline::SCreationParams> params,
  core::smart_refctd_ptr<IGPURayTracingPipeline>* const output)
{
    std::fill_n(output,params.size(),nullptr);
    IGPURayTracingPipeline::SCreationParams::SSpecializationValidationResult specConstantValidation = commonCreatePipelines(pipelineCache,params,[this](const IGPUShader::SSpecInfo& info)->bool
    {
        if (!info.shader->wasCreatedBy(this))
        {
            NBL_LOG_ERROR("The shader was not created by this device");
            return false;
        }
        return true;
    });
    if (!specConstantValidation)
    {
        NBL_LOG_ERROR("Invalid parameters were given");
        return false;
    }

    const auto& features = getEnabledFeatures();

    // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-vkCreateRayTracingPipelinesKHR-rayTracingPipeline-03586
    if (!features.rayTracingPipeline)
    {
        NBL_LOG_ERROR("Feature `ray tracing pipeline` is not enabled");
        return false;
    }

    for (const auto& param : params)
    {
        const bool skipAABBs = bool(param.flags & IGPURayTracingPipeline::SCreationParams::FLAGS::SKIP_AABBS);
        const bool skipBuiltin = bool(param.flags & IGPURayTracingPipeline::SCreationParams::FLAGS::SKIP_BUILT_IN_PRIMITIVES);

        // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-rayTraversalPrimitiveCulling-03597
        if (skipAABBs && skipBuiltin)
        {
            NBL_LOG_ERROR("Flags must not include both SKIP_AABBS and SKIP_BUILT_IN_PRIMITIVE!");
            return false;
        }

        // https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#VUID-VkRayTracingPipelineCreateInfoKHR-rayTraversalPrimitiveCulling-03596
        if (skipAABBs && !features.rayTraversalPrimitiveCulling)
        {
          NBL_LOG_ERROR("Feature `rayTraversalPrimitiveCulling` is not enabled when pipeline is created with SKIP_AABBS");
          return false;
        }

        // https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#VUID-VkRayTracingPipelineCreateInfoKHR-rayTraversalPrimitiveCulling-03597
        if (skipBuiltin && !features.rayTraversalPrimitiveCulling)
        {
          NBL_LOG_ERROR("Feature `rayTraversalPrimitiveCulling` is not enabled when pipeline is created with SKIP_BUILT_IN_PRIMITIVES");
          return false;
        }

    }

    const auto& limits = getPhysicalDeviceLimits();
    for (const auto& param : params)
    {
        // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-maxPipelineRayRecursionDepth-03589
        if (param.cached.maxRecursionDepth > limits.maxRayRecursionDepth)
        {
            NBL_LOG_ERROR("Invalid maxRecursionDepth. maxRecursionDepth(%u) exceed the limits(%u)", param.cached.maxRecursionDepth, limits.maxRayRecursionDepth);
            return false;
        }
        if (param.getShaders().empty())
        {
            NBL_LOG_ERROR("Pipeline must have at least one shader.");
            return false;
        }
    }

    createRayTracingPipelines_impl(pipelineCache,params,output,specConstantValidation);
    
    bool retval = true;
    for (auto i=0u; i<params.size(); i++)
    {
        if (!output[i])
        {
            NBL_LOG_ERROR("RayTracingPipeline was not created (params[%u])", i);
            return false;
        }
    }
    return retval;
}
#include "nbl/undef_logging_macros.h"