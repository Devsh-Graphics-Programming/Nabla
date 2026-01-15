#ifndef _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_
#define _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_

#include "nbl/asset/IPipelineLayout.h"
#include "nbl/video/declarations.h"

namespace nbl::ext::envmap_importance_sampling
{

class EnvmapImportanceSampling
{
  public:

    struct SCachedCreationParameters
    {
        // using streaming_buffer_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;
        //
        // static constexpr inline auto RequiredAllocateFlags = core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
        // static constexpr inline auto RequiredUsageFlags = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
        //
        // DrawMode drawMode = ADM_DRAW_BOTH;

        core::smart_refctd_ptr<video::IUtilities> utilities;

        //! optional, default MDI buffer allocated if not provided
        // core::smart_refctd_ptr<streaming_buffer_t> streamingBuffer = nullptr;
    };

    struct SCreationParameters : public SCachedCreationParameters
    {
        video::IQueue* transfer = nullptr;  // only used to make the 24 element index buffer and instanced pipeline on create
        core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;

        core::smart_refctd_ptr<video::IGPUPipelineLayout> genLumaPipelineLayout = nullptr;

        inline bool validate() const
        {
            const auto validation = std::to_array
            ({
                std::make_pair(bool(assetManager), "Invalid `creationParams.assetManager` is nullptr!"),
                std::make_pair(bool(utilities), "Invalid `creationParams.utilities` is nullptr!"),
                std::make_pair(bool(transfer), "Invalid `creationParams.transfer` is nullptr!"),
                std::make_pair(bool(utilities->getLogicalDevice()->getPhysicalDevice()->getQueueFamilyProperties()[transfer->getFamilyIndex()].queueFlags.hasFlags(video::IQueue::FAMILY_FLAGS::TRANSFER_BIT)), "Invalid `creationParams.transfer` is not capable of transfer operations!")
            });

            system::logger_opt_ptr logger = utilities->getLogger();
            for (const auto& [ok, error] : validation)
                if (!ok)
                {
                    logger.log(error, system::ILogger::ELL_ERROR);
                    return false;
                }

            assert(bool(assetManager->getSystem()));

            return true;
        }

    };

    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createGenLumaPipelineLayout(video::ILogicalDevice* device, const core::smart_refctd_ptr<video::IGPUSampler>* sampler);

    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createMeasureLumaPipelineLayout(video::ILogicalDevice* device);

    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createGenWarpMapPipelineLayout(video::ILogicalDevice* device);

    //! mounts the extension's archive to given system - useful if you want to create your own shaders with common header included
    static const core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias = "");

    static core::smart_refctd_ptr<video::IGPUComputePipeline> createGenLumaPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout);

    static core::smart_refctd_ptr<video::IGPUComputePipeline> createMeasureLumaPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout);
  private:
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_lumaGenPipeline;
  
};

}
#endif
