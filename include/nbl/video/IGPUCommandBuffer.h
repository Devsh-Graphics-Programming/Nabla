#ifndef __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/asset/ICommandBuffer.h"
/*
#include "nbl/video/IGPUImage.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
*/
#include "nbl/video/IGPUDescriptorSet.h"
/*
#include "nbl/video/IGPUPipelineLayout.h"
*/
#include "nbl/video/IGPUEvent.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUCommandPool.h"

namespace nbl::video
{

class NBL_API IGPUCommandBuffer :
    public core::IReferenceCounted,
    public asset::ICommandBuffer<
        IGPUBuffer,
        IGPUImage,
        IGPUImageView,
        IGPURenderpass,
        IGPUFramebuffer,
        IGPUGraphicsPipeline,
        IGPUComputePipeline,
        IGPUDescriptorSet,
        IGPUPipelineLayout,
        IGPUEvent,
        IGPUCommandBuffer
    >,
    public IBackendObject
{
    using base_t = asset::ICommandBuffer<
        IGPUBuffer,
        IGPUImage,
        IGPUImageView,
        IGPURenderpass,
        IGPUFramebuffer,
        IGPUGraphicsPipeline,
        IGPUComputePipeline,
        IGPUDescriptorSet,
        IGPUPipelineLayout,
        IGPUEvent,
        IGPUCommandBuffer
    >;

public:

    inline bool isResettable() const
    {
        return m_cmdpool->getCreationFlags().hasFlags(IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
    }

    inline bool canReset() const
    {
        if(isResettable())
            return m_state != ES_PENDING;
        return false;
    }

    virtual bool begin(core::bitflag<E_USAGE> _flags, const SInheritanceInfo* inheritanceInfo = nullptr)
    {
        if (!isResettable())
        {
            if(m_state != ES_INITIAL)
            {
                assert(false);
                return false;
            }
        }

        if(m_state == ES_PENDING)
        {
            assert(false);
            return false;
        }

        if (inheritanceInfo != nullptr)
            m_cachedInheritanceInfo = *inheritanceInfo;

        return base_t::begin(_flags);
    }

    virtual bool reset(core::bitflag<E_RESET_FLAGS> _flags)
    {
        if (!canReset())
        {
            assert(false);
            return false;
        }
        return base_t::reset(_flags);
    }

    uint32_t getQueueFamilyIndex() const { return m_cmdpool->getQueueFamilyIndex(); }

    IGPUCommandPool* getPool() const { return m_cmdpool.get(); }

    bool regenerateMipmaps(IGPUImage* img, uint32_t lastReadyMip, asset::IImage::E_ASPECT_FLAGS aspect) override
    {
        const uint32_t qfam = getQueueFamilyIndex();

        IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
        barrier.srcQueueFamilyIndex = qfam;
        barrier.dstQueueFamilyIndex = qfam;
        barrier.image = core::smart_refctd_ptr<video::IGPUImage>(img);
        barrier.subresourceRange.aspectMask = aspect;
        barrier.subresourceRange.levelCount = 1u;
        barrier.subresourceRange.baseArrayLayer = 0u;
        barrier.subresourceRange.layerCount = img->getCreationParameters().arrayLayers;

        asset::SImageBlit blitRegion = {};
        blitRegion.srcSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.srcSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.srcSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.srcOffsets[0] = { 0, 0, 0 };

        blitRegion.dstSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.dstSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.dstSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.dstOffsets[0] = { 0, 0, 0 };

        auto mipsize = img->getMipSize(lastReadyMip);

        uint32_t mipWidth = mipsize.x;
        uint32_t mipHeight = mipsize.y;
        uint32_t mipDepth = mipsize.z;
        for (uint32_t i = lastReadyMip + 1u; i < img->getCreationParameters().mipLevels; ++i)
        {
            const uint32_t srcLoD = i - 1u;
            const uint32_t dstLoD = i;

            barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
            barrier.oldLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = asset::EIL_TRANSFER_SRC_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = dstLoD;

            if (srcLoD > lastReadyMip)
            {
                if (!pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u, nullptr, 0u, nullptr, 1u, &barrier))
                    return false;
            }

            const auto srcMipSz = img->getMipSize(srcLoD);

            blitRegion.srcSubresource.mipLevel = srcLoD;
            blitRegion.srcOffsets[1] = { srcMipSz.x, srcMipSz.y, srcMipSz.z };

            blitRegion.dstSubresource.mipLevel = dstLoD;
            blitRegion.dstOffsets[1] = { mipWidth, mipHeight, mipDepth };

            if (!blitImage(img, asset::EIL_TRANSFER_SRC_OPTIMAL, img, asset::EIL_TRANSFER_DST_OPTIMAL, 1u, &blitRegion, asset::ISampler::ETF_LINEAR))
                return false;

            if (mipWidth > 1u) mipWidth /= 2u;
            if (mipHeight > 1u) mipHeight /= 2u;
            if (mipDepth > 1u) mipDepth /= 2u;
        }

        return true;
    }

    SInheritanceInfo getCachedInheritanceInfo() const
    {
        return m_cachedInheritanceInfo;
    }

    // OpenGL: nullptr, because commandbuffer doesn't exist in GL (we might expose the linked list command storage in the future)
    // Vulkan: const VkCommandBuffer*
    virtual const void* getNativeHandle() const = 0;

protected:
    friend class IGPUQueue;

    IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool) : base_t(lvl), IBackendObject(std::move(dev)), m_cmdpool(_cmdpool)
    {
    }
    virtual ~IGPUCommandBuffer() = default;

    core::smart_refctd_ptr<IGPUCommandPool> m_cmdpool;
    SInheritanceInfo m_cachedInheritanceInfo;

    inline bool validate_updateBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData)
    {
        if (!this->isCompatibleDevicewise(dstBuffer))
            return false;
        if ((dstOffset & 0x03ull) != 0ull)
            return false;
        if ((dataSize & 0x03ull) != 0ull)
            return false;
        if (dataSize > 65536ull)
            return false;
        return dstBuffer->getCachedCreationParams().canUpdateSubRange;
    }

    static void bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count, const IGPUDescriptorSet* const* _descSets, const IGPUPipelineLayout** const _destPplnLayouts)
    {
        int32_t compatibilityLimits[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
        {
            const int32_t lim = _destPplnLayouts[i] ? //if no descriptor set bound at this index
                _destPplnLayouts[i]->isCompatibleUpToSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _newLayout) : -1;

            compatibilityLimits[i] = lim;
        }

        /*
        https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#descriptorsets-compatibility
        When binding a descriptor set (see Descriptor Set Binding) to set number N, if the previously bound descriptor sets for sets zero through N-1 were all bound using compatible pipeline layouts, then performing this binding does not disturb any of the lower numbered sets.
        */
        for (int32_t i = 0; i < static_cast<int32_t>(_first); ++i)
            if (compatibilityLimits[i] < i)
                _destPplnLayouts[i] = nullptr;
        /*
        If, additionally, the previous bound descriptor set for set N was bound using a pipeline layout compatible for set N, then the bindings in sets numbered greater than N are also not disturbed.
        */
        if (compatibilityLimits[_first] < static_cast<int32_t>(_first))
            for (uint32_t i = _first + 1u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
                _destPplnLayouts[i] = nullptr;
    }
};

}

#endif
