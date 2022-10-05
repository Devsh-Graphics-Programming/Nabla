#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl::video
{

IDescriptorPool::SDescriptorOffsets IDescriptorPool::allocateDescriptors(const IGPUDescriptorSetLayout* layout)
{
    SDescriptorOffsets offsets;

    for (uint32_t i = 0u; i < asset::EDT_COUNT; ++i)
    {
        const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(i);
        const auto count = layout->getTotalDescriptorCount(type);
        if (count == 0ull)
            continue;

        if (m_flags & ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[type] = m_generalAllocators[type].alloc_addr(count, 1u);
        else
            offsets.data[type] = m_linearAllocators[type].alloc_addr(count, 1u);

        assert(core::LinearAddressAllocator<uint32_t>::invalid_address == core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address);
        assert((offsets.data[type] != core::LinearAddressAllocator<uint32_t>::invalid_address) && "PANIC: Allocation failed. This shoudn't have happened!");
    }

    return offsets;
}

}