#ifndef __IRR_I_GPU_QUEUE_FAMILY_H_INCLUDED__
#define __IRR_I_GPU_QUEUE_FAMILY_H_INCLUDED__

#include <nbl/asset/IImage.h>  //for VkExtent3D only
#include <type_traits>

namespace nbl
{
namespace video
{
//possibly move into IGPUQueueFamily
enum E_QUEUE_FLAGS : uint32_t
{
    EQF_GRAPHICS_BIT = 0x01,
    EQF_COMPUTE_BIT = 0x02,
    EQF_TRANSFER_BIT = 0x04,
    EQF_SPARSE_BINDING_BIT = 0x08,
    EQF_PROTECTED_BIT = 0x10
};

//possibly move into IGPUQueueFamily
struct SQueueFamilyProperties
{
    std::underlying_type_t<E_QUEUE_FLAGS> queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    asset::VkExtent3D minImageTransferGranularity;
};

}
}

#endif