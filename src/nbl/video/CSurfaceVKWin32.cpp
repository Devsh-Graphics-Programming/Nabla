#include "nbl/video/surface/CSurfaceVKWin32.h"

#include "nbl/video/CVulkanConnection.h"

namespace nbl {
namespace video
{

core::smart_refctd_ptr<CSurfaceVKWin32> CSurfaceVKWin32::create(const IAPIConnection* api, SCreationParams&& params)
{
    const CVulkanConnection* vk = static_cast<const CVulkanConnection*>(api);
    VkInstance instance = vk->getInternalObject();

    return core::make_smart_refctd_ptr<CSurfaceVKWin32>(instance, std::move(params));
}

}
}