#include "nbl/video/surface/CSurfaceVKWin32.h"

#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{

core::smart_refctd_ptr<CSurfaceVKWin32> CSurfaceVKWin32::create(const IAPIConnection* api, SCreationParams&& params)
{
    return core::make_smart_refctd_ptr<CSurfaceVKWin32>(
        static_cast<const CVulkanConnection*>(api)->getInternalObject(), std::move(params));
}

}