#include "nbl/video/debug/CVulkanDebugCallback.h"

namespace nbl::video
{

CVulkanDebugCallback::~CVulkanDebugCallback()
{
    vkDestroyDebugUtilsMessengerEXT(m_api->getInternalObject(), m_vkDebugUtilsMessengerEXT, nullptr);
}

}