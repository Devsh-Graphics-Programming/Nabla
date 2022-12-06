#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"
#include "nbl/video/utilities/renderdoc.h"
#include "nbl/video/ECommonEnums.h"
#include "nbl/asset/utils/CGLSLCompiler.h"

namespace nbl::video
{

class IPhysicalDevice;

class NBL_API2 IAPIConnection : public core::IReferenceCounted
{
    public:

        // Equivalent to Instance Extensions and Layers
        // Any device feature that has an api connection feature dependency that is not enabled is considered to be unsupported,
        //  for example you need to enable E_SWAPCHAIN_MODE::ESM_SURFACE in order for the physical device to report support in SPhysicalDeviceFeatures::swapchainMode
        struct SFeatures
        {
            // VK_KHR_surface, VK_KHR_win32_surface, VK_KHR_display(TODO)
            core::bitflag<E_SWAPCHAIN_MODE> swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
            
            // VK_LAYER_KHRONOS_validation (instance layer) 
            bool validations = false;

            // VK_EXT_debug_utils
            // When combined with validation layers, even more detailed feedback on the application�s use of Vulkan will be provided.
            //  The ability to create a debug messenger which will pass along debug messages to an application supplied callback.
            //  The ability to identify specific Vulkan objects using a name or tag to improve tracking.
            //  The ability to identify specific sections within a VkQueue or VkCommandBuffer using labels to aid organization and offline analysis in external tools.
            bool debugUtils = false;
        };

        virtual E_API_TYPE getAPIType() const = 0;

        virtual IDebugCallback* getDebugCallback() const = 0;

        core::SRange<IPhysicalDevice* const> getPhysicalDevices() const;

        const SFeatures& getEnabledFeatures() const { return m_enabledFeatures; };

    protected:
        IAPIConnection(const SFeatures& enabledFeatures);

        std::vector<std::unique_ptr<IPhysicalDevice>> m_physicalDevices;
        renderdoc_api_t* m_rdoc_api;
        SFeatures m_enabledFeatures = {};
};

}


#endif