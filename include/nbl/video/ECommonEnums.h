#ifndef __NBL_VIDEO_E_COMMON_ENUMS_H_INCLUDED__
#define __NBL_VIDEO_E_COMMON_ENUMS_H_INCLUDED__

namespace nbl::video
{

enum E_SWAPCHAIN_MODE : uint8_t
{
    ESM_NONE = 0,
    ESM_SURFACE = 0x01,
    // ESM_DISPLAY = 0x02 TODO, as we won't write the API interfaces to deal with direct-to-display swapchains yet.,
    /* TODO: VK_KHR_swapchain if SURFACE or DISPLAY flag present & KHR_display_swapchain if DISPLAY flag present */
    // The VK_KHR_swapchain (device) extension is the device-level companion to the VK_KHR_surface (instance)
    // The VK_KHR_display_swapchain - device extension:
        // Requires VK_KHR_swapchain to be enabled for any device-level functionality
        // Requires VK_KHR_display to be enabled for any device-level functionality
    // The VK_KHR_display - instance extension
        // Requires VK_KHR_surface to be enabled
};

}

#endif