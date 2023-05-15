#ifndef _NBL_UI_I_GRAPHICAL_APPLICATION_FRAMEWORK_H_INCLUDED_
#define _NBL_UI_I_GRAPHICAL_APPLICATION_FRAMEWORK_H_INCLUDED_

#include "nbl/ui/IWindow.h"

#include <vector>
#include <string>

namespace nbl::ui
{

// no because nothing in Nabla uses it
class IGraphicalApplicationFramework
{
	public:
		virtual nbl::ui::IWindow* getWindow() = 0;
		virtual video::IAPIConnection* getAPIConnection() = 0;
		virtual video::ILogicalDevice* getLogicalDevice() = 0;
		virtual video::IGPURenderpass* getRenderpass() = 0;

		virtual void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) = 0;
		virtual void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& window) = 0;
		virtual void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) = 0;
		virtual void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) = 0;

		virtual uint32_t getSwapchainImageCount() = 0;
		virtual nbl::asset::E_FORMAT getDepthFormat() = 0;
};

}
#endif