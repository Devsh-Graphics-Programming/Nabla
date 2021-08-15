#include "CVulkanConnection.h"

namespace nbl::video
{

#if 0
core::smart_refctd_ptr<ISurface> CVulkanConnection::createSurface(ui::IWindow* window) const
{
    // Todo(achal): handle other platforms
#ifdef _NBL_PLATFORM_WINDOWS_
    {
        ui::IWindowWin32* w32 = static_cast<ui::IWindowWin32*>(window);

        CSurfaceVKWin32::SCreationParams params;
        params.hinstance = GetModuleHandle(NULL);
        params.hwnd = w32->getNativeHandle();

        return CSurfaceVKWin32::create(this, std::move(params));
    }
#endif
}
#endif

}