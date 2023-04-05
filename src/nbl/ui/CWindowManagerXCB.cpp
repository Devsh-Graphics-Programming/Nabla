#include "nbl/ui/CWindowManagerXCB.h"

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/ui/CWindowManagerXCB.h"
#include "nbl/ui/CWindowXCB.h"

using namespace nbl;
using namespace nbl::ui;

core::smart_refctd_ptr<IWindowManagerXCB> IWindowManagerXCB::create()
{
    return core::make_smart_refctd_ptr<CWindowManagerXCB>();
}


CWindowManagerXCB::CWindowManagerXCB() {
}

core::smart_refctd_ptr<IWindow> CWindowManagerXCB::createWindow(IWindow::SCreationParams&& creationParams)
{
    std::string title = std::string(creationParams.windowCaption);
    auto window = core::make_smart_refctd_ptr<CWindowXCB>(core::smart_refctd_ptr<CWindowManagerXCB>(this), std::move(creationParams));
    window->setCaption(title);
    return window;
}

bool CWindowManagerXCB::setWindowSize_impl(IWindow* window, uint32_t width, uint32_t height) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowSize_impl(width, height);
    return true;
}

bool CWindowManagerXCB::setWindowPosition_impl(IWindow* window, int32_t x, int32_t y) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowPosition_impl(x, y);
    return true;
}

bool CWindowManagerXCB::setWindowRotation_impl(IWindow* window, bool landscape) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowRotation_impl(landscape);
    return true;
}

bool CWindowManagerXCB::setWindowVisible_impl(IWindow* window, bool visible) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowVisible_impl(visible);
    return true;
}

bool CWindowManagerXCB::setWindowMaximized_impl(IWindow* window, bool maximized) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowMaximized_impl(maximized);
    return true;
}

#endif