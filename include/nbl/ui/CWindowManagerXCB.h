#ifndef _NBL_UI_C__WINDOWMANAGER_XCB_INCLUDED_
#define _NBL_UI_C__WINDOWMANAGER_XCB_INCLUDED_

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/Types.h"

#include "nbl/ui/IWindow.h"
#include "nbl/ui/IWindowManagerXCB.h"

namespace nbl::ui
{

class CWindowManagerXCB final : public IWindowManagerXCB
{
public:

	bool setWindowSize_impl(IWindow* window, uint32_t width, uint32_t height) override;
	bool setWindowPosition_impl(IWindow* window, int32_t x, int32_t y) override;
	bool setWindowRotation_impl(IWindow* window, bool landscape) override;
	bool setWindowVisible_impl(IWindow* window, bool visible) override;
	bool setWindowMaximized_impl(IWindow* window, bool maximized) override;

	inline SDisplayInfo getPrimaryDisplayInfo() const override final {
		return SDisplayInfo();
	}

    CWindowManagerXCB();
    ~CWindowManagerXCB() override = default;
	
	core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override;

	void destroyWindow(IWindow* wnd) override final {}

    const Xcb& getXcbFunctionTable() const override { return m_xcb; }
    const XcbIcccm& getXcbIcccmFunctionTable() const override { return m_xcbIcccm; }

private:
	Xcb m_xcb = Xcb("xcb"); // function tables
	XcbIcccm m_xcbIcccm = XcbIcccm("xcb-icccm");
};


}
#endif
#endif