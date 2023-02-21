#ifndef C_WINDOW_MANAGER_XCB
#define C_WINDOW_MANAGER_XCB

#ifdef _NBL_PLATFORM_LINUX_
#include "nbl/core/decl/Types.h"

#include "nbl/system/DefaultFuncPtrLoader.h"

#include "nbl/ui/IWindow.h"
#include "nbl/ui/IWindowManager.h"

#include <functional>
#include <memory>
#include <string>

#include <xcb/xcb.h>
#include <xcb/xcb_icccm.h>
#include <xcb/xproto.h>

namespace nbl::ui
{

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xcb, system::DefaultFuncPtrLoader,
    xcb_destroy_window,
    xcb_generate_id,
    xcb_create_window,
    xcb_connect,
    xcb_disconnect,
    xcb_map_window,
    xcb_get_setup,
    xcb_setup_roots_iterator,
    xcb_flush,
    xcb_intern_atom,
    xcb_intern_atom_reply,
    xcb_unmap_window,
    xcb_get_property,
    xcb_get_property_reply,
    xcb_get_property_value_length,
    xcb_change_property,
    xcb_configure_window_checked,
    xcb_get_property_value,
    xcb_wait_for_event,
    xcb_send_event,
    xcb_request_check,
    xcb_delete_property,
    xcb_change_window_attributes,
    xcb_warp_pointer,
    xcb_query_pointer,
    xcb_query_pointer_reply,
    xcb_get_selection_owner_reply,
    xcb_get_selection_owner
);

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(XcbIcccm, system::DefaultFuncPtrLoader,
    xcb_icccm_set_wm_hints,
    xcb_icccm_size_hints_set_size,
    xcb_icccm_size_hints_set_min_size,
    xcb_icccm_size_hints_set_max_size,
    xcb_icccm_set_wm_normal_hints
);

class CWindowManagerXcb : public IWindowManager
{
public:

	virtual bool setWindowSize_impl(IWindow* window, uint32_t width, uint32_t height) override;
	virtual bool setWindowPosition_impl(IWindow* window, int32_t x, int32_t y) override;
	virtual bool setWindowRotation_impl(IWindow* window, bool landscape) override;
	virtual bool setWindowVisible_impl(IWindow* window, bool visible) override;
	virtual bool setWindowMaximized_impl(IWindow* window, bool maximized) override;

	inline SDisplayInfo getPrimaryDisplayInfo() const override final {
		return SDisplayInfo();
	}

    CWindowManagerXcb();
    ~CWindowManagerXcb() override = default;

	virtual core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override;

	virtual void destroyWindow(IWindow* wnd) override final {}

    const Xcb& getXcbFunctionTable() const { return m_xcb; }
    const XcbIcccm& getXcbIcccmFunctionTable() const { return m_xcbIcccm; }

private:

	Xcb m_xcb = Xcb("xcb"); // function tables
	XcbIcccm m_xcbIcccm = XcbIcccm("xcb-icccm");
};


}
#endif
#endif