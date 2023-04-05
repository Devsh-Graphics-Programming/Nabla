#ifndef _NBL_UI_I_WINDOWMANAGER_XCB_INCLUDED_
#define _NBL_UI_I_WINDOWMANAGER_XCB_INCLUDED_

#include "nbl/ui/IWindowManager.h"

#ifdef _NBL_PLATFORM_LINUX_

#include <xcb/xcb.h>
#include <xcb/xcb_icccm.h>
#include <xcb/xproto.h>

namespace nbl::ui {

class IWindowManagerXCB : public IWindowManager 
{
	public:
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

		NBL_API2 static core::smart_refctd_ptr<IWindowManagerXCB> create();
};

} // namespace nbl::ui

#endif
#endif