#ifndef _NBL_UI_I_WINDOWMANAGER_XCB_INCLUDED_
#define _NBL_UI_I_WINDOWMANAGER_XCB_INCLUDED_

#include "nbl/ui/IWindowManager.h"

#ifdef _NBL_PLATFORM_LINUX_
namespace nbl::ui
{

// Native X11 window manager built on XCB, also works under XWayland.
//
// Threading and lifetime rules:
// - The manager owns one `xcb_connection_t`, opened by `create()` (from `$DISPLAY`) and closed when the manager is destroyed.
//   `create()` returns nullptr if no X server can be reached.
// - One dedicated thread per manager reads all X events. Every `IWindow::IEventCallback` of every window created by this
//   manager is called on that thread, never on the thread that created the window, and never concurrently with itself.
// - Windows hold a reference to their manager, so the manager (and its connection) outlives all its windows.
// - Window creation and destruction are processed on the event thread, so once a window's destructor returns no more
//   callbacks will be made for it.
// - Callbacks may create windows, and may drop the last reference to any window other than the one they were raised for.
//   Dropping the last reference to the window whose callback is running is not allowed (asserted in debug builds),
//   because `IWindow::IEventCallback` still updates that window's flags after the callback returns.
//   Defer it, e.g. by flagging the window and releasing it from your own thread.
// - `onWindowClosed` is raised for `WM_DELETE_WINDOW`. If it returns true the window is unmapped (hidden), not destroyed.
//   The X window lives until the last reference to the `IWindow` (held by e.g. a Vulkan surface) is dropped.
// - All other manager and window methods may be called from any thread. They only send requests to the X server,
//   the resulting state changes are reported back through the callbacks.
class IWindowManagerXcb : public IWindowManager
{
	public:
		NBL_API2 static core::smart_refctd_ptr<IWindowManagerXcb> create();
};

}
#endif
#endif
