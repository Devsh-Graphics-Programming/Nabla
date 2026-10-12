#include "nbl/ui/CWindowXcb.h"
#include "nbl/ui/CWindowManagerXcb.h"

#ifdef _NBL_PLATFORM_LINUX_
#include <cstdlib>

using namespace nbl;
using namespace nbl::ui;

CWindowXcb::CWindowXcb(SCreationParams&& params, core::smart_refctd_ptr<CWindowManagerXcb>&& winManager, xcb_window_t window) :
	IWindowXcb(std::move(params)), m_windowManager(std::move(winManager)), m_native{m_windowManager->getConnection(),window}
{
	// not mapped yet, `onMapped` will clear it with an `onWindowShown` like `WM_SHOWWINDOW` does on Win32
	m_flags |= ECF_HIDDEN;
	// the window manager tells us about these through `_NET_WM_STATE` once mapped
	m_flags &= ~core::bitflag(ECF_MINIMIZED);
	m_flags &= ~core::bitflag(ECF_MAXIMIZED);
	// we track these ourselves from focus and crossing events
	m_flags &= ~core::bitflag(ECF_INPUT_FOCUS);
	m_flags &= ~core::bitflag(ECF_MOUSE_FOCUS);
}

void CWindowXcb::setCaption(const std::string_view& caption)
{
	auto* const connection = m_native.connection;
	const auto& atoms = m_windowManager->getAtoms();
	xcb_change_property(connection,XCB_PROP_MODE_REPLACE,m_native.window,XCB_ATOM_WM_NAME,XCB_ATOM_STRING,8,caption.size(),caption.data());
	xcb_change_property(connection,XCB_PROP_MODE_REPLACE,m_native.window,atoms._NET_WM_NAME,atoms.UTF8_STRING,8,caption.size(),caption.data());
	xcb_flush(connection);
}

void CWindowXcb::onDeleteRequested()
{
	if (!m_cb || m_cb->onWindowClosed(this))
	{
		// Don't destroy, a Vulkan surface or swapchain may still reference the X window. It goes away with the last `IWindow` reference.
		xcb_unmap_window(m_native.connection,m_native.window);
		xcb_flush(m_native.connection);
	}
}

void CWindowXcb::onConfigured(const xcb_configure_notify_event_t* ev)
{
	int32_t x = ev->x;
	int32_t y = ev->y;
	// Real events are relative to the parent, which is the frame of a reparenting window manager.
	// Synthetic ones sent by the window manager are already in root coordinates (ICCCM 4.1.5).
	if (!(ev->response_type&0x80u))
	{
		auto* const connection = m_native.connection;
		const auto cookie = xcb_translate_coordinates(connection,m_native.window,m_windowManager->getRootWindow(),0,0);
		if (auto* reply=xcb_translate_coordinates_reply(connection,cookie,nullptr))
		{
			x = reply->dst_x;
			y = reply->dst_y;
			free(reply);
		}
	}
	if (!m_cb)
		return;
	if (x!=m_x || y!=m_y)
		(void)m_cb->onWindowMoved(this,x,y);
	if (ev->width!=m_width || ev->height!=m_height)
		(void)m_cb->onWindowResized(this,ev->width,ev->height);
}

void CWindowXcb::onMapped()
{
	// restoring from minimized maps the window again, `updateState` handles that
	updateState();
	if (m_cb && isHidden())
		(void)m_cb->onWindowShown(this);
}

void CWindowXcb::onUnmapped()
{
	// minimizing also unmaps, but that is not hiding the window
	updateState();
	if (m_cb && !m_minimizedState && !isHidden())
		(void)m_cb->onWindowHidden(this);
}

void CWindowXcb::onFocusChanged(const bool keyboard, const bool gained)
{
	const auto flag = keyboard ? ECF_INPUT_FOCUS:ECF_MOUSE_FOCUS;
	// X sends several focus events per change (e.g. for the frame), only report real transitions
	if (m_flags.hasFlags(flag)==gained)
		return;
	if (gained)
		m_flags |= flag;
	else
		m_flags &= ~core::bitflag(flag);
	if (!m_cb)
		return;
	if (keyboard)
	{
		if (gained)
			m_cb->onGainedKeyboardFocus(this);
		else
			m_cb->onLostKeyboardFocus(this);
	}
	else
	{
		if (gained)
			m_cb->onGainedMouseFocus(this);
		else
			m_cb->onLostMouseFocus(this);
	}
}

void CWindowXcb::updateState()
{
	auto* const connection = m_native.connection;
	const auto& atoms = m_windowManager->getAtoms();

	// send both requests before waiting
	const auto netStateCookie = xcb_get_property(connection,0,m_native.window,atoms._NET_WM_STATE,XCB_ATOM_ATOM,0,32);
	const auto wmStateCookie = xcb_get_property(connection,0,m_native.window,atoms.WM_STATE,atoms.WM_STATE,0,2);

	bool hidden = false, maximizedVert = false, maximizedHorz = false;
	if (auto* reply=xcb_get_property_reply(connection,netStateCookie,nullptr))
	{
		if (reply->format==32)
		{
			const auto* state = reinterpret_cast<const xcb_atom_t*>(xcb_get_property_value(reply));
			const auto count = xcb_get_property_value_length(reply)/sizeof(xcb_atom_t);
			for (auto i=0u; i<count; i++)
			{
				hidden = hidden || state[i]==atoms._NET_WM_STATE_HIDDEN;
				maximizedVert = maximizedVert || state[i]==atoms._NET_WM_STATE_MAXIMIZED_VERT;
				maximizedHorz = maximizedHorz || state[i]==atoms._NET_WM_STATE_MAXIMIZED_HORZ;
			}
		}
		free(reply);
	}
	bool iconic = false;
	if (auto* reply=xcb_get_property_reply(connection,wmStateCookie,nullptr))
	{
		// ICCCM `WM_STATE`: state, icon window
		if (reply->format==32 && xcb_get_property_value_length(reply)>=4)
			iconic = *reinterpret_cast<const uint32_t*>(xcb_get_property_value(reply))==3u;
		free(reply);
	}

	const bool minimized = hidden || iconic;
	const bool maximized = maximizedVert && maximizedHorz;
	if (minimized!=m_minimizedState)
	{
		m_minimizedState = minimized;
		if (minimized)
		{
			if (m_cb)
				(void)m_cb->onWindowMinimized(this);
		}
		else // there is no restore callback, but `IWindowManager::minimize` checks the flag
			m_flags &= ~core::bitflag(ECF_MINIMIZED);
	}
	if (maximized!=m_maximizedState)
	{
		m_maximizedState = maximized;
		if (maximized)
		{
			if (m_cb)
				(void)m_cb->onWindowMaximized(this);
		}
		else
			m_flags &= ~core::bitflag(ECF_MAXIMIZED);
	}
}
#endif
