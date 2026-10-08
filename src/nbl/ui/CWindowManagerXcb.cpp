#include "nbl/ui/CWindowManagerXcb.h"
#include "nbl/ui/CWindowXcb.h"

#ifdef _NBL_PLATFORM_LINUX_
#include <poll.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

using namespace nbl;
using namespace nbl::ui;

namespace
{
// ICCCM and Motif constants, we don't depend on xcb-icccm for a handful of them
constexpr uint32_t WM_STATE_NORMAL = 1u;
constexpr uint32_t WM_STATE_ICONIC = 3u;

constexpr uint32_t WM_HINTS_INPUT = 1u<<0;
constexpr uint32_t WM_HINTS_STATE = 1u<<1;

constexpr uint32_t WM_SIZE_HINT_US_POSITION = 1u<<0;
constexpr uint32_t WM_SIZE_HINT_P_POSITION = 1u<<2;
constexpr uint32_t WM_SIZE_HINT_P_MIN_SIZE = 1u<<4;
constexpr uint32_t WM_SIZE_HINT_P_MAX_SIZE = 1u<<5;

constexpr uint32_t MWM_HINTS_FUNCTIONS = 1u<<0;
constexpr uint32_t MWM_HINTS_DECORATIONS = 1u<<1;
constexpr uint32_t MWM_FUNC_RESIZE = 1u<<1;
constexpr uint32_t MWM_FUNC_MOVE = 1u<<2;
constexpr uint32_t MWM_FUNC_MINIMIZE = 1u<<3;
constexpr uint32_t MWM_FUNC_MAXIMIZE = 1u<<4;
constexpr uint32_t MWM_FUNC_CLOSE = 1u<<5;

constexpr uint32_t NET_WM_STATE_ADD = 1u;
constexpr uint32_t NET_WM_SOURCE_APPLICATION = 1u;

// the timeout also bounds how long a create/destroy request can wait for the event thread, same as the Win32 timer
constexpr int EventPollTimeoutMS = 8;
}

core::smart_refctd_ptr<IWindowManagerXcb> IWindowManagerXcb::create()
{
	int screenIx = 0;
	xcb_connection_t* connection = xcb_connect(nullptr,&screenIx);
	if (xcb_connection_has_error(connection))
	{
		xcb_disconnect(connection);
		return nullptr;
	}

	xcb_screen_t* screen = nullptr;
	for (auto it=xcb_setup_roots_iterator(xcb_get_setup(connection)); it.rem; screenIx--,xcb_screen_next(&it))
	if (screenIx==0)
	{
		screen = it.data;
		break;
	}
	if (!screen)
	{
		xcb_disconnect(connection);
		return nullptr;
	}

	CWindowManagerXcb::SAtoms atoms;
	{
		struct SAtomRequest
		{
			xcb_atom_t* dst;
			const char* name;
			xcb_intern_atom_cookie_t cookie;
		};
		#define NBL_XCB_ATOM(NAME) SAtomRequest{&atoms.NAME,#NAME,{}}
		SAtomRequest requests[] = {
			NBL_XCB_ATOM(WM_PROTOCOLS),
			NBL_XCB_ATOM(WM_DELETE_WINDOW),
			NBL_XCB_ATOM(WM_STATE),
			NBL_XCB_ATOM(WM_CHANGE_STATE),
			NBL_XCB_ATOM(UTF8_STRING),
			NBL_XCB_ATOM(_NET_WM_NAME),
			NBL_XCB_ATOM(_NET_WM_STATE),
			NBL_XCB_ATOM(_NET_WM_STATE_HIDDEN),
			NBL_XCB_ATOM(_NET_WM_STATE_MAXIMIZED_VERT),
			NBL_XCB_ATOM(_NET_WM_STATE_MAXIMIZED_HORZ),
			NBL_XCB_ATOM(_NET_WM_STATE_FULLSCREEN),
			NBL_XCB_ATOM(_NET_WM_STATE_ABOVE),
			NBL_XCB_ATOM(_MOTIF_WM_HINTS)
		};
		#undef NBL_XCB_ATOM
		// send all the requests before waiting for any reply
		for (auto& request : requests)
			request.cookie = xcb_intern_atom(connection,0,strlen(request.name),request.name);
		bool success = true;
		for (auto& request : requests)
		{
			xcb_intern_atom_reply_t* reply = xcb_intern_atom_reply(connection,request.cookie,nullptr);
			success = success && reply;
			*request.dst = reply ? reply->atom:XCB_ATOM_NONE;
			free(reply);
		}
		if (!success)
		{
			xcb_disconnect(connection);
			return nullptr;
		}
	}

	return core::make_smart_refctd_ptr<CWindowManagerXcb>(connection,screen,atoms);
}

CWindowManagerXcb::CWindowManagerXcb(xcb_connection_t* connection, xcb_screen_t* screen, const SAtoms& atoms) :
	m_connection{connection}, m_screen(screen), m_atoms(atoms), m_eventThread(this)
{
}

IWindowManager::SDisplayInfo CWindowManagerXcb::getPrimaryDisplayInfo() const
{
	// TODO: use RandR to get the primary output, the X screen spans all the monitors
	SDisplayInfo info{};
	info.x = 0;
	info.y = 0;
	info.resX = m_screen->width_in_pixels;
	info.resY = m_screen->height_in_pixels;
	info.name = "X11 screen";
	return info;
}

core::smart_refctd_ptr<IWindow> CWindowManagerXcb::createWindow(IWindow::SCreationParams&& creationParams)
{
	// same normalization as Win32
	if (creationParams.flags.hasFlags(IWindow::ECF_CAN_RESIZE) || creationParams.flags.hasFlags(IWindow::ECF_CAN_MAXIMIZE))
		creationParams.flags |= IWindow::ECF_RESIZABLE;
	// X11 rejects zero sized windows
	creationParams.width = std::max(creationParams.width,1u);
	creationParams.height = std::max(creationParams.height,1u);

	const bool map = !creationParams.flags.hasFlags(IWindow::ECF_HIDDEN);
	// Called from a callback, the event thread can't wait on its own queue, but it already owns the window table
	if (m_eventThread.isEventThread())
	{
		const xcb_window_t nativeWindow = createNativeWindow(creationParams);
		if (nativeWindow==XCB_WINDOW_NONE)
			return nullptr;
		auto window = core::make_smart_refctd_ptr<CWindowXcb>(std::move(creationParams),core::smart_refctd_ptr<CWindowManagerXcb>(this),nativeWindow);
		registerWindow(window.get(),map);
		return window;
	}

	CAsyncQueue::future_t<xcb_window_t> future;
	m_eventThread.request(&future,SRequestParams_CreateWindow{.params=&creationParams});
	auto nativeWindow = future.acquire();
	if (!nativeWindow || *nativeWindow==XCB_WINDOW_NONE)
		return nullptr;

	auto window = core::make_smart_refctd_ptr<CWindowXcb>(std::move(creationParams),core::smart_refctd_ptr<CWindowManagerXcb>(this),*nativeWindow);
	// the window must be known to the event thread before it's mapped, otherwise we'd lose the first events
	CAsyncQueue::future_t<void> registered;
	m_eventThread.request(&registered,SRequestParams_RegisterWindow{.window=window.get(),.map=map});
	registered.wait();
	return window;
}

void CWindowManagerXcb::destroyWindow(IWindow* wnd)
{
	const auto nativeWindow = static_cast<CWindowXcb*>(wnd)->getNativeHandle().window;
	// the last reference to some other window got dropped by a callback
	if (m_eventThread.isEventThread())
	{
		// The window whose event is being dispatched can't die here, `IWindow::IEventCallback` still writes its flags after the callback returns
		assert(wnd!=m_dispatchingWindow);
		unregisterWindow(nativeWindow);
		return;
	}
	CAsyncQueue::future_t<void> future;
	m_eventThread.request(&future,SRequestParams_DestroyWindow{.window=nativeWindow});
	future.wait();
}

void CWindowManagerXcb::setSizeHints(xcb_window_t window, int32_t x, int32_t y, uint32_t width, uint32_t height, bool fixedSize) const
{
	// ICCCM `WM_SIZE_HINTS`: flags, 4 obsolete fields, min, max, increments, min/max aspect, base size, gravity
	uint32_t hints[18] = {};
	hints[0] = WM_SIZE_HINT_US_POSITION|WM_SIZE_HINT_P_POSITION;
	hints[1] = static_cast<uint32_t>(x);
	hints[2] = static_cast<uint32_t>(y);
	if (fixedSize)
	{
		hints[0] |= WM_SIZE_HINT_P_MIN_SIZE|WM_SIZE_HINT_P_MAX_SIZE;
		hints[5] = hints[7] = width;
		hints[6] = hints[8] = height;
	}
	xcb_change_property(m_connection.handle,XCB_PROP_MODE_REPLACE,window,XCB_ATOM_WM_NORMAL_HINTS,XCB_ATOM_WM_SIZE_HINTS,32,18,hints);
}

bool CWindowManagerXcb::setWindowSize_impl(IWindow* window, const uint32_t width, const uint32_t height)
{
	const auto nativeWindow = static_cast<CWindowXcb*>(window)->getNativeHandle().window;
	// a window the user can't resize has its min and max size locked, move both
	if (!window->isResizable())
		setSizeHints(nativeWindow,window->getX(),window->getY(),width,height,true);
	const uint32_t values[2] = {std::max(width,1u),std::max(height,1u)};
	xcb_configure_window(m_connection.handle,nativeWindow,XCB_CONFIG_WINDOW_WIDTH|XCB_CONFIG_WINDOW_HEIGHT,values);
	xcb_flush(m_connection.handle);
	return true;
}

bool CWindowManagerXcb::setWindowPosition_impl(IWindow* window, const int32_t x, const int32_t y)
{
	const uint32_t values[2] = {static_cast<uint32_t>(x),static_cast<uint32_t>(y)};
	xcb_configure_window(m_connection.handle,static_cast<CWindowXcb*>(window)->getNativeHandle().window,XCB_CONFIG_WINDOW_X|XCB_CONFIG_WINDOW_Y,values);
	xcb_flush(m_connection.handle);
	return true;
}

bool CWindowManagerXcb::setWindowVisible_impl(IWindow* window, const bool visible)
{
	const auto nativeWindow = static_cast<CWindowXcb*>(window)->getNativeHandle().window;
	if (visible)
		xcb_map_window(m_connection.handle,nativeWindow);
	else
		xcb_unmap_window(m_connection.handle,nativeWindow);
	xcb_flush(m_connection.handle);
	return true;
}

bool CWindowManagerXcb::setWindowMaximized_impl(IWindow* window, const bool maximized)
{
	const auto nativeWindow = static_cast<CWindowXcb*>(window)->getNativeHandle().window;
	if (maximized)
		sendToRoot(nativeWindow,m_atoms._NET_WM_STATE,{NET_WM_STATE_ADD,m_atoms._NET_WM_STATE_MAXIMIZED_VERT,m_atoms._NET_WM_STATE_MAXIMIZED_HORZ,NET_WM_SOURCE_APPLICATION,0});
	else // `IWindowManager::minimize` lands here
		sendToRoot(nativeWindow,m_atoms.WM_CHANGE_STATE,{WM_STATE_ICONIC,0,0,0,0});
	xcb_flush(m_connection.handle);
	return true;
}

void CWindowManagerXcb::sendToRoot(xcb_window_t window, xcb_atom_t type, const uint32_t (&data)[5]) const
{
	// `xcb_send_event` always copies 32 bytes
	xcb_client_message_event_t ev = {};
	ev.response_type = XCB_CLIENT_MESSAGE;
	ev.format = 32;
	ev.window = window;
	ev.type = type;
	std::copy_n(data,5,ev.data.data32);
	xcb_send_event(m_connection.handle,0,m_screen->root,XCB_EVENT_MASK_SUBSTRUCTURE_REDIRECT|XCB_EVENT_MASK_SUBSTRUCTURE_NOTIFY,reinterpret_cast<const char*>(&ev));
}

xcb_window_t CWindowManagerXcb::createNativeWindow(const IWindow::SCreationParams& params)
{
	auto* const connection = m_connection.handle;
	const auto flags = params.flags;

	const xcb_window_t window = xcb_generate_id(connection);
	{
		// no background, so the server doesn't clear what Vulkan presented while resizing
		const uint32_t valueMask = XCB_CW_BACK_PIXMAP|XCB_CW_EVENT_MASK;
		const uint32_t values[2] = {
			XCB_BACK_PIXMAP_NONE,
			XCB_EVENT_MASK_STRUCTURE_NOTIFY|XCB_EVENT_MASK_PROPERTY_CHANGE|XCB_EVENT_MASK_FOCUS_CHANGE|XCB_EVENT_MASK_ENTER_WINDOW|XCB_EVENT_MASK_LEAVE_WINDOW|XCB_EVENT_MASK_EXPOSURE
		};
		const auto cookie = xcb_create_window_checked(
			connection,XCB_COPY_FROM_PARENT,window,m_screen->root,
			static_cast<int16_t>(params.x),static_cast<int16_t>(params.y),static_cast<uint16_t>(params.width),static_cast<uint16_t>(params.height),
			0,XCB_WINDOW_CLASS_INPUT_OUTPUT,m_screen->root_visual,valueMask,values
		);
		if (xcb_generic_error_t* error=xcb_request_check(connection,cookie))
		{
			free(error);
			return XCB_WINDOW_NONE;
		}
	}

	// ask to be told about the close button instead of getting killed
	xcb_change_property(connection,XCB_PROP_MODE_REPLACE,window,m_atoms.WM_PROTOCOLS,XCB_ATOM_ATOM,32,1,&m_atoms.WM_DELETE_WINDOW);
	{
		constexpr char wmClass[] = "nabla\0Nabla";
		xcb_change_property(connection,XCB_PROP_MODE_REPLACE,window,XCB_ATOM_WM_CLASS,XCB_ATOM_STRING,8,sizeof(wmClass),wmClass);
	}
	xcb_change_property(connection,XCB_PROP_MODE_REPLACE,window,XCB_ATOM_WM_NAME,XCB_ATOM_STRING,8,params.windowCaption.size(),params.windowCaption.data());
	xcb_change_property(connection,XCB_PROP_MODE_REPLACE,window,m_atoms._NET_WM_NAME,m_atoms.UTF8_STRING,8,params.windowCaption.size(),params.windowCaption.data());

	// window managers ignore the position of `xcb_create_window` unless we insist
	setSizeHints(window,params.x,params.y,params.width,params.height,!flags.hasFlags(IWindow::ECF_CAN_RESIZE));
	{
		// ICCCM `WM_HINTS`: flags, input, initial_state, icon pixmap, icon window, icon x/y, icon mask, window group
		uint32_t hints[9] = {};
		hints[0] = WM_HINTS_INPUT|WM_HINTS_STATE;
		hints[1] = 1u;
		hints[2] = flags.hasFlags(IWindow::ECF_MINIMIZED) ? WM_STATE_ICONIC:WM_STATE_NORMAL;
		xcb_change_property(connection,XCB_PROP_MODE_REPLACE,window,XCB_ATOM_WM_HINTS,XCB_ATOM_WM_HINTS,32,9,hints);
	}
	{
		// Motif hints are what X11 window managers still honour for decorations and the title bar buttons
		uint32_t functions = MWM_FUNC_MOVE|MWM_FUNC_CLOSE;
		if (flags.hasFlags(IWindow::ECF_CAN_RESIZE))
			functions |= MWM_FUNC_RESIZE;
		if (flags.hasFlags(IWindow::ECF_CAN_MINIMIZE))
			functions |= MWM_FUNC_MINIMIZE;
		if (flags.hasFlags(IWindow::ECF_CAN_MAXIMIZE))
			functions |= MWM_FUNC_MAXIMIZE;
		const bool decorated = !flags.hasFlags(IWindow::ECF_BORDERLESS) && !flags.hasFlags(IWindow::ECF_FULLSCREEN);
		// flags, functions, decorations, input mode, status
		const uint32_t hints[5] = {MWM_HINTS_FUNCTIONS|MWM_HINTS_DECORATIONS,functions,decorated ? 1u:0u,0u,0u};
		xcb_change_property(connection,XCB_PROP_MODE_REPLACE,window,m_atoms._MOTIF_WM_HINTS,m_atoms._MOTIF_WM_HINTS,32,5,hints);
	}
	{
		// EWMH initial state, only read by the window manager when the window gets mapped
		core::vector<xcb_atom_t> state;
		if (flags.hasFlags(IWindow::ECF_FULLSCREEN))
			state.push_back(m_atoms._NET_WM_STATE_FULLSCREEN);
		if (flags.hasFlags(IWindow::ECF_ALWAYS_ON_TOP))
			state.push_back(m_atoms._NET_WM_STATE_ABOVE);
		if (flags.hasFlags(IWindow::ECF_MAXIMIZED))
		{
			state.push_back(m_atoms._NET_WM_STATE_MAXIMIZED_VERT);
			state.push_back(m_atoms._NET_WM_STATE_MAXIMIZED_HORZ);
		}
		if (!state.empty())
			xcb_change_property(connection,XCB_PROP_MODE_REPLACE,window,m_atoms._NET_WM_STATE,XCB_ATOM_ATOM,32,state.size(),state.data());
	}
	xcb_flush(connection);
	return window;
}

void CWindowManagerXcb::registerWindow(CWindowXcb* window, const bool map)
{
	const auto nativeWindow = window->getNativeHandle().window;
	m_windows[nativeWindow] = window;
	if (map)
	{
		xcb_map_window(m_connection.handle,nativeWindow);
		xcb_flush(m_connection.handle);
	}
}

void CWindowManagerXcb::unregisterWindow(xcb_window_t window)
{
	m_windows.erase(window);
	xcb_destroy_window(m_connection.handle,window);
	xcb_flush(m_connection.handle);
}

CWindowXcb* CWindowManagerXcb::findWindow(xcb_window_t window) const
{
	auto found = m_windows.find(window);
	return found!=m_windows.end() ? found->second:nullptr;
}

void CWindowManagerXcb::dispatchEvent(const xcb_generic_event_t* event)
{
	// the top bit flags events sent with `SendEvent`, e.g. by the window manager
	switch (event->response_type&0x7fu)
	{
		case XCB_CLIENT_MESSAGE:
		{
			const auto* ev = reinterpret_cast<const xcb_client_message_event_t*>(event);
			if (ev->type==m_atoms.WM_PROTOCOLS && ev->format==32 && ev->data.data32[0]==m_atoms.WM_DELETE_WINDOW)
			if (auto* window=beginDispatch(ev->window))
				window->onDeleteRequested();
			break;
		}
		case XCB_CONFIGURE_NOTIFY:
		{
			const auto* ev = reinterpret_cast<const xcb_configure_notify_event_t*>(event);
			if (auto* window=beginDispatch(ev->window))
				window->onConfigured(ev);
			break;
		}
		case XCB_MAP_NOTIFY:
		{
			const auto* ev = reinterpret_cast<const xcb_map_notify_event_t*>(event);
			if (auto* window=beginDispatch(ev->window))
				window->onMapped();
			break;
		}
		case XCB_UNMAP_NOTIFY:
		{
			const auto* ev = reinterpret_cast<const xcb_unmap_notify_event_t*>(event);
			if (auto* window=beginDispatch(ev->window))
				window->onUnmapped();
			break;
		}
		case XCB_PROPERTY_NOTIFY:
		{
			const auto* ev = reinterpret_cast<const xcb_property_notify_event_t*>(event);
			if (ev->atom==m_atoms._NET_WM_STATE || ev->atom==m_atoms.WM_STATE)
			if (auto* window=beginDispatch(ev->window))
				window->updateState();
			break;
		}
		case XCB_FOCUS_IN: [[fallthrough]];
		case XCB_FOCUS_OUT:
		{
			const auto* ev = reinterpret_cast<const xcb_focus_in_event_t*>(event);
			// pointer focus details are about the window under the pointer, not ours
			if (ev->detail!=XCB_NOTIFY_DETAIL_POINTER)
			if (auto* window=beginDispatch(ev->event))
				window->onFocusChanged(true,(event->response_type&0x7fu)==XCB_FOCUS_IN);
			break;
		}
		case XCB_ENTER_NOTIFY: [[fallthrough]];
		case XCB_LEAVE_NOTIFY:
		{
			const auto* ev = reinterpret_cast<const xcb_enter_notify_event_t*>(event);
			if (ev->detail!=XCB_NOTIFY_DETAIL_INFERIOR)
			if (auto* window=beginDispatch(ev->event))
				window->onFocusChanged(false,(event->response_type&0x7fu)==XCB_ENTER_NOTIFY);
			break;
		}
		default:
			// errors (response type 0) of unchecked requests also land here, nothing we can do about them
			break;
	}
}

void CWindowManagerXcb::CAsyncQueue::background_work()
{
	auto* const connection = m_manager->m_connection.handle;
	// a dead connection would make `poll` return immediately forever
	if (xcb_connection_has_error(connection))
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(EventPollTimeoutMS));
		return;
	}

	const auto dispatchAll = [&]() -> bool
	{
		bool any = false;
		// also returns events which got queued while we waited on a reply
		while (xcb_generic_event_t* event=xcb_poll_for_event(connection))
		{
			m_manager->dispatchEvent(event);
			m_manager->m_dispatchingWindow = nullptr;
			free(event);
			any = true;
		}
		return any;
	};
	if (dispatchAll())
		return;

	pollfd pfd = {};
	pfd.fd = xcb_get_file_descriptor(connection);
	pfd.events = POLLIN;
	if (poll(&pfd,1,EventPollTimeoutMS)>0)
		dispatchAll();
}

void CWindowManagerXcb::CAsyncQueue::process_request(base_t::future_base_t* _future_base, SRequest& req)
{
	std::visit([&](auto& params)->void
	{
		using params_t = std::remove_reference_t<decltype(params)>;
		using retval_t = typename params_t::retval_t;
		auto* retval = base_t::future_storage_cast<retval_t>(_future_base);
		if constexpr (std::is_same_v<params_t,SRequestParams_CreateWindow>)
			retval->construct(m_manager->createNativeWindow(*params.params));
		else if constexpr (std::is_same_v<params_t,SRequestParams_RegisterWindow>)
			m_manager->registerWindow(params.window,params.map);
		else if constexpr (std::is_same_v<params_t,SRequestParams_DestroyWindow>)
			m_manager->unregisterWindow(params.window);
		else
			assert(false);
	},req.params);
}
#endif
