#include "nbl/ui/CWindowManagerWin32.h"
#include "nbl/ui/CWindowWin32.h"

#ifdef _NBL_PLATFORM_WINDOWS_
#include <codecvt>
#include <xlocbuf>
#include <hidusage.h>

using namespace nbl;
using namespace nbl::ui;

core::smart_refctd_ptr<IWindowManagerWin32> IWindowManagerWin32::create()
{
	return core::make_smart_refctd_ptr<CWindowManagerWin32>();
}

IWindowManager::SDisplayInfo CWindowManagerWin32::getPrimaryDisplayInfo() const
{
	RECT size;
	BOOL res_ok = SystemParametersInfo(SPI_GETWORKAREA, 0, &size, 0);
	SDisplayInfo info{};
	info.resX = size.right - size.left;
	info.resY = size.bottom - size.top;
	info.x = size.left; // When would this be not 0 though??
	info.y = size.top;
	return info;
}

static inline DWORD getWindowStyle(const core::bitflag<IWindow::E_CREATE_FLAGS> flags)
{
	DWORD style = WS_POPUP;

	if (!flags.hasFlags(IWindow::ECF_FULLSCREEN))
	{
		if (!flags.hasFlags(IWindow::ECF_BORDERLESS))
		{
			style |= WS_BORDER;
			style |= (WS_SYSMENU | WS_CAPTION);
		}
		// ? not sure about those below
		style |= WS_CLIPCHILDREN;
		style |= WS_CLIPSIBLINGS;
	}
	if (flags.hasFlags(IWindow::ECF_MINIMIZED))
	{
		style |= WS_MINIMIZE;
	}
	if (flags.hasFlags(IWindow::ECF_MAXIMIZED))
	{
		style |= WS_MAXIMIZE;
	}
	if (flags.hasFlags(IWindow::ECF_ALWAYS_ON_TOP))
	{
		style |= WS_EX_TOPMOST;
	}
	if (!flags.hasFlags(IWindow::ECF_HIDDEN))
	{
		style |= WS_VISIBLE;
	}
	style |= WS_OVERLAPPEDWINDOW;
	if (!flags.hasFlags(IWindow::ECF_CAN_RESIZE))
	{
		style &= ~WS_SIZEBOX;
	}
	if (!flags.hasFlags(IWindow::ECF_CAN_MAXIMIZE))
	{
		style &= ~WS_MAXIMIZEBOX;
	}
	if (!flags.hasFlags(IWindow::ECF_CAN_MINIMIZE))
	{
		style &= ~WS_MINIMIZEBOX;
	}

	return style;
}

core::smart_refctd_ptr<IWindow> CWindowManagerWin32::createWindow(IWindow::SCreationParams&& creationParams)
{
	// this could be common to all `createWindow` impl
	if (creationParams.flags.hasFlags(IWindow::ECF_CAN_RESIZE) || creationParams.flags.hasFlags(IWindow::ECF_CAN_MAXIMIZE))
		creationParams.flags |= IWindow::ECF_RESIZABLE;
	// win32 minimize is weird, its a resize to 0,0
	if (creationParams.flags.hasFlags(IWindow::ECF_CAN_MINIMIZE))
		creationParams.flags |= IWindow::ECF_CAN_RESIZE;

	CAsyncQueue::future_t<IWindowWin32::native_handle_t> future;
	m_windowThreadManager.request(&future, SRequestParams_CreateWindow{
		.windowCaption = std::move(creationParams.windowCaption),
		.width = creationParams.width,
		.height = creationParams.height,
		.x = creationParams.x,
		.y = creationParams.y,
		.flags = creationParams.flags
	});
	if (auto handle = future.acquire())
		return core::make_smart_refctd_ptr<CWindowWin32>(std::move(creationParams),core::smart_refctd_ptr<CWindowManagerWin32>(this),*handle);
	return nullptr;
}

bool CWindowManagerWin32::setWindowSize_impl(IWindow* window, const uint32_t width, const uint32_t height)
{
	// Calculate real window size based on client size
	RECT clientSize;
	clientSize.left = 0;
	clientSize.top = 0;
	clientSize.right = width;
	clientSize.bottom = height;

	const DWORD style = getWindowStyle(window->getFlags().value);
	bool res = AdjustWindowRect(&clientSize, style, false);
	assert(res);

	CAsyncQueue::future_t<void> future;
	m_windowThreadManager.request(&future,SRequestParams_SetWindowSize{
		.nativeWindow = static_cast<IWindowWin32*>(window)->getNativeHandle(),
		.width = clientSize.right-clientSize.left,
		.height = clientSize.bottom-clientSize.top
	});
	return true;
}

void CWindowManagerWin32::CAsyncQueue::background_work()
{
	MSG msg;
	{
		constexpr uint32_t timeoutInMS = 8; // gonna become 10 anyway	
		const UINT_PTR timerId = SetTimer(NULL, NULL, timeoutInMS, NULL);
		const bool result = GetMessage(&msg, nullptr, 0, 0);

		KillTimer(NULL, timerId);

		if (!result || (msg.message == WM_TIMER && msg.hwnd == NULL && msg.wParam == timerId))
			return;
	}
	TranslateMessage(&msg);
	DispatchMessage(&msg);
}

void CWindowManagerWin32::CAsyncQueue::process_request(base_t::future_base_t* _future_base, SRequest& req)
{
	std::visit([=](auto& visitor) {
		using retval_t = std::remove_reference_t<decltype(visitor)>::retval_t;
		visitor(base_t::future_storage_cast<retval_t>(_future_base));
	}, req.params);
}

void CWindowManagerWin32::SRequestParams_CreateWindow::operator()(core::StorageTrivializer<retval_t>* retval)
{
	HINSTANCE hinstance = GetModuleHandle(NULL);

	const char* classname = "Nabla Engine";

	WNDCLASSEXA wcex;
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = CWindowWin32::WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hinstance;
	wcex.hIcon = NULL;
	wcex.hCursor = nullptr;
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = 0;
	wcex.lpszClassName = classname;
	wcex.hIconSm = 0;

	RegisterClassExA(&wcex);
	// calculate client size

	RECT clientSize;
	clientSize.top = y;
	clientSize.left = x;
	clientSize.right = clientSize.left + width;
	clientSize.bottom = clientSize.top + height;

	const DWORD style = getWindowStyle(flags);

	// TODO:
	// if (hasMouseCaptured())
	// if (hasInputFocus())
	// if (hasMouseFocus())

	AdjustWindowRect(&clientSize, style, FALSE);

	const int32_t realWidth = clientSize.right - clientSize.left;
	const int32_t realHeight = clientSize.bottom - clientSize.top;

					
	auto nativeWindow = CreateWindowA(
		classname, windowCaption.c_str(), style,
		clientSize.left, clientSize.top, realWidth, realHeight,
		NULL, NULL, hinstance, NULL
	);

	//
	if (!flags.hasFlags(CWindowWin32::ECF_HIDDEN))
		ShowWindow(nativeWindow, SW_SHOWNORMAL);
	UpdateWindow(nativeWindow);

	// fix ugly ATI driver bugs. Thanks to ariaci
	// TODO still needed?
	MoveWindow(nativeWindow, clientSize.left, clientSize.top, realWidth, realHeight, TRUE);

	{
		//TODO: thoroughly test this stuff	(what is this about, you need to register devices yourself!? I thought Windows can give you a list of raw input devices!?)
		constexpr uint32_t INPUT_DEVICES_COUNT = 5;
		RAWINPUTDEVICE inputDevices[INPUT_DEVICES_COUNT];
		inputDevices[0].hwndTarget = nativeWindow;
		inputDevices[0].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
		inputDevices[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
		inputDevices[0].usUsage = HID_USAGE_GENERIC_POINTER;

		inputDevices[1].hwndTarget = nativeWindow;
		inputDevices[1].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
		inputDevices[1].usUsagePage = HID_USAGE_PAGE_GENERIC;
		inputDevices[1].usUsage = HID_USAGE_GENERIC_MOUSE;

		inputDevices[2].hwndTarget = nativeWindow;
		inputDevices[2].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
		inputDevices[2].usUsagePage = HID_USAGE_PAGE_GENERIC;
		inputDevices[2].usUsage = HID_USAGE_GENERIC_KEYBOARD;

		inputDevices[3].hwndTarget = nativeWindow;
		inputDevices[3].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
		inputDevices[3].usUsagePage = HID_USAGE_PAGE_GAME;
		inputDevices[3].usUsage = HID_USAGE_GENERIC_JOYSTICK;

		inputDevices[4].hwndTarget = nativeWindow;
		inputDevices[4].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
		inputDevices[4].usUsagePage = HID_USAGE_PAGE_GAME;
		inputDevices[4].usUsage = HID_USAGE_GENERIC_GAMEPAD;

		RegisterRawInputDevices(inputDevices, INPUT_DEVICES_COUNT, sizeof(RAWINPUTDEVICE));
	}

	retval->construct(nativeWindow);
}
void CWindowManagerWin32::SRequestParams_ChangeCursorVisibility::operator()(core::StorageTrivializer<retval_t>* retval)
{
	if (visible)
	{
		int ret = ShowCursor(true);
		while (ret < 0)  ret = ShowCursor(true);
	}
	else
	{
		int ret = ShowCursor(false);
		while (ret >= 0)  ret = ShowCursor(false);
	}
}
void CWindowManagerWin32::SRequestParams_ShowWindow::operator()(core::StorageTrivializer<retval_t>* retval)
{
	const static int showCmd[] = {SW_HIDE,SW_SHOW,SW_MINIMIZE,SW_MAXIMIZE};
	ShowWindow(nativeWindow,showCmd[static_cast<uint8_t>(state)]);
}
#endif