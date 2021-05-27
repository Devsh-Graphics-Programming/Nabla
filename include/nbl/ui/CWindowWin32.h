#ifndef __C_WINDOW_WIN32_H_INCLUDED__
#define __C_WINDOW_WIN32_H_INCLUDED__

#include "nbl/ui/IWindowWin32.h"
#include "os.h"

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
{

class CWindowWin32 final : public IWindowWin32
{
	// TODO design for event listener etc, WndProc will somehow pass events to provided event listener
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

public:
	//TODO Do we even want to have this ctor now since the window should be created in a separate thread??
	explicit CWindowWin32(core::smart_refctd_ptr<system::ISystem>&& sys, native_handle_t hwnd) : IWindowWin32(std::move(sys))
	{
		RECT rect;
		GetWindowRect(hwnd, &rect);

		m_width = rect.right - rect.left;
		m_height = rect.bottom - rect.top;


		// TODO m_flags
	}

	native_handle_t getNativeHandle() const override { return m_threadHandler.getNativeWindow(); }

	static core::smart_refctd_ptr<CWindowWin32> create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		if ((_flags & (ECF_MINIMIZED | ECF_MAXIMIZED)) == (ECF_MINIMIZED | ECF_MAXIMIZED))
			return nullptr;

		CWindowWin32* win = new CWindowWin32(std::move(sys), _w, _h, _flags);
		return core::smart_refctd_ptr<CWindowWin32>(win, core::dont_grab);
	}

private:
    CWindowWin32(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags) : 
		IWindowWin32(std::move(sys), _w, _h, _flags), m_threadHandler(_w, _h, _flags)
    {
		m_width = _w; m_height = _h;
		SetWindowLongPtr(m_threadHandler.getNativeWindow(), GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
    }


    //native_handle_t m_native;
private:
	class CThreadHandler final : public system::IThreadHandler<CThreadHandler>
	{
		using base_t = system::IThreadHandler<CThreadHandler>;
		friend base_t;
	public:
		CThreadHandler(uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
		{
			HINSTANCE hinstance = GetModuleHandle(NULL);

			const char* classname = __TEXT("Nabla Engine");

			WNDCLASSEX wcex;
			wcex.cbSize = sizeof(WNDCLASSEX);
			wcex.style = CS_HREDRAW | CS_VREDRAW;
			wcex.lpfnWndProc = WndProc;
			wcex.cbClsExtra = 0;
			wcex.cbWndExtra = 0;
			wcex.hInstance = hinstance;
			wcex.hIcon = NULL;
			wcex.hCursor = 0; // LoadCursor(NULL, IDC_ARROW);
			wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
			wcex.lpszMenuName = 0;
			wcex.lpszClassName = classname;
			wcex.hIconSm = 0;

			RegisterClassEx(&wcex);

			// calculate client size

			RECT clientSize;
			clientSize.top = 0;
			clientSize.left = 0;
			clientSize.right = _w;
			clientSize.bottom = _h;

			DWORD style = WS_POPUP; // TODO why popup?

			if ((_flags & ECF_FULLSCREEN) == 0)
			{
				if ((_flags & ECF_BORDERLESS) == 0)
				{
					style |= WS_BORDER;
					style |= (WS_SYSMENU | WS_CAPTION);
				}
				// ? not sure about those below
				style |= WS_CLIPCHILDREN;
				style |= WS_CLIPSIBLINGS;
			}
			if (_flags & ECF_MINIMIZED)
			{
				style |= WS_MINIMIZE;
			}
			if (_flags & ECF_MAXIMIZED)
			{
				style |= WS_MAXIMIZE;
			}
			if (_flags & ECF_ALWAYS_ON_TOP)
			{
				style |= WS_EX_TOPMOST;
			}
			if ((_flags & ECF_HIDDEN) == 0)
			{
				style |= WS_VISIBLE;
			}


			// TODO:
			// if (hasMouseCaptured())
			// if (hasInputFocus())
			// if (hasMouseFocus())

			AdjustWindowRect(&clientSize, style, FALSE);

			const int32_t realWidth = clientSize.right - clientSize.left;
			const int32_t realHeight = clientSize.bottom - clientSize.top;

			int32_t windowLeft = (GetSystemMetrics(SM_CXSCREEN) - realWidth) / 2;
			int32_t windowTop = (GetSystemMetrics(SM_CYSCREEN) - realHeight) / 2;

			if (windowLeft < 0)
				windowLeft = 0;
			if (windowTop < 0)
				windowTop = 0;	// make sure window menus are in screen on creation

			if (_flags & ECF_FULLSCREEN)
			{
				windowLeft = 0;
				windowTop = 0;
			}

			// create window

			 nativeWindow = CreateWindow(classname, __TEXT(""), style, windowLeft, windowTop,
				realWidth, realHeight, NULL, NULL, hinstance, NULL);
			if ((_flags & ECF_HIDDEN) == 0)
				ShowWindow(nativeWindow, SW_SHOWNORMAL);
			UpdateWindow(nativeWindow);

			// fix ugly ATI driver bugs. Thanks to ariaci
			// TODO still needed?
			MoveWindow(nativeWindow, windowLeft, windowTop, realWidth, realHeight, TRUE);
		}

		native_handle_t getNativeWindow() const
		{
			return nativeWindow;
		}
	private:
		void init()
		{
			// All the stuff was already initialized in the ctor
		}
		void exit()
		{
			DestroyWindow(nativeWindow);
		}
		void work(base_t::lock_t& lock)
		{
			// @criss not sure why PeekMessage is better than GetMessage, since we'll be spining all the time here
			// Is a spinlock a better solution than blocking a thread?
			keepRunning = GetMessage(&message, nativeWindow, 0, 0);
			TranslateMessage(&message);
			DispatchMessage(&message);
		}
		bool wakeupPredicate() const { return keepRunning; }
		bool continuePredicate() const { return keepRunning; }
	private:
		bool keepRunning = true;
		MSG message;
		native_handle_t nativeWindow = nullptr;
	};

private:
	CThreadHandler m_threadHandler;
};

}
}

#endif

#endif
