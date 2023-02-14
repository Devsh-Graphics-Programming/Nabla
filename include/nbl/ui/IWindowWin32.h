#ifndef __NBL_I_WINDOW_WIN32_H_INCLUDED__
#define __NBL_I_WINDOW_WIN32_H_INCLUDED__

#include "nbl/system/DefaultFuncPtrLoader.h"

#include "nbl/ui/IWindow.h"


#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl::ui
{

class NBL_API2 IWindowWin32 : public IWindow
{
    protected:
        virtual ~IWindowWin32() = default;
        inline IWindowWin32(SCreationParams&& params) : IWindow(std::move(params)) {}

    public:
        using IWindow::IWindow;

        using native_handle_t = HWND;

        virtual const native_handle_t& getNativeHandle() const = 0;

		static DWORD getWindowStyle(IWindow::E_CREATE_FLAGS flags)
		{
			DWORD style = WS_POPUP;

			if ((flags & IWindow::ECF_FULLSCREEN) == 0)
			{
				if ((flags & IWindow::ECF_BORDERLESS) == 0)
				{
					style |= WS_BORDER;
					style |= (WS_SYSMENU | WS_CAPTION);
				}
				// ? not sure about those below
				style |= WS_CLIPCHILDREN;
				style |= WS_CLIPSIBLINGS;
			}
			if (flags & IWindow::ECF_MINIMIZED)
			{
				style |= WS_MINIMIZE;
			}
			if (flags & IWindow::ECF_MAXIMIZED)
			{
				style |= WS_MAXIMIZE;
			}
			if (flags & IWindow::ECF_ALWAYS_ON_TOP)
			{
				style |= WS_EX_TOPMOST;
			}
			if ((flags & IWindow::ECF_HIDDEN) == 0)
			{
				style |= WS_VISIBLE;
			}
			style |= WS_OVERLAPPEDWINDOW;

			return style;
		}
};

}

#endif

#endif
