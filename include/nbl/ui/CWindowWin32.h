#ifndef __C_WINDOW_WIN32_H_INCLUDED__
#define __C_WINDOW_WIN32_H_INCLUDED__

#include "nbl/ui/IWindowWin32.h"
#include "os.h"
#include <queue>

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
{

class CWindowWin32 final : public IWindowWin32
{
	// TODO design for event listener etc, WndProc will somehow pass events to provided event listener

public:
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	//TODO
	explicit CWindowWin32(core::smart_refctd_ptr<system::ISystem>&& sys, native_handle_t hwnd) : IWindowWin32(std::move(sys))
	{
		assert(false);
		RECT rect;
		GetWindowRect(hwnd, &rect);

		m_width = rect.right - rect.left;
		m_height = rect.bottom - rect.top;
		// TODO m_flags
	}

	native_handle_t getNativeHandle() const override { return m_native; }

	static core::smart_refctd_ptr<CWindowWin32> create(core::smart_refctd_ptr<system::ISystem>&& sys, int _x, int _y, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		if ((_flags & (ECF_MINIMIZED | ECF_MAXIMIZED)) == (ECF_MINIMIZED | ECF_MAXIMIZED))
			return nullptr;

		CWindowWin32* win = new CWindowWin32(std::move(sys), _x, _y, _w, _h, _flags);
		return core::smart_refctd_ptr<CWindowWin32>(win, core::dont_grab);
	}

	~CWindowWin32() override;
private:
	CWindowWin32(core::smart_refctd_ptr<system::ISystem>&& sys, int _x, int _y, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);

    native_handle_t m_native;
};

}
}

#endif

#endif
