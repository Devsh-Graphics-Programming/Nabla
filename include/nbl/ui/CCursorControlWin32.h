#ifndef __NBL_SYSTEM_C_CURSOR_CONTROL_WIN32_H_INCLUDED__
#define __NBL_SYSTEM_C_CURSOR_CONTROL_WIN32_H_INCLUDED__

#include "nbl/ui/ICursorControl.h"
#include "nbl/ui/CWindowManagerWin32.h"

#ifdef _NBL_PLATFORM_WINDOWS_
namespace nbl::ui
{
class NBL_API2 CCursorControlWin32 final : public ICursorControl
{
		core::smart_refctd_ptr<CWindowManagerWin32> m_windowManager;

	public:
		inline CCursorControlWin32(core::smart_refctd_ptr<CWindowManagerWin32>&& wmgr) : m_windowManager(std::move(wmgr)) {}

		void setVisible(bool visible) override;
		bool isVisible() const override;

		void setPosition(SPosition pos) override;
		void setRelativePosition(IWindow* window, SRelativePosition pos) override;

		SPosition getPosition() override;
		SRelativePosition getRelativePosition(IWindow* window) override;
};
}
#endif

#endif