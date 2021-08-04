#ifndef __NBL_SYSTEM_C_CURSOR_CONTROL_WIN32_H_INCLUDED__
#define __NBL_SYSTEM_C_CURSOR_CONTROL_WIN32_H_INCLUDED__
#include "nbl/ui/ICursorControl.h"
#include "nbl/ui/CWindowManagerWin32.h"

namespace nbl::ui
{
	class CCursorControlWin32 final : public ICursorControl
	{
		core::smart_refctd_ptr<CWindowManagerWin32> m_windowManager;
	public:
		CCursorControlWin32(core::smart_refctd_ptr<CWindowManagerWin32>&& wmgr) : m_windowManager(std::move(wmgr)) {}
		void setVisible(bool visible) override;
		bool isVisible() const override;
		void setPosition(int32_t x, int32_t y) override;
		void setPosition(const core::vector2d<int32_t>& pos) override;
		core::vector2di32_SIMD getPosition() override;
		core::vector2df_SIMD getRelativePosition() override;
	};
}
#endif