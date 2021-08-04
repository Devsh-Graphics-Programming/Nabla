#include <Windows.h>

#include "nbl/ui/CCursorControlWin32.h"


namespace nbl::ui
{
	void CCursorControlWin32::setVisible(bool visible)
	{
		m_windowManager->setCursorVisibility(visible);
	}
	bool CCursorControlWin32::isVisible() const
	{
		CURSORINFO ci = { sizeof(CURSORINFO) };
		GetCursorInfo(&ci);
		return ci.flags; // returning flags cause they're equal to 0 when the cursor is hidden
	}
	void CCursorControlWin32::setPosition(int32_t x, int32_t y)
	{
		SetCursorPos(x, y);
	}
	void CCursorControlWin32::setPosition(const core::vector2d<int32_t>& pos)
	{
		SetCursorPos(pos.X, pos.Y);
	}
	core::vector2di32_SIMD CCursorControlWin32::getPosition()
	{
		POINT cursorPos;
		GetCursorPos(&cursorPos);
		return core::vector2di32_SIMD{ cursorPos.x, cursorPos.y };
	}
	core::vector2df_SIMD CCursorControlWin32::getRelativePosition()
	{
		int32_t screenWidth = GetSystemMetrics(SM_CXSCREEN);
		int32_t screenHeight = GetSystemMetrics(SM_CYSCREEN);
		POINT cursorPos;
		GetCursorPos(&cursorPos);

		return core::vector2df_SIMD((cursorPos.x / float(screenWidth) - 0.5) * 2, (cursorPos.y / float(screenHeight) - 0.5) * 2);
	}
}