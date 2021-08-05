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
	void CCursorControlWin32::setPosition(CCursorControlWin32::SPosition position)
	{
		SetCursorPos(position.x, position.y);
	}
	void CCursorControlWin32::setRelativePosition(CCursorControlWin32::SRelativePosition position)
	{
		SPosition nativePos;
		int32_t screenWidth = GetSystemMetrics(SM_CXSCREEN);
		int32_t screenHeight = GetSystemMetrics(SM_CYSCREEN);
		nativePos.x = (position.x / 2 + 0.5) * screenWidth;
		nativePos.y = (position.y / 2 + 0.5) * screenHeight;
		SetCursorPos(nativePos.x, nativePos.y);
	}
	CCursorControlWin32::SPosition CCursorControlWin32::getPosition()
	{
		POINT cursorPos;
		GetCursorPos(&cursorPos);
		return { cursorPos.x, cursorPos.y };
	}
	CCursorControlWin32::SRelativePosition CCursorControlWin32::getRelativePosition()
	{
		int32_t screenWidth = GetSystemMetrics(SM_CXSCREEN);
		int32_t screenHeight = GetSystemMetrics(SM_CYSCREEN);
		POINT cursorPos;
		GetCursorPos(&cursorPos);

		return { ((cursorPos.x + 0.5f) / float(screenWidth) - 0.5f) * 2, ((cursorPos.y + 0.5f) / float(screenHeight) - 0.5f) * 2 };
	}
}