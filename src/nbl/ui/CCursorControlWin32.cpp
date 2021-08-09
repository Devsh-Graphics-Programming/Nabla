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
	void CCursorControlWin32::setRelativePosition(IWindow* window, CCursorControlWin32::SRelativePosition position)
	{
		SPosition nativePos;
		int32_t w = window->getWidth();
		int32_t h = window->getHeight();
		nativePos.x = (position.x / 2.f + 0.5f) * w + window->getX();
		nativePos.y = (position.y / 2.f + 0.5f) * h + window->getY();
		SetCursorPos(nativePos.x, nativePos.y);
	}
	CCursorControlWin32::SPosition CCursorControlWin32::getPosition()
	{
		POINT cursorPos;
		GetCursorPos(&cursorPos);
		return { cursorPos.x, cursorPos.y };
	}
	CCursorControlWin32::SRelativePosition CCursorControlWin32::getRelativePosition(IWindow* window)
	{
		POINT cursorPos;
		GetCursorPos(&cursorPos);

		return { ((cursorPos.x + 0.5f - window->getX()) / float(window->getWidth()) - 0.5f) * 2, ((cursorPos.y + 0.5f - window->getY()) / float(window->getWidth()) - 0.5f) * 2 };
	}
}