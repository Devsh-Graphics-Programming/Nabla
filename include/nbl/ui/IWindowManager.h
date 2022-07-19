#ifndef _NBL_UI_I_WINDOWMANAGER_
#define _NBL_UI_I_WINDOWMANAGER_

#include <nbl/core/IReferenceCounted.h>
#include "IWindow.h"

namespace nbl::ui
{

struct SDisplayInfo
{
	int32_t x;
	int32_t y;
	uint32_t resX;
	uint32_t resY;
	std::string name; // this one is really more of a placeholder right now
};

class NBL_API2 IWindowManager : public core::IReferenceCounted
{
	public:
		virtual core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) = 0;
		virtual SDisplayInfo getPrimaryDisplayInfo() const = 0;

		virtual bool setWindowSize_impl(IWindow* window, uint32_t width, uint32_t height) = 0;
		virtual bool setWindowPosition_impl(IWindow* window, int32_t x, int32_t y) = 0;
		virtual bool setWindowRotation_impl(IWindow* window, bool landscape) = 0;
		virtual bool setWindowVisible_impl(IWindow* window, bool visible) = 0;
		virtual bool setWindowMaximized_impl(IWindow* window, bool maximized) = 0;

		inline bool setWindowSize(IWindow* window, const uint32_t width, const uint32_t height)
		{
			auto cb = window->getEventCallback();
			if (window->getManager() != this || !window->isResizable() || cb && !cb->onWindowResized(window, width, height))
				return false;

			return setWindowSize_impl(window, width, height);
		}

		inline bool setWindowPosition(IWindow* window, const int32_t x, const int32_t y)
		{
			auto cb = window->getEventCallback();
			if (window->getManager() != this || !window->isResizable() || cb && !cb->onWindowMoved(window, x, y))
				return false;

			return setWindowPosition_impl(window, x, y);
		}

		inline bool setWindowRotation(IWindow* window, const bool landscape)
		{
			// TODO
			/*
			auto cb = window->getEventCallback();
			if (window->getManager() != this || !window->canRotate() || landscape == window->isRotationLandscape() || cb && !cb->onWindowRotated(window))
				return false;

			return setWindowRotation_impl(window, landscape);
			*/
			return false;
		}

		inline bool show(IWindow* window)
		{
			auto cb = window->getEventCallback();
			if (window->getManager() != this || !window->isResizable() || !window->isHidden() || cb && !cb->onWindowShown(window))
				return false;

			return setWindowVisible_impl(window, true);
		}

		inline bool hide(IWindow* window)
		{
			auto cb = window->getEventCallback();
			if (window->getManager() != this || !window->isResizable() || window->isHidden() || cb && !cb->onWindowHidden(window))
				return false;

			return setWindowVisible_impl(window, false);
		}

		inline bool maximize(IWindow* window)
		{
			auto cb = window->getEventCallback();
			if (window->getManager() != this || !window->isResizable() || window->isMaximized() || cb && !cb->onWindowMaximized(window))
				return false;

			return setWindowMaximized_impl(window, true);
		}

		inline bool minimize(IWindow* window)
		{
			auto cb = window->getEventCallback();
			if (window->getManager() != this || !window->isResizable() || window->isMinimized() || cb && !cb->onWindowMinimized(window))
				return false;

			return setWindowMaximized_impl(window, false);
		}
	private:
		virtual void destroyWindow(IWindow* wnd) = 0;

	protected:
		virtual ~IWindowManager() = default;
};

}
#endif