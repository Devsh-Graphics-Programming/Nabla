#ifndef I_WINDOWMANAGER
#define I_WINDOWMANAGER
#include <nbl/core/IReferenceCounted.h>
#include "IWindow.h"
namespace nbl::ui
{
	class IWindowManager : public core::IReferenceCounted
	{
		virtual core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) = 0;
		virtual void destroyWindow(IWindow* wnd) = 0;
	protected:
		virtual ~IWindowManager() = default;
	};
}
#endif