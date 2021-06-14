#include <IWindowManager.h>
#include <X11/Xlib.h>
#include <CWindowX11.h>
namespace nbl::ui
{
class CWindowManagerX11 : public IWindowManager
{
public:
    CWindowManagerX11() = default;
    ~CWindowManagerX11() override = default;
private:
    class CThreadHandler final : public system::IThreadHandler<CThreadHandler>
	{
		using base_t = system::IThreadHandler<CThreadHandler>;
		friend base_t;
	public:
		CThreadHandler()
		{
			this->start();
		}
	private:
		void init() {
			display = XOpenDisplay(nullptr);
		}
		void exit() {}
		void work(lock_t& lock)
		{
			XEvent event;
			XNextEvent(display, event);
			Window nativeWindow = event.xany.window;

			XPointer windowCharPtr = nullptr;
			XFindContext(display, nativeWindow, &windowCharPtr);
			CWindowX11* currentWindow = static_cast<CWindowX11*>(windowCharPtr);

			auto* eventCallback = nativeWindow->getEventCallback();

			currentWindow->processEvent(event);
		}
		bool wakeupPredicate() const { return true; }
		bool continuePredicate() const { return true; } 

	private:
		Display* display;
	} m_windowThreadManager;
}

}