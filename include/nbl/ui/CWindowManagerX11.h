#ifdef _NBL_PLATFORM_LINUX_
#ifndef C_WINDOW_MANAGER_X11
#define C_WINDOW_MANAGER_X11

#include <nbl/ui/IWindowManager.h>
#include <X11/Xlib.h>
#include <nbl/ui/CWindowX11.h>
#include <X11/Xresource.h>
#include <X11/extensions/XInput.h>
#include <X11/extensions/xf86vmode.h>
#include "nbl/system/SReadWriteSpinLock.h"
#include <string>

namespace nbl::ui
{

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(X11, system::DefaultFuncPtrLoader
    ,XSetErrorHandler
    ,XOpenDisplay
    ,XFree
    ,XGetVisualInfo
    ,XCreateColormap
    ,XCreateWindow
    ,XMapRaised
    ,XInternAtom
    ,XSetWMProtocols
    ,XSetInputFocus
    ,XGrabKeyboard
    ,XGrabPointer
    ,XWarpPointer
    ,XGetErrorDatabaseText
    ,XGetErrorText
    ,XGetGeometry
    ,XFindContext
    ,XrmUniqueQuark
    ,XSaveContext
    ,XResizeWindow
    ,XMoveWindow
    ,XNextEvent
);

// TODO add more
NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xinput, system::DefaultFuncPtrLoader
	,XListInputDevices
	,XOpenDevice
	,XCloseDevice
	,XSetDeviceMode
	,XSelectExtensionEvent
	,XGetDeviceMotionEvents
	,XFreeDeviceMotionEvents	
);

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xrandr, system::DefaultFuncPtrLoader
    ,XF86VidModeSwitchToMode
    ,XF86VidModeSetViewPort
    ,XF86VidModeQueryExtension
    ,XF86VidModeGetAllModeLines
);

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xxf86vm, system::DefaultFuncPtrLoader
    ,XRRGetScreenInfo
    ,XRRSetScreenConfig
    ,XRRFreeScreenConfigInfo
    ,XRRQueryExtension
    ,XRRConfigSizes
);

static X11 x11("X11");
static Xinput xinput("Xinput");
static Xrandr xrandr("Xrandr");
static Xxf86vm xxf86vm("Xxf86vm");

class CWindowManagerX11 : public IWindowManager
{
    public:
        CWindowManagerX11();
        ~CWindowManagerX11() override = default;

        core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override;
        void destroyWindow(IWindow* wnd) override;
    protected:
        template<typename Key, typename Val>
        struct MultithreadedMap
        {
            public:
                MultithreadedMap() = default;
                MultithreadedMap(const MultithreadedMap&) = delete;
                MultithreadedMap(MultithreadedMap&&) = delete;
                MultithreadedMap& operator=(const MultithreadedMap&) = delete;
                MultithreadedMap& operator=(MultithreadedMap&&) = delete;

                mutable system::SReadWriteSpinLock m_lock;

                std::map<Key, Val> m_map;

                inline void insert(const Key& _key, const Val& _val)
                {
                    auto lk = lock_write();
                    m_map.insert(std::make_pair(_key, _val));
                }

                inline Val* read(const Key& _object)
                {
                    auto lk = lock_read();
                    auto r = m_map.find(_object);
                    if (r == m_map.end())
                        return nullptr;

                    return &(r->second);
                }

            protected:
                auto lock_read() const { return system::read_lock_guard<>(m_lock); }
                auto lock_write() const { return system::write_lock_guard<>(m_lock); }
        };
    private:
        class CThreadHandler final : public system::IThreadHandler<CThreadHandler>
        {
            using base_t = system::IThreadHandler<CThreadHandler>;
            friend base_t;
            friend CWindowManagerX11;

            public:
                CThreadHandler()
                {
                    this->start();
                }

            private:
                void work(lock_t& lock);
                void init();
                bool wakeupPredicate() const { return true; }
                bool continuePredicate() const { return true; }

                MultithreadedMap<Window, CWindowX11*> *m_windowsMapPtr;

                Display* m_dpy;
        } m_windowThreadManager;

        MultithreadedMap<Window, CWindowX11*> m_windowsMap;
        Display* m_dpy;

        core::vector<XID> getConnectedMice() const;
        core::vector<XID> getConnectedKeyboards() const;
};

}
#endif
#endif