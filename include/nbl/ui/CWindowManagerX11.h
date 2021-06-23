#ifdef _NBL_PLATFORM_LINUX_
#ifndef C_WINDOW_MANAGER_X11
#define C_WINDOW_MANAGER_X11

#include <IWindowManager.h>
#include <X11/Xlib.h>
#include <CWindowX11.h>
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
    ,XUniqueContext
    ,XSaveContext
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

#ifdef _NBL_LINUX_X11_RANDR_
NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xrandr, system::DefaultFuncPtrLoader
    ,XF86VidModeModeInfo
    ,XF86VidModeSwitchToMode
    ,XF86VidModeSetViewPort
    ,XF86VidModeQueryExtension
    ,XF86VidModeGetAllModeLines
    ,XF86VidModeSwitchToMode
    ,XF86VidModeSetViewPort
);
#endif
#ifdef _NBL_LINUX_X11_VIDMODE_
NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xxf86vm, system::DefaultFuncPtrLoader
    ,XRRGetScreenInfo
    ,XRRSetScreenConfig
    ,XRRFreeScreenConfigInfo
    ,XRRQueryExtension
    ,XRRGetScreenInfo
    ,XRRConfigSizes
    ,XRRSetScreenConfig
    ,XRRFreeScreenConfigInfo
);
#endif

class CWindowManagerX11 : public IWindowManager
{
public:
    CWindowManagerX11();
    ~CWindowManagerX11() override = default;

	core::smart_refctd_ptr<IWindow> createWindow(const IWindow::SCreationParams& creationParams) override;
	void destroyWindow(IWindow* wnd) override;
private:
	std::vector<XID> getConnectedMice() const;
	std::vector<XID> getConnectedKeyboards() const;

	Display* m_dpy;
private:
	enum E_REQUEST_TYPE
	{
		ERT_CREATE_WINDOW,
		ERT_DESTROY_WINDOW
	};
	template <E_REQUEST_TYPE ERT>
	struct SRequestParamsBase
	{
		static inline constexpr E_REQUEST_TYPE type = ERT;
	};
	struct SRequestParams_CreateWindow : SRequestParamsBase<ERT_CREATE_WINDOW>
	{
		SRequestParams_CreateWindow(int32_t _x, 
		int32_t _y,
		uint32_t _w,
		uint32_t _h, 
		CWindowX11::E_CREATE_FLAGS _flags, 
		CWindowX11::native_handle_t* wnd, 
		const std::string_view& caption, 
		Display* dsp) :
		x(_x), y(_y), width(_w), height(_h), flags(_flags), nativeWindow(wnd), windowCaption(caption), display(dsp)
		{}
		int32_t x, y;
		uint32_t width, height;
		CWindowX11::E_CREATE_FLAGS flags;
		CWindowX11::native_handle_t* nativeWindow;
		Display* display;
		std::string windowCaption;
	};
	struct SRequestParams_DestroyWindow : SRequestParamsBase<ERT_DESTROY_WINDOW>
	{
		Display* display;
		CWindowX11::native_handle_t nativeWindow;
	};
	struct SRequest : system::impl::IAsyncQueueDispatcherBase::request_base_t
	{
		E_REQUEST_TYPE type;
		union
		{
			SRequestParams_CreateWindow createWindowParam;
			SRequestParams_DestroyWindow destroyWindowParam;
		};
	};
    class CThreadHandler final : public system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u>
	{
		using base_t = system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u>
		friend base_t;
	public:
		CThreadHandler(Display* dpy)
		{
			display = dpy;
			this->start();
		}
		void createWindow(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowx11::E_CREATE_FLAGS _flags, CWindowx11::native_handle_t* wnd, const std::string_view& caption)
		{
			SRequestParams_CreateWindow params = SRequestParams_CreateWindow(_x, _y, _w, _h, _flags, wnd, display, caption);
			auto& rq = request(params);
			waitForCompletion(rq);
		}
		void destroyWindow(CWindowWin11::native_handle_t window)
		{
			SRequestParams_DestroyWindow params;
			params.nativeWindow = window;
			params.display = display;
			auto& rq = request(params);
			waitForCompletion(rq);
		}	
	private:
		void init();
		void exit() {}
		void background_work(lock_t& lock);
		void process_request(SRequest& req);

		template <typename RequestParams>
		void request_impl(SRequest& req, RequestParams&& params)
		{
			req.type = params.type;
			if constexpr (std::is_same_v<RequestParams, SRequestParams_CreateWindow>)
			{
				req.createWindowParam = std::move(params);
			}
			else
			{
				req.destroyWindowParam = std::move(params);
			}
		}
	private:
		void waitForCompletion(SRequest& req)
		{
			auto lk = req.wait();
		}
	private:
		Display* display;
	} m_windowThreadManager;
		X11 x11("X11");
		Xinput xinput("Xinput");
#ifdef _NBL_LINUX_X11_RANDR_
    	Xrandr xrandr("Xrandr");
#endif
#ifdef _NBL_LINUX_X11_VIDMODE_
    	Xxf86vm xxf86vm("Xxf86vm");
#endif	

}


}
#endif
#endif