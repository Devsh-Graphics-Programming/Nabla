#include <IWindowManager.h>
#include <X11/Xlib.h>
#include <CWindowX11.h>
#include <string>
namespace nbl::ui
{
class CWindowManagerX11 : public IWindowManager
{
public:
    CWindowManagerX11() = default;
    ~CWindowManagerX11() override = default;
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
		SRequestParams_CreateWindow(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t* wnd, const std::string_view& caption) :
			x(_x), y(_y), width(_w), height(_h), flags(_flags), nativeWindow(wnd), windowCaption(caption)
		{}
		int32_t x, y;
		uint32_t width, height;
		CWindowX11::E_CREATE_FLAGS flags;
		CWindowX11::native_handle_t* nativeWindow;
		std::string windowCaption;
	};
	struct SRequestParams_DestroyWindow : SRequestParamsBase<ERT_DESTROY_WINDOW>
	{
		CWindowWin32::native_handle_t nativeWindow;
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
		CThreadHandler()
		{
			this->start();
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
}

}