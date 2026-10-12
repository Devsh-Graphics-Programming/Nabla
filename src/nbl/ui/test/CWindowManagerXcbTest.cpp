// Regression test for the threading rules documented in `IWindowManagerXcb.h`, needs an X server but no GPU.
// Exit codes: 0 pass, 1 failed check, 2 timed out (deadlock), 77 skipped (no X server).
#include "nabla.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace nbl;
using namespace nbl::ui;

namespace
{
constexpr auto Timeout = std::chrono::seconds(10);

std::atomic_bool g_failed = false;
#define NBL_CHECK(COND) if (!(COND)) { std::fprintf(stderr,"FAILED: %s (%s:%d)\n",#COND,__FILE__,__LINE__); g_failed = true; }

template<typename F>
bool waitFor(F&& pred)
{
	const auto end = std::chrono::steady_clock::now()+Timeout;
	while (!pred())
	{
		if (std::chrono::steady_clock::now()>end)
			return false;
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	return true;
}

class CChildCallback final : public IWindow::IEventCallback
{
	public:
		std::atomic_bool shown = false;

	private:
		bool onWindowShown_impl() override {shown = true; return true;}
};

// Creates a second window from inside a callback, then drops it from inside another callback of the first window
class CParentCallback final : public IWindow::IEventCallback
{
	public:
		CParentCallback(IWindowManager* manager, std::thread::id mainThread) : m_manager(manager), m_mainThread(mainThread) {}

		std::atomic_bool childCreated = false;
		std::atomic_bool childDropped = false;
		std::atomic_bool resized = false;
		core::smart_refctd_ptr<CChildCallback> childCallback = core::make_smart_refctd_ptr<CChildCallback>();
		// only touched on the event thread
		core::smart_refctd_ptr<IWindow> child;

	private:
		bool onWindowShown_impl() override
		{
			NBL_CHECK(std::this_thread::get_id()!=m_mainThread);
			if (!child)
			{
				IWindow::SCreationParams params = {};
				params.callback = childCallback;
				params.width = 64;
				params.height = 64;
				params.windowCaption = "NAB-9 child";
				// used to deadlock, the event thread waited on its own request queue
				child = m_manager->createWindow(std::move(params));
				NBL_CHECK(child);
				childCreated = true;
			}
			return true;
		}
		bool onWindowResized_impl(uint32_t w, uint32_t h) override
		{
			NBL_CHECK(std::this_thread::get_id()!=m_mainThread);
			// last reference to another window, destroyed inline on the event thread
			if (child && w==200u)
			{
				child = nullptr;
				childDropped = true;
			}
			resized = true;
			return true;
		}

		IWindowManager* const m_manager;
		const std::thread::id m_mainThread;
};
}

int main()
{
	// a deadlock must fail the test instead of hanging CTest
	std::thread([](){
		std::this_thread::sleep_for(Timeout*3);
		std::fprintf(stderr,"FAILED: timed out, deadlock?\n");
		std::_Exit(2);
	}).detach();

	auto manager = IWindowManagerXcb::create();
	if (!manager)
	{
		std::fprintf(stderr,"SKIPPED: no X server (DISPLAY=%s)\n",std::getenv("DISPLAY") ? std::getenv("DISPLAY"):"");
		return 77;
	}

	auto callback = core::make_smart_refctd_ptr<CParentCallback>(manager.get(),std::this_thread::get_id());
	{
		IWindow::SCreationParams params = {};
		params.callback = callback;
		params.x = 32;
		params.y = 32;
		params.width = 128;
		params.height = 128;
		params.flags = IWindow::ECF_RESIZABLE;
		params.windowCaption = "NAB-9 parent";
		auto parent = manager->createWindow(std::move(params));
		NBL_CHECK(parent);
		if (!parent)
			return 1;

		NBL_CHECK(waitFor([&](){return callback->childCreated.load();}));
		NBL_CHECK(waitFor([&](){return callback->childCallback->shown.load();}));

		NBL_CHECK(manager->setWindowSize(parent.get(),200,150));
		NBL_CHECK(waitFor([&](){return callback->childDropped.load();}));
		// the event thread must still be alive after destroying a window inline
		callback->resized = false;
		NBL_CHECK(manager->setWindowSize(parent.get(),220,170));
		NBL_CHECK(waitFor([&](){return callback->resized.load();}));
		// destroyed from the main thread, waits for the event thread
	}
	callback = nullptr;
	manager = nullptr;

	if (g_failed)
		return 1;
	std::printf("PASSED\n");
	return 0;
}
