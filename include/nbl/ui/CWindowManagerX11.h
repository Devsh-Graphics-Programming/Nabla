#include <IWindowManager.h>

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
		void waitForCompletion(SRequest& req)
		{
			auto lk = req.wait();
		}
	private:
		void init() {}
		void exit() {}
		void work(lock_t& lock)
		{
		}
		bool wakeupPredicate() const { return true; }
		bool continuePredicate() const { return true; } 
	} m_windowThreadManager;
}

}