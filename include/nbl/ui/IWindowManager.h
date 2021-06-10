#ifndef I_WINDOWMANAGER
#define I_WINDOWMANAGER
#include <nbl/core/IReferenceCounted.h>

namespace nbl::ui
{
	class IWindowManager : core::IReferenceCounted
	{
	protected:
		virtual ~IWindowManager() = default;
	};
}
#endif