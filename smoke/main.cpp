#include <nabla.h>
#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>

using namespace nbl;
using namespace nbl::system;
using namespace nbl::core;
using namespace nbl::asset;

class Smoke final : public system::IApplicationFramework
{
	using base_t = system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		const auto argc = argv.size();

		if (isAPILoaded())
		{
			std::cout << "Loaded Nabla API";
		}
		else
		{
			std::cerr << "Could not load Nabla API, terminating!";
			return false;
		}

		return true;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(Smoke)
