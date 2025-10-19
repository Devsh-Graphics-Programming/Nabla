#define ENABLE_SMOKE

using namespace nbl;
using namespace nbl::system;
using namespace nbl::core;
using namespace nbl::asset;

#ifdef ENABLE_SMOKE

class Smoke final : public system::IApplicationFramework
{
    using base_t = system::IApplicationFramework;

public:
    using base_t::base_t;

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        const char* sdk = std::getenv("NBL_INSTALL_DIRECTORY");

        if (sdk)
        {
            auto dir = std::filesystem::absolute(std::filesystem::path(sdk).make_preferred()).string();
            std::cout << "[INFO]: NBL_INSTALL_DIRECTORY = \"" << dir.c_str() << "\"\n";
        }
        else
            std::cerr << "[INFO]: NBL_INSTALL_DIRECTORY env was not defined!\n";

        if (isAPILoaded())
        {
            std::cout << "[INFO]: Loaded Nabla API\n";
        }
        else
        {
            std::cerr << "[ERROR]: Could not load Nabla API, terminating!\n";
            return false;
        }

        nbl::video::vulkaninfo();

        return true;
    }

    void workLoopBody() override {}
    bool keepRunning() override { return false; }

private:
    smart_refctd_ptr<CVulkanConnection> m_api;
};

NBL_MAIN_FUNC(Smoke)
#else
int main() { return 0; }
#endif