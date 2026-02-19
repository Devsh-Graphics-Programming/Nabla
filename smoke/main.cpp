#include <afxwin.h>

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

        if (!AfxWinInit(GetModuleHandle(nullptr), nullptr, GetCommandLineA(), 0))
        {
            std::cerr << "[ERROR]: Could not init AFX, terminating!\n";
            return false;
        }

        try {
            createAfxDummyWindow(320, 240, nullptr, _T("Dummy 1"));
            exportGpuProfiles();
            createAfxDummyWindow(320, 240, nullptr, _T("Dummy 2"));
        }
        catch (const std::exception& e) { 
            std::cerr << "[ERROR]: " << e.what() << '\n';
            return false;
        }
        catch (...) {
            std::cerr << "[ERROR]: Unknown exception!\n";
            return false;
        }

        return true;
    }

    void workLoopBody() override {}
    bool keepRunning() override { return false; }

    bool onAppTerminated() override
    {
        AfxWinTerm();
        return true;
    }

private:
    static void exportGpuProfiles()
    {
        std::string buf, arg1, arg2 = "-o", arg3;

        for (size_t i = 0;; i++)
        {
            auto six = std::to_string(i);
            arg1 = "--json=" + six;
            arg3 = "device_" + six + ".json";

            auto args = std::to_array<const char*>({ arg1.data(), arg2.data(), arg3.data()});
            int code = nbl::video::vulkaninfo(args);

            if (code != 0)
                break;

            std::ifstream input(arg3);
            
            while (std::getline(input, buf))
                std::cout << buf << "\n";

            std::cout << "\n\n";
        }
    }

    static bool createAfxDummyWindow(int w, int h, HWND parent, LPCTSTR windowName)
    {
        CWnd wnd;
        LPCTSTR cls = AfxRegisterWndClass(0, ::LoadCursor(nullptr, IDC_ARROW));
        if (!cls) return false;

        if (!wnd.CreateEx(0, cls, windowName, WS_POPUP | WS_VISIBLE, 0, 0, w, h, parent, nullptr))
            return false;

        MSG msg {};
        const ULONGLONG end = GetTickCount64() + 1000;
        while (GetTickCount64() < end) {
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            Sleep(1);
        }

        wnd.DestroyWindow();
        return true;
    }
};

NBL_MAIN_FUNC(Smoke)
#else
int main() { return 0; }
#endif