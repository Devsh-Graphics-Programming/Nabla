// TODO: Cypi


// nsc input/simple_shader.hlsl -T ps_6_0 -E Main -Fo output/shader.ps

#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdlib>



class ShaderCompiler final : public system::IApplicationFramework
{
    using base_t = system::IApplicationFramework;

public:
    using base_t::base_t;

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        if (!base_t::onAppInitialized(std::move(system)))
            return false;
        
        auto argc = argv.size();

        core::vector<std::string> arguments(argv + 1, argv + argc);

        for (auto i=0; i<argc; i++)
        {
            if (argv[i] == "-no-nbl-builtins")
            {
                no_nbl_builtins = true;
                break;
            }
        }

        std::string command = "dxc.exe";
        for (std::string arg : arguments)
        {
            command.append(" ").append(arg);
        }

        int execute = std::system(command.c_str());

        std::cout << "-no-nbl-builtins - " << no_nbl_builtins;


        return true;
    }


    void workLoopBody() override {}

    bool keepRunning() override { return false; }


private:
    bool no_nbl_builtins{false};
};