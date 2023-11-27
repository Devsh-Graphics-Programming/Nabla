// TODO: Cypi


// nsc input/simple_shader.hlsl -T ps_6_0 -E Main -Fo output/shader.ps

#include <iostream>
#include <cstdlib>
#include <vector>


bool no_nbl_builtins;


bool noNblBuiltinsEnabled(const std::vector<std::string>& args)
{
    for (auto i=0; i<args.size(); i++)
    {
        if (args[i] == "-no-nbl-builtins")
            return true;
    }
    return false;
}



int main(int argc, char* argv[])
{
    // std::cout << "\n\t\t:: ::\n";
    // std::cout << "\tNABLA SHADER COMPILER";
    // std::cout << "\n\t\t:: ::\n\n";


    std::vector<std::string> arguments(argv + 1, argv + argc);

    no_nbl_builtins = noNblBuiltinsEnabled(arguments);

    std::string command = "dxc.exe";
    for (std::string arg : arguments)
    {
        command.append(" ").append(arg);
    }

    int execute = std::system(command.c_str());

    std::string flag_set = no_nbl_builtins ? "TRUE" : "FALSE";

    std::cout << "-no-nbl-builtins - " << flag_set;


    std::cout << "\n\n\n\n  PRESS ENTER TO EXIT(): ";
    std::cin.get();
}