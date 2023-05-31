# Creates a header file for builtin resources


# parameters are
#0 - path to the .py file
#1 - output file path
#2 - cmake source dir
#3 - list of paths to resource files

import sys
import os
if  len(sys.argv) < 4 :
    print(sys.argv[0] + " - Incorrect argument count")
else:
    #arguments
    outputFilename = sys.argv[1]
    cmakeSourceDir = sys.argv[2]
    resourcesFile  = sys.argv[3]
    resourcesNamespace = sys.argv[4]
    guardSuffix = sys.argv[5]
    isSharedLibrary = True if sys.argv[6] == "True" else False

    file = open(resourcesFile, 'r')
    resourcePaths = file.readlines()

    #opening a file
    outp = open(outputFilename,"w+")

    outp.write("#ifndef _" + guardSuffix + "_BUILTINRESOURCEDATA_H_\n")
    outp.write("#define _" + guardSuffix + "_BUILTINRESOURCEDATA_H_\n")
  
    outp.write("#ifdef __INTELLISENSE__\n")
    outp.write("#include <codeanalysis\warnings.h>\n")
    outp.write("#pragma warning( push )\n")
    outp.write("#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )\n")
    outp.write("#endif // __INTELLISENSE__\n")

    outp.write("#include <stdlib.h>\n")
    outp.write("#include <cstdint>\n")
    outp.write("#include <string>\n")
    outp.write("#include <unordered_map>\n")
    outp.write("#include <utility>\n#include <nbl/core/string/StringLiteral.h>\n\n")
    
    if isSharedLibrary:   
        outp.write("#if defined(__NBL_BUILDING_TARGET__) // currently compiling the target, this define is passed through the commandline\n")
        outp.write("#if defined(_MSC_VER)\n")
        outp.write("#define NBL_BR_API __declspec(dllexport)\n")
        outp.write("#elif defined(__GNUC__)\n")
        outp.write('#define NBL_BR_API __attribute__ ((visibility ("default")))' + "\n")
        outp.write("#endif\n")
        outp.write("#else\n")
        outp.write("#if defined(_MSC_VER)\n")
        outp.write("#define NBL_BR_API __declspec(dllimport)\n")
        outp.write("#else\n")
        outp.write("#define NBL_BR_API\n")
        outp.write("#endif\n")
        outp.write("#endif\n\n")
    
    outp.write("namespace " + resourcesNamespace + " { \n")
    
    if isSharedLibrary:
        outp.write("\t\tNBL_BR_API std::pair<const uint8_t*, size_t> get_resource_runtime(const std::string& filename);\n\n")
    else:
        outp.write("\t\tstd::pair<const uint8_t*, size_t> get_resource_runtime(const std::string& filename);\n\n")
    
    outp.write("\t\ttemplate<nbl::core::StringLiteral Path>\n")
    outp.write("\t\tconst std::pair<const uint8_t*, size_t> get_resource();\n")
    
    #Iterating through input list
    for z in resourcePaths:
        itemData = z.split(',')
        x = itemData[0].rstrip()
        
        if isSharedLibrary:
            outp.write('\n\t\ttemplate<> NBL_BR_API const std::pair<const uint8_t*, size_t> get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("%s")>();' % x)
        else:
            outp.write('\n\t\ttemplate<> const std::pair<const uint8_t*, size_t> get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("%s")>();' % x)
        
        if len(itemData) > 1:
            for i in range(1, len(itemData)):
                if isSharedLibrary:
                    outp.write('\n\t\ttemplate<> NBL_BR_API const std::pair<const uint8_t*, size_t> get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("%s")>();' % itemData[i].rstrip())
                else:
                    outp.write('\n\t\ttemplate<> const std::pair<const uint8_t*, size_t> get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("%s")>();' % itemData[i].rstrip())

    outp.write("\n\t}\n")
    
    outp.write("#ifdef __INTELLISENSE__\n")
    outp.write("#pragma warning( pop )\n")
    outp.write("#endif // __INTELLISENSE__\n")
    
    outp.write("#endif // _" + guardSuffix + "_BUILTINRESOURCEDATA_H_")

    outp.close()
