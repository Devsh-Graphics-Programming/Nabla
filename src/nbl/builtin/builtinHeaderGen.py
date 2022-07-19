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

    with open(resourcesFile, "r") as f:
        resourcePaths = f.read().rstrip().split(',')

    #opening a file
    outp = open(outputFilename,"w+")

    outp.write("#ifndef BUILTINRESOURCEDATA_H\n")
    outp.write("#define BUILTINRESOURCEDATA_H\n")
    outp.write("#include <stdlib.h>\n")
    outp.write("#include <cstdint>\n")
    outp.write("#include <string>\n")
    outp.write("#include <unordered_map>\n")
    outp.write("#include <utility>\n#include <nbl/core/string/UniqueStringLiteralType.h>\n#include <nbl/builtin/common.h>\n")
    outp.write("namespace nbl { \n\tnamespace builtin { \n")

    #Iterating through input list
    for x in resourcePaths:
        outp.write('\n\t\textern template const std::pair<const uint8_t*, size_t> get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("%s")>();' % x)

    outp.write("\n\t}\n}")
    outp.write("\n#endif")

    outp.close()
