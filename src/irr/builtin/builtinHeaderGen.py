# Creates a header file for builtin resources


# parameters are
#0 - path to the .py file
#1 - output file path
#2 - cmake source dir
#3 - list of paths to resource files

import sys

# for getting the file size
import os

if  len(sys.argv) < 4 :
    print(sys.argv[0] + " - Incorrect argument count")
else:
    print("Running")
    #arguments
    outputFilename = sys.argv[1]
    cmakeSourceDir = sys.argv[2]
    resourcePaths = sys.argv[3].split(';')
    print(sys.argv[3])

    #opening a file
    outp = open(outputFilename,"w+")

    print("Created file at %s" % outputFilename)

    outp.write("#include <stdlib.h>\n")
    outp.write("#include <cstdint>\n")
    outp.write("#include <utility>\n#include <irr\\core\\string\\UniqueStringLiteralType.h>\n")
    outp.write("namespace irr { \n\tnamespace builtin { \n\t\ttemplate<typename StringUniqueLiteralType>\n")
    outp.write("\t\tconst std::pair<const uint8_t*, size_t> get_resource() \n\t\t{\n\t\t\treturn { nullptr,0ull };\n\t\t}")

    #Iterating through input list
    for x in resourcePaths:
        outp.write('\n\t\textern template const std::pair<const uint8_t*, size_t> get_resource<IRR_CORE_UNIQUE_STRING_LITERAL_TYPE("%s")>();' % x)


    outp.write("\n\t}\n}")

    outp.close()