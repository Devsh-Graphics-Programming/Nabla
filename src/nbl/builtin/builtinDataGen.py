# Creates a c++ file for builtin resources that contains binary data of all resources

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

    with open(resourcesFile, "r") as f:
        resourcePaths = f.read().rstrip().split(',')

    #opening a file
    outp = open(outputFilename,"w+")
    
    outp.write("#include \"" + correspondingHeaderFile + "\"\n")
    outp.write("\tnamespace " + resourcesNamespace + " {\n")

    outp.write("template<nbl::core::StringLiteral Path>")
    outp.write("const std::pair<const uint8_t*, size_t> get_resource();")
  
    # writing binary  data of all files in a loop
    for x in resourcePaths:
        outp.write('\n\ttemplate<> const std::pair<const uint8_t*, size_t> get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("%s")>()' % x)
        outp.write('\n\t{')
        outp.write('\n\t\tstatic const uint8_t data[] = {\n\t\t\t')
        try:
            with open(cmakeSourceDir+'/'+x, "rb") as f:
                index = 0
                byte = f.read(1)
                while byte != b"":
                    outp.write("0x%s, " % byte.hex())
                    index += 1  
                    if index % 20 == 0 :
                        outp.write("\n\t\t\t")
                    byte = f.read(1)
            # don't write null terminator, it messes up non-text files

        except IOError: 
            # file not found
            print('Error: BuiltinResources - file with the following path not found: ' + x)
            outp.write('\n\n Error: BuiltinResources - file with the following path not found: %s' % x )

        outp.write('\n\t\t};')
        outp.write('\n\t\treturn { data, sizeof(data) };')
        outp.write('\n\t}')


    outp.write("\tstd::pair<const uint8_t*, size_t> get_resource_runtime(const std::string& filename) {\n")
    outp.write("\t\tstatic std::unordered_map<std::string, int> resourcesByFilename( {\n")
    counter = 1
    for x in resourcePaths:
        outp.write("\t\t\t{\"%s\", %d},\n" % (x,counter))
        counter+= 1
    outp.write("\t\t});\n\n")
    outp.write("\t\tauto resource = resourcesByFilename.find(filename);\n")
    outp.write("\t\tif(resource == resourcesByFilename.end()) return { nullptr,0ull };\n")
    outp.write("\t\tswitch (resource->second) \n\t\t\t{\n")
    counter = 1
    for x in resourcePaths:
        outp.write("\t\t\tcase %d:\n\t\t\t\t\treturn get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(\"%s\")>();\n" % (counter,x))
        counter+= 1
  
    outp.write("\t\t\tdefault:\n")
    outp.write("\t\t\t\treturn { nullptr,0ull };\n")
    outp.write("\t\t}\n\t}")
    outp.write("\n}")
    outp.close()