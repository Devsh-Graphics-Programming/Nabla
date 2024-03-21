# Creates a c++ file for builtin resources that contains binary data of all resources

# parameters are
# 0 - path to the .py file
# 1 - output file path
# 2 - cmake source dir
# 3 - list of paths to resource files

import sys, os

if  len(sys.argv) < 4 :
    print(sys.argv[0] + " - Incorrect argument count")
else:
    outputFilename = sys.argv[1]
    cmakeSourceDir = sys.argv[2]
    resourcesFile  = sys.argv[3]
    resourcesNamespace = sys.argv[4]
    correspondingHeaderFile = sys.argv[5]

    file = open(resourcesFile, 'r')
    resourcePaths = file.readlines()

    outp = open(outputFilename, "w+")
    
    outp.write(f"""
    #include "{correspondingHeaderFile}"
    namespace {resourcesNamespace}
    {{
    
    static constexpr SBuiltinFile DUMMY_BUILTIN_FILE = {{ .contents = nullptr, .size = 0, .xx256Hash = 69, .modified = {{}} }};
    
    template<nbl::core::StringLiteral Path>
    const SBuiltinFile& get_resource();
    """)
     
    # writing binary data of all files
    for z in resourcePaths:
        itemData = z.split(',')
        x = itemData[0].rstrip()
        
        outp.write(f"""
        template<> const SBuiltinFile& get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{x}")>()
        {{
        static constexpr uint8_t data[] = {{
        """)
            
        try:
            with open(cmakeSourceDir+'/'+x, "rb") as f:
                index = 0
                byte = f.read(1)
                while byte != b"":
                    outp.write("0x%s, " % byte.hex())
                    index += 1  
                    if index % 20 == 0 :
                        outp.write("""
                        """)
                    byte = f.read(1)
            # don't write null terminator, it messes up non-text files

        except IOError: 
            # file not found
            print(f"Error: BuiltinResources - file with the following path not found: {x}")
            outp.write(f"""
            Error: BuiltinResources - file with the following path not found: {x}
            """)
            
        outp.write(f"""
        }};
        
        static constexpr SBuiltinFile builtinFile = {{ .contents = data, .size = sizeof(data), .xx256Hash = 69, .modified = {{}} }};    
        
        return builtinFile;
        }}
        """)
        
        if len(itemData) > 1:
            for i in range(1, len(itemData)):
                outp.write(f"""
                template<> const SBuiltinFile& get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{itemData[i].rstrip()}")>()
                {{
                    return get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{x}")>();
                }}
                """)
    
    outp.write(f"""
    const SBuiltinFile& get_resource_runtime(const std::string& filename) {{
    static std::unordered_map<std::string, int> resourcesByFilename( {{
    """)

    counter = 1
    
    for z in resourcePaths:
        itemData = z.split(',')
        x = itemData[0].rstrip()
        
        outp.write("\t\t\t{\"%s\", %d},\n" % (x,counter))
        
        if len(itemData) > 1:
            for i in range(1, len(itemData)):
                outp.write("\t\t\t{\"%s\", %d},\n" % (itemData[i].rstrip(),counter))
                
        counter+= 1
        
    outp.write(f"""
    }});
    
    auto resource = resourcesByFilename.find(filename);
    if(resource == resourcesByFilename.end())
        return DUMMY_BUILTIN_FILE;
        
    switch (resource->second)
    {{
    """)

    counter = 1
    
    for z in resourcePaths:
        itemData = z.split(',')
        x = itemData[0].rstrip()
        
        outp.write(f"""
        case {counter}:
            return get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{x}")>();
        """)
        
        counter+= 1
  
    outp.write(f"""
    default:
        return DUMMY_BUILTIN_FILE;
    }}
    }}
    }}
    """)
    
    outp.close()