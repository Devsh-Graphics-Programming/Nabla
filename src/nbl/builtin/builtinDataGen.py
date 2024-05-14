# Creates a c++ file for builtin resources that contains binary data of all resources

# parameters are
# 0 - path to the .py file
# 1 - output file path
# 2 - cmake source dir
# 3 - list of paths to resource files

import sys, os, subprocess, json
from datetime import datetime, timezone

if  len(sys.argv) < 4 :
    print(sys.argv[0] + " - Incorrect argument count")
else:
    outputFilename = sys.argv[1]
    cmakeSourceDir = sys.argv[2]
    resourcesFile  = sys.argv[3]
    resourcesNamespace = sys.argv[4]
    correspondingHeaderFile = sys.argv[5]
    xxHash256Exe = sys.argv[6]
    
    forceConstexprHash = True if not xxHash256Exe else False

    file = open(resourcesFile, 'r')
    resourcePaths = file.readlines()

    outp = open(outputFilename, "w+")
    
    outp.write(f"""
    #include "{correspondingHeaderFile}"
    #include "nbl/core/xxHash256.h"
    
    namespace {resourcesNamespace}
    {{
    
    static constexpr nbl::system::SBuiltinFile DUMMY_BUILTIN_FILE = {{ .contents = nullptr, .size = 0, .xx256Hash = 69, .modified = {{}} }};
    
    template<nbl::core::StringLiteral Path>
    const nbl::system::SBuiltinFile& get_resource();
    """)
     
    # writing binary data of all files
    for z in resourcePaths:
        itemData = z.split(',')
        x = itemData[0].rstrip()
        inputBuiltinResource = cmakeSourceDir+'/'+x
        
        outp.write(f"""
        template<> const nbl::system::SBuiltinFile& get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{x}")>()
        {{
        static constexpr uint8_t data[] = {{
        """)
            
        try:
            with open(inputBuiltinResource, "rb") as f:
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
            
        modificationDateT = datetime.fromtimestamp(os.path.getmtime(inputBuiltinResource), timezone.utc) # since the Unix epoch (00:00:00 UTC on 1 January 1970).
        
        cppHashInitS = ""        
        
        if forceConstexprHash:
            cppHashInitS = "nbl::core::XXHash_256(data, sizeof(data))"
        else:
            jsonContent = subprocess.run([xxHash256Exe, "--file", inputBuiltinResource], capture_output=True, text=True, shell=True)
            hashArray = []
        
            if jsonContent.returncode == 0:
                try:
                    jOutput = json.loads(jsonContent.stdout)
                    hashArray = [int(x) for x in jOutput.get("u64hash", [])]
                except ValueError as e:
                    print("Failed to parse JSON or convert hash elements to integers. Error:", e)
            else:
                print("Failed to execute the command. Error:", jsonContent.stderr)
                
            cppHashInitS = f"{{ {hashArray[0]},{hashArray[1]},{hashArray[2]},{hashArray[3]} }}"
            
        outp.write(f"""
        }};
        
        static constexpr nbl::system::SBuiltinFile builtinFile = {{ .contents = data, .size = sizeof(data), .xx256Hash = {cppHashInitS}, 
        .modified = {{
            .tm_sec = {modificationDateT.second},
            .tm_min = {modificationDateT.minute},
            .tm_hour = {modificationDateT.hour},
            .tm_mday = {modificationDateT.day},
            .tm_mon = {modificationDateT.month - 1},
            .tm_year = {modificationDateT.year - 1900},
            .tm_isdst = 0}} 
        }};    
        
        return builtinFile;
        }}
        """)
        
        if len(itemData) > 1:
            for i in range(1, len(itemData)):
                outp.write(f"""
                template<> const nbl::system::SBuiltinFile& get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{itemData[i].rstrip()}")>()
                {{
                    return get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{x}")>();
                }}
                """)
    
    outp.write(f"""
    const nbl::system::SBuiltinFile& get_resource_runtime(const std::string& filename) {{
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