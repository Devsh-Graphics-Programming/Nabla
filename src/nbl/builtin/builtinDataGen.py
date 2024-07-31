# Creates a c++ file for builtin resources that contains binary data of all resources

import argparse, os, subprocess, json
from datetime import datetime, timezone


parser = argparse.ArgumentParser(description="Creates a c++ file for builtin resources that contains binary data of all resources")
parser.add_argument('--outputBuiltinPath', required=True, help="output path of generated C++ builtin data source")
parser.add_argument('--outputArchivePath', required=True, help="output path of generated C++ archive data source")
parser.add_argument('--bundleAbsoluteEntryPath', required=True, help="\"absolute path\" for an archive which will store a given bundle of builtin resources")
parser.add_argument('--resourcesFile', required=True, help="path to file containing resources list")
parser.add_argument('--resourcesNamespace', required=True, help="a C++ namespace builtin resources will be wrapped into")
parser.add_argument('--correspondingHeaderFile', required=True, help="filename of previosly generated header (via buitinHeaderGen.py)")
parser.add_argument('--xxHash256Exe', default="", nargs='?', help="path to xxHash256 executable")

def execute(args):
    outputBuiltinPath = args.outputBuiltinPath
    outputArchivePath = args.outputArchivePath
    bundleAbsoluteEntryPath = args.bundleAbsoluteEntryPath
    resourcesFile = args.resourcesFile
    resourcesNamespace = args.resourcesNamespace
    correspondingHeaderFile = args.correspondingHeaderFile
    xxHash256Exe = args.xxHash256Exe
    
    forceConstexprHash = True if not xxHash256Exe else False

    file = open(resourcesFile, 'r')
    resourcePaths = file.readlines()

    outp = open(outputBuiltinPath, "w+")
    
    outp.write(f"""
    #include "{correspondingHeaderFile}"
    #include "nbl/core/hash/xxHash256.h"
    
    namespace {resourcesNamespace}
    {{
    
    static constexpr nbl::system::SBuiltinFile DUMMY_BUILTIN_FILE = {{ .contents = nullptr, .size = 0, .xx256Hash = 69, .modified = {{}} }};
    
    template<nbl::core::StringLiteral Path>
    const nbl::system::SBuiltinFile& get_resource();
    """)

    resourcesInitList = ""

    # writing binary data of all files + archive source in-place
    for id, z in enumerate(resourcePaths):
        itemData = z.split(',')
        x = itemData[0].rstrip()
        inputBuiltinResource = bundleAbsoluteEntryPath+'/'+x
        
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
            raise(IOError) # must throw back and fail the script

        fileSize = os.path.getsize(inputBuiltinResource)

        for item in itemData:
            resourcesInitList += f"\t\t\t{{\"{item.rstrip()}\", {fileSize}, 0xdeadbeefu, {id}, nbl::system::IFileArchive::E_ALLOCATOR_TYPE::EAT_NULL}},\n"
            
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

    archiveSource = f"""
#include "CArchive.h"

using namespace {resourcesNamespace};

static const std::shared_ptr<nbl::core::vector<nbl::system::IFileArchive::SFileList::SEntry>> k_builtinArchiveFileList = std::make_shared<nbl::core::vector<nbl::system::IFileArchive::SFileList::SEntry>>(
	nbl::core::vector<nbl::system::IFileArchive::SFileList::SEntry>{{
{resourcesInitList}
}});

CArchive::CArchive(nbl::system::logger_opt_smart_ptr&& logger)
	: nbl::system::CFileArchive(nbl::system::path(pathPrefix.data()),std::move(logger), k_builtinArchiveFileList)
{{
	
}}
"""
    
    outp = open(outputArchivePath, "w+")
    outp.write(archiveSource)
    outp.close()


if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    execute(args)