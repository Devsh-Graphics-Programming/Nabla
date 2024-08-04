# Creates a header file for builtin resources

import argparse, sys, os


parser = argparse.ArgumentParser(description="Creates a c++ file for builtin resources that contains binary data of all resources")
parser.add_argument('--outputBuiltinPath', required=True, help="output path of generated C++ builtin header source")
parser.add_argument('--outputArchivePath', required=True, help="output path of generated C++ archive header source")
parser.add_argument('--archiveBundlePath', required=True, help="path for an archive which will store a given bundle of builtin resources")
parser.add_argument('--resourcesFile', required=True, help="path for a file which containins list of resources")
parser.add_argument('--resourcesNamespace', required=True, help="a C++ namespace builtin resources will be wrapped into")
parser.add_argument('--guardSuffix', required=True, help="include guard suffix name, for C header files")
parser.add_argument('--isSharedLibrary', required=True, choices=["True", "False"])

def execute(args):
    outputBuiltinPath = args.outputBuiltinPath
    outputArchivePath = args.outputArchivePath
    archiveBundlePath = args.archiveBundlePath
    resourcesFile = args.resourcesFile
    resourcesNamespace = args.resourcesNamespace
    guardSuffix = args.guardSuffix
    isSharedLibrary = True if args.isSharedLibrary == "True" else False
    
    NBL_BR_API = "NBL_BR_API" if isSharedLibrary else ""

    file = open(resourcesFile, 'r')
    resourcePaths = file.readlines()

    outp = open(outputBuiltinPath, "w+")
    
    outp.write(f"""
#ifndef _{guardSuffix}_BUILTINRESOURCEDATA_H_
#define _{guardSuffix}_BUILTINRESOURCEDATA_H_

#ifdef __INTELLISENSE__
#include <codeanalysis\warnings.h>
#pragma warning( push )
#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#endif // __INTELLISENSE__

#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <nbl/system/SBuiltinFile.h>
#include <nbl/core/string/StringLiteral.h>
    """)
    
    if isSharedLibrary:   
        outp.write(f"""
    #if defined(__NBL_BUILDING_TARGET__) // currently compiling the target, this define is passed through the commandline
        #if defined(_MSC_VER)
            #define NBL_BR_API __declspec(dllexport)
        #elif defined(__GNUC__)
             #define NBL_BR_API __attribute__ ((visibility ("default")))
    #endif
    #else
        #if defined(_MSC_VER)
            #define NBL_BR_API __declspec(dllimport)
        #else
            #define NBL_BR_API
        #endif
    #endif
        """)
    
    outp.write(f"""
    namespace {resourcesNamespace}
{{
    {NBL_BR_API}
    const nbl::system::SBuiltinFile& get_resource_runtime(const std::string& filename);

    template<nbl::core::StringLiteral Path>
    const nbl::system::SBuiltinFile& get_resource();
    """)
        
    # Iterating through input list
    for z in resourcePaths:
        itemData = z.split(',')
        x = itemData[0].rstrip()
        
        outp.write(f"""
        template<> {NBL_BR_API}
        const nbl::system::SBuiltinFile& get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{x}")>();
        """)
        
        if len(itemData) > 1:
            for i in range(1, len(itemData)):
                outp.write(f"""
                template<> {NBL_BR_API}
                const nbl::system::SBuiltinFile& get_resource<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("{itemData[i].rstrip()}")>();
                """)

    outp.write(f"""
}}
#ifdef __INTELLISENSE__
#pragma warning( pop )
#endif // __INTELLISENSE__
#endif // _{guardSuffix}_BUILTINRESOURCEDATA_H_
    """)
    
    outp.close()

    archiveHeader = f"""
#ifndef _{guardSuffix}_C_ARCHIVE_H_
#define _{guardSuffix}_C_ARCHIVE_H_

#include "nbl/system/CFileArchive.h"
#include "nbl/core/def/smart_refctd_ptr.h"
#include "{os.path.basename(outputBuiltinPath)}"
#include <memory>

namespace {resourcesNamespace}
{{
constexpr std::string_view pathPrefix = "{archiveBundlePath}";

inline bool hasPathPrefix(nbl::system::path _path)
{{
	_path.make_preferred();
	const auto prefix = nbl::system::path(pathPrefix).make_preferred();
	return _path.string().find(prefix.string())==0ull;
}}

class {NBL_BR_API} CArchive final : public nbl::system::CFileArchive
{{
	public:
		CArchive(nbl::system::logger_opt_smart_ptr&& logger);
			
	protected:
		file_buffer_t getFileBuffer(const nbl::system::IFileArchive::SFileList::found_t& found) override
		{{
				auto resource = get_resource_runtime(found->pathRelativeToArchive.string());
				return {{const_cast<uint8_t*>(resource.contents),resource.size,nullptr}};
		}}			
}};
}}

#endif // _{guardSuffix}_C_ARCHIVE_H_
"""

    outp = open(outputArchivePath, "w+")
    outp.write(archiveHeader)
    outp.close()

if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    execute(args)