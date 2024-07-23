# Creates a header file for builtin resources

# TODO: use argparse not this by-hand-shit

import sys, os

if  len(sys.argv) < 8 :
    print(sys.argv[0] + " - Incorrect argument count")
else:
    outputBuiltinPath = sys.argv[1]
    outputArchivePath = sys.argv[2]
    archiveBundlePath = sys.argv[3]
    resourcesFile  = sys.argv[4]
    resourcesNamespace = sys.argv[5]
    guardSuffix = sys.argv[6]
    isSharedLibrary = True if sys.argv[7] == "True" else False
    
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
