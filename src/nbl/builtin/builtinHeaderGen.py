# Creates a header file for builtin resources

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
    guardSuffix = sys.argv[5]
    isSharedLibrary = True if sys.argv[6] == "True" else False
    
    NBL_BR_API = "NBL_BR_API" if isSharedLibrary else ""

    file = open(resourcesFile, 'r')
    resourcePaths = file.readlines()

    outp = open(outputFilename, "w+")
    
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
