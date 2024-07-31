#ifndef _NBL_SYSTEM_PATH_H_INCLUDED_
#define _NBL_SYSTEM_PATH_H_INCLUDED_

#include "nbl/core/hash/blake.h"

#include <filesystem>


//TODO: Figure out where to move this
namespace nbl::core
{
using string = std::string;

template<>
inline void blake3_hasher_update(blake3_hasher& self, const core::string& input)
{
	::blake3_hasher_update(&self,input.data(),input.length()+1);
}
}

namespace nbl::system
{
using path = std::filesystem::path;
 
inline nbl::core::string extension_wo_dot(const path& _filename)
{
    std::string extension = _filename.extension().string();
    if (extension.empty())
        return extension;

    return extension.substr(1u, nbl::core::string::npos);
}

inline path filename_wo_extension(const path& filename)
{
    path ret = filename;
    return ret.replace_extension();
}
}
#endif