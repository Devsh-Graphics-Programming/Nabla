#ifndef _NBL_SYSTEM_PATH_H_INCLUDED_
#define _NBL_SYSTEM_PATH_H_INCLUDED_

#include "nbl/core/hash/blake.h"

#include <filesystem>


//TODO: Figure out where to move this
namespace nbl
{
namespace core
{
using string = std::string;

template<typename Dummy>
struct blake3_hasher::update_impl<core::string,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const core::string& input)
	{
        hasher << std::span<const char>(input);
	}
};
}

namespace system
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

namespace core
{
template<typename Dummy>
struct blake3_hasher::update_impl<system::path,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const system::path& input)
	{
        hasher << input.string();
	}
};
}
}
#endif