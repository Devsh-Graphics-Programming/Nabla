// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_PATH_H_INCLUDED__
#define __NBL_PATH_H_INCLUDED__

#include "irrString.h"

#include "nbl/core/compile_config.h"

namespace nbl
{
namespace io
{
//! Type used for all file system related strings.
/** This type will transparently handle different file system encodings. */
typedef core::string<char> path;

//! Used in places where we identify objects by a filename, but don't actually work with the real filename
/** Irrlicht is internally not case-sensitive when it comes to names.
    Also this class is a first step towards support for correctly serializing renamed objects.
*
struct SNamedPath
{
	//! Constructor
	SNamedPath() {}

	//! Constructor
	SNamedPath(const path& p) : Path(p), InternalName( PathToName(p) )
	{
	}

	//! Is smaller comparator
	bool operator <(const SNamedPath& other) const
	{
		return InternalName < other.InternalName;
	}

	//! Set the path.
	void setPath(const path& p)
	{
		Path = p;
		InternalName = PathToName(p);
	}

	//! Get the path.
	const path& getPath() const
	{
		return Path;
	};

	//! Get the name which is used to identify the file.
	//! This string is similar to the names and filenames used before Irrlicht 1.7
	const path& getInternalName() const
	{
		return InternalName;
	}

	//! Implicit cast to io::path
	operator core::stringc() const
	{
		return core::stringc(getPath());
	}
	//! Implicit cast to io::path
	operator core::stringw() const
	{
		return core::stringw(getPath());
	}



	// convert the given path string to a name string.
	static inline path PathToName(const path& p)
	{
		path name(p);
		name.replace( '\\', '/' );
		name.make_lower();
		return name;
	}

private:
	path Path;
	path InternalName;
};
*/

struct SNamedPath  // : public AllocationOverrideDefault
{
    //! Constructor
    SNamedPath() {}

    //! Constructor
    SNamedPath(const path& p)
        : InternalName(PathToName(p))
    {
    }

    //! Is smaller comparator
    inline bool operator<(const SNamedPath& other) const
    {
        return InternalName < other.InternalName;
    }

    //! Set the path.
    inline void setPath(const path& p)
    {
        InternalName = PathToName(p);
    }

    //! Get the name which is used to identify the file.
    //! This string is similar to the names and filenames used before Irrlicht 1.7
    inline const path& getInternalName() const
    {
        return InternalName;
    }
    /*
	//! Implicit cast to io::path
	operator core::stringc() const
	{
		return getInternalName();
	}
	//! Implicit cast to io::path
	operator core::stringw() const
	{
		return core::stringw(getInternalName());
	}
*/

    // convert the given path string to a name string.
    static inline path PathToName(const path& p)
    {
        path name(p);
        //handleBackslashes(&name);
        name.replace('\\', '/');  //! On Linux just delete them
#ifndef _NBL_POSIX_API_
        name.make_lower();
#endif  // _NBL_POSIX_API_
        return name;
    }

private:
    path InternalName;
};

}  // io
}  // nbl

#endif
