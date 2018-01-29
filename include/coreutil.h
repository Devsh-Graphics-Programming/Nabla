// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_CORE_UTIL_H_INCLUDED__
#define __IRR_CORE_UTIL_H_INCLUDED__

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include "stddef.h"
#include "string.h"
#include "irrString.h"
#include "path.h"

class FW_Mutex;

namespace irr
{
namespace core
{

/*! \file coreutil.h
	\brief File containing useful basic utility functions
*/

template<class T>
inline void findAndReplaceAll(T& source, T const& findStr, T const& replaceStr)
{
    for(size_t i = 0; (i = source.find(findStr, i)) != std::string::npos;)
    {
        source.replace(i, findStr.length(), replaceStr);
        i += replaceStr.length();
    }
}

template<class T>
inline void findAndReplaceAll(std::basic_string<T>& source, const T* findStr, const T* replaceStr)
{
    findAndReplaceAll(source,std::basic_string<T>(findStr),std::basic_string<T>(replaceStr));
}


template<typename T>
inline bool equalsIgnoreCaseSubStr(const T& str1, const size_t& pos1, const T& str2, const size_t& pos2, const size_t& len)
{
    if (str1.begin()+len+pos1>str1.end())
        return false;
    if (str2.begin()+len+pos2>str2.end())
        return false;

    for (typename T::const_iterator c1 = str1.begin()+pos1, c2 = str2.begin()+pos2; c1 != str1.begin()+pos1+len; ++c1, ++c2)
    {
        if (tolower(*c1) != tolower(*c2))
            return false;
    }
    return true;
}

template<typename T>
inline bool equalsIgnoreCase(const T& str1, const T& str2)
{
    if (str1.size() != str2.size())
        return false;

    return equalsIgnoreCaseSubStr<T>(str1,0,str2,0,str2.size());
}


template<typename T>
inline int32_t strcmpi(const T& str1, const T& str2)
{
    if (str1.size()!=str2.size())
        return str1.size()-str2.size();

    for (typename T::const_iterator c1 = str1.begin(), c2 = str2.begin(); c1 != str1.end(); ++c1, ++c2)
    {
        int32_t val1 = tolower(*c1);
        int32_t val2 = tolower(*c2);
        if (val1 != val2)
            return val1-val2;
    }
    return 0;
}

template<typename T>
inline T lastChar(const std::basic_string<T>& str1)
{
    if (str1.size())
    {
        return *(str1.end()-1);
    }
    return 0;
}


extern std::string WStringToUTF8String(const std::wstring& inString);

extern std::wstring UTF8StringToWString(const std::string& inString);

extern std::wstring UTF8StringToWString(const std::string& inString, uint32_t inReplacementforInvalid);

// ----------- some basic quite often used string functions -----------------

//! search if a filename has a proper extension
inline int32_t isFileExtension (	const io::path& filename,
								const io::path& ext0,
								const io::path& ext1,
								const io::path& ext2)
{
	int32_t extPos = filename.findLast ( '.' );
	if ( extPos < 0 )
		return 0;

	extPos += 1;
	if ( filename.equals_substring_ignore_case ( ext0, extPos ) ) return 1;
	if ( filename.equals_substring_ignore_case ( ext1, extPos ) ) return 2;
	if ( filename.equals_substring_ignore_case ( ext2, extPos ) ) return 3;
	return 0;
}

//! search if a filename has a proper extension
inline bool hasFileExtension (	const io::path& filename,
								const io::path& ext0,
								const io::path& ext1 = "",
								const io::path& ext2 = "")
{
	return isFileExtension ( filename, ext0, ext1, ext2 ) > 0;
}

//! cut the filename extension from a source file path and store it in a dest file path
inline io::path& cutFilenameExtension ( io::path &dest, const io::path &source )
{
	int32_t endPos = source.findLast ( '.' );
	dest = source.subString ( 0, endPos < 0 ? source.size () : endPos );
	return dest;
}

//! get the filename extension from a file path
inline io::path& getFileNameExtension ( io::path &dest, const io::path &source )
{
	int32_t endPos = source.findLast ( '.' );
	if ( endPos < 0 )
		dest = "";
	else
		dest = source.subString ( endPos, source.size () );
	return dest;
}

//! delete path from filename
inline io::path& deletePathFromFilename(io::path& filename)
{
	// delete path from filename
	const char* s = filename.c_str();
	const char* p = s + filename.size();

	// search for path separator or beginning
	while ( *p != '/' && *p != '\\' && p != s ) //! On Linux just delete them
		p--;

	if ( p != s )
	{
		++p;
		filename = p;
	}
	return filename;
}

//! trim paths
inline io::path& deletePathFromPath(io::path& filename, int32_t pathCount)
{
	// delete path from filename
	int32_t i = filename.size();

	// search for path separator or beginning
	while ( i>=0 )
	{
		if ( filename[i] == '/' || filename[i] == '\\' ) //! On Linux just delete the '\\' from path
		{
			if ( --pathCount <= 0 )
				break;
		}
		--i;
	}

	if ( i>0 )
	{
		filename [ i + 1 ] = 0;
		filename.validate();
	}
	else
		filename="";
	return filename;
}

//! looks if file is in the same directory of path. returns offset of directory.
//! 0 means in same directory. 1 means file is direct child of path
inline int32_t isInSameDirectory ( const io::path& path, const io::path& file )
{
	int32_t subA = 0;
	int32_t subB = 0;
	int32_t pos;

	if ( path.size() && !path.equalsn ( file, path.size() ) )
		return -1;

	pos = 0;
	while ( (pos = path.findNext ( '/', pos )) >= 0 )
	{
		subA += 1;
		pos += 1;
	}

	pos = 0;
	while ( (pos = file.findNext ( '/', pos )) >= 0 )
	{
		subB += 1;
		pos += 1;
	}

	return subB - subA;
}

// expect change soon
inline void handleBackslashes(io::path* inout)
{
    inout->replace('\\' , '/'); //! On Linux just delete them
}

// splits a path into components
inline void splitFilename(const io::path &name, io::path* path=0,
		io::path* filename=0, io::path* extension=0, bool make_lower=false)
{
	int32_t i = name.size();
	int32_t extpos = i;

	// search for path separator or beginning
	while ( i >= 0 )
	{
		if ( name[i] == '.' )
		{
			extpos = i;
			if ( extension )
				*extension = name.subString ( extpos + 1, name.size() - (extpos + 1), make_lower );
		}
		else
		if ( name[i] == '/' || name[i] == '\\' ) //! On Linux just delete them
		{
			if ( filename )
				*filename = name.subString ( i + 1, extpos - (i + 1), make_lower );
			if ( path )
			{
				*path = name.subString ( 0, i + 1, make_lower );
				handleBackslashes(path);
			}
			return;
		}
		i -= 1;
	}
	if ( filename )
		*filename = name.subString ( 0, extpos, make_lower );
}

//! some standard function ( to remove dependencies )
#undef isdigit
#undef isspace
#undef isupper
inline int32_t isdigit(int32_t c) { return c >= '0' && c <= '9'; }
inline int32_t isspace(int32_t c) { return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' || c == '\v'; }
inline int32_t isupper(int32_t c) { return c >= 'A' && c <= 'Z'; }




std::vector<std::string> getBackTrace(void);





/*
   xxHash256 - A fast checksum algorithm
   Copyright (C) 2012, Yann Collet & Maciej Adamczyk.
   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

************************************************************************
   This file contains a super-fast hash function for checksumming
   purposes, designed for large (1 KB++) inputs.
   Known limitations:
    * they use 64-bit math and will not be so fast on 32-bit machines
    * on architectures that disallow unaligned memory access, the input
      must be 8-byte aligned. Aligning it on other architectures
      is not needed, but will improve performance.
    * it produces different results on big and small endian.

   Changelog:
    v0: initial release
    v1: the strong hash is faster and stronger, it obsoletes the fast one
*/


inline void XXHash_256(const void* input, size_t len, uint64_t* out)
{
//**************************************
// Macros
//**************************************
#define _rotl(x,r) ((x << r) | (x >> (64 - r)))

//**************************************
// Constants
//**************************************
    const uint64_t PRIME = 11400714819323198393ULL;

    const uint8_t* p = (uint8_t*)input;
    const uint8_t* const bEnd = p + len;
    uint64_t v1 = len * PRIME;
    uint64_t v2 = v1;
    uint64_t v3 = v1;
    uint64_t v4 = v1;

    const size_t big_loop_step = 4 * 4 * sizeof(uint64_t);
    const size_t small_loop_step = 4 * sizeof(uint64_t);
    // Set the big loop limit early enough, so the well-mixing small loop can be executed twice after it
    const uint8_t* const big_loop_limit   = bEnd - big_loop_step - 2 * small_loop_step;
    const uint8_t* const small_loop_limit = bEnd - small_loop_step;

    while (p < big_loop_limit)
    {
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v1 += v2 *= PRIME;
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 += v3 *= PRIME;
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 += v4 *= PRIME;
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 += v1 *= PRIME;
    }

    while (p < small_loop_limit)
    {
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 += v1 *= PRIME;
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 += v2 *= PRIME;
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 += v3 *= PRIME;
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v1 += v4 *= PRIME;
    }
#undef _rotl
    memcpy(out, p, bEnd - p);

    out[0] += v1;
    out[1] += v2;
    out[2] += v3;
    out[3] += v4;
}

class LeakDebugger
{
        std::string name;
    public:
        class StackTrace
        {
                std::vector<std::string> stackTrace;
            public:
                StackTrace()
                {
                }

                StackTrace(const std::vector<std::string>& trc) : stackTrace(trc)
                {
                }

                const std::vector<std::string>& getTrace() const {return stackTrace;}

                bool operator<(const StackTrace& o) const
                {
                    if (stackTrace.size()<o.stackTrace.size())
                        return true;
                    else if (stackTrace.size()==o.stackTrace.size())
                    {
                        for (size_t i=0; i<stackTrace.size(); i++)
                        {
                            if (stackTrace[i]==o.stackTrace[i])
                                continue;

                            return stackTrace[i]<o.stackTrace[i];
                        }
                        return false;
                    }
                    else
                        return false;
                }

                inline void printStackToOStream(std::ostringstream& strm) const
                {
                    for (size_t i=0; i<stackTrace.size(); i++)
                    {
                        for (size_t j=0; j<i; j++)
                            strm << " ";

                        strm << stackTrace[i] << "\n";
                    }
                    strm << "\n";
                }
        };

        LeakDebugger(const std::string& nameOfDbgr);
        ~LeakDebugger();

        void registerObj(const void* obj);

        void deregisterObj(const void* obj);

        void dumpLeaks();
    private:
        FW_Mutex* tsafer;
        std::map<const void*,StackTrace> tracker;
};

} // end namespace core
} // end namespace irr

#endif
