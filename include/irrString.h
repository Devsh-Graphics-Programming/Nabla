// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_STRING_H_INCLUDED__
#define __NBL_STRING_H_INCLUDED__

#include <stdio.h>
#include <string>
#include <stdlib.h>

#include "nbl/core/alloc/aligned_allocator.h"
#include "nbl/core/alloc/AlignedBase.h"

namespace nbl
{
namespace core
{
//! Very simple string class with some useful features.
/** string<char> and string<wchar_t> both accept Unicode AND ASCII/Latin-1,
so you can assign Unicode to string<char> and ASCII/Latin-1 to string<wchar_t>
(and the other way round) if you want to.

However, note that the conversation between both is not done using any encoding.
This means that char strings are treated as ASCII/Latin-1, not UTF-8, and
are simply expanded to the equivalent wchar_t, while Unicode/wchar_t
characters are truncated to 8-bit ASCII/Latin-1 characters, discarding all
other information in the wchar_t.
*/

enum eLocaleID
{
    NBL_LOCALE_ANSI = 0,
    NBL_LOCALE_GERMAN = 1
};

static eLocaleID locale_current = NBL_LOCALE_ANSI;
static inline void locale_set(eLocaleID id)
{
    locale_current = id;
}

//! Returns a character converted to lower case
static inline uint32_t locale_lower(uint32_t x)
{
    switch(locale_current)
    {
        case NBL_LOCALE_GERMAN:
        case NBL_LOCALE_ANSI:
            break;
    }
    // ansi
    return x >= 'A' && x <= 'Z' ? x + 0x20 : x;
}

//! Returns a character converted to upper case
static inline uint32_t locale_upper(uint32_t x)
{
    switch(locale_current)
    {
        case NBL_LOCALE_GERMAN:
        case NBL_LOCALE_ANSI:
            break;
    }

    // ansi
    return x >= 'a' && x <= 'z' ? x + ('A' - 'a') : x;
}

template<typename T, typename TAlloc = aligned_allocator<T> >
class string
{
public:
    typedef T char_type;

    //! Default constructor
    string()
        : array(0), allocated(1), used(1)
    {
        array = allocator.allocate(1);  // new T[1];
        array[0] = 0;
    }

    //! Constructor
    string(const string<T, TAlloc>& other)
        : array(0), allocated(0), used(0)
    {
        *this = other;
    }

    //! Constructor from other string types
    template<class B, class A>
    string(const string<B, A>& other)
        : array(0), allocated(0), used(0)
    {
        *this = other;
    }

    //! Constructor for copying a string from a pointer with a given length
    template<class B>
    string(const B* const c, uint32_t length)
        : array(0), allocated(0), used(0)
    {
        if(!c)
        {
            // correctly init the string to an empty one
            *this = "";
            return;
        }

        allocated = used = length + 1;
        array = allocator.allocate(used);  // new T[used];

        for(uint32_t l = 0; l < length; ++l)
            array[l] = (T)c[l];

        array[length] = 0;
    }

    //! Constructor for unicode and ascii strings
    template<class B>
    string(const B* const c)
        : array(0), allocated(0), used(0)
    {
        *this = c;
    }

    //! Destructor
    ~string()
    {
        allocator.deallocate(array, allocated);  // delete [] array;
    }

    //! Assignment operator
    string<T, TAlloc>& operator=(const string<T, TAlloc>& other)
    {
        if(this == &other)
            return *this;

        used = other.size() + 1;
        if(used > allocated)
        {
            allocator.deallocate(array, allocated);  // delete [] array;
            allocated = used;
            array = allocator.allocate(used);  //new T[used];
        }

        const T* p = other.c_str();
        for(uint32_t i = 0; i < used; ++i, ++p)
            array[i] = *p;

        return *this;
    }

    //! Assignment operator for other string types
    template<class B, class A>
    string<T, TAlloc>& operator=(const string<B, A>& other)
    {
        *this = other.c_str();
        return *this;
    }

    //! Assignment operator for strings, ascii and unicode
    template<class B>
    string<T, TAlloc>& operator=(const B* const c)
    {
        if(!c)
        {
            if(!array)
            {
                array = allocator.allocate(1);  //new T[1];
                allocated = 1;
            }
            used = 1;
            array[0] = 0x0;
            return *this;
        }

        if((void*)c == (void*)array)
            return *this;

        uint32_t len = 0;
        const B* p = c;
        do
        {
            ++len;
        }
        while(*p++);

        // we'll keep the old string for a while, because the new
        // string could be a part of the current string.
        T* oldArray = array;

        used = len;
        if(used > allocated)
            array = allocator.allocate(used);  //new T[used];

        for(uint32_t l = 0; l < len; ++l)
            array[l] = (T)c[l];

        if(oldArray != array)
            allocator.deallocate(oldArray, allocated);  // delete [] oldArray;
        if(used > allocated)
            allocated = used;

        return *this;
    }

    //! Append operator for other strings
    string<T, TAlloc> operator+(const string<T, TAlloc>& other) const
    {
        string<T, TAlloc> str(*this);
        str.append(other);

        return str;
    }

    //! Append operator for strings, ascii and unicode
    template<class B>
    string<T, TAlloc> operator+(const B* const c) const
    {
        string<T, TAlloc> str(*this);
        str.append(c);

        return str;
    }

    //! Direct access operator
    T& operator[](const uint32_t index)
    {
        _NBL_DEBUG_BREAK_IF(index >= used)  // bad index
        return array[index];
    }

    //! Direct access operator
    const T& operator[](const uint32_t index) const
    {
        _NBL_DEBUG_BREAK_IF(index >= used)  // bad index
        return array[index];
    }

    //! Equality operator
    bool operator==(const T* const str) const
    {
        if(!str)
            return false;

        uint32_t i;
        for(i = 0; array[i] && str[i]; ++i)
            if(array[i] != str[i])
                return false;

        return (!array[i] && !str[i]);
    }

    //! Equality operator
    bool operator==(const string<T, TAlloc>& other) const
    {
        for(uint32_t i = 0; array[i] && other.array[i]; ++i)
            if(array[i] != other.array[i])
                return false;

        return used == other.used;
    }

    //! Is smaller comparator
    bool operator<(const string<T, TAlloc>& other) const
    {
        for(uint32_t i = 0; array[i] && other.array[i]; ++i)
        {
            const int32_t diff = array[i] - other.array[i];
            if(diff)
                return (diff < 0);
        }

        return (used < other.used);
    }

    //! Inequality operator
    bool operator!=(const T* const str) const
    {
        return !(*this == str);
    }

    //! Inequality operator
    bool operator!=(const string<T, TAlloc>& other) const
    {
        return !(*this == other);
    }

    //! Returns length of the string's content
    /** \return Length of the string's content in characters, excluding
	the trailing NUL. */
    uint32_t size() const
    {
        return used - 1;
    }

    void resize(const size_t& newSz)
    {
        reallocate(newSz + 1);
        used = newSz + 1;
    }

    //! Informs if the string is empty or not.
    //! \return True if the string is empty, false if not.
    bool empty() const
    {
        return (size() == 0);
    }

    //! Returns character string
    /** \return pointer to C-style NUL terminated string. */
    const T* c_str() const
    {
        return array;
    }

    //! Makes the string lower case.
    string<T, TAlloc>& make_lower()
    {
        for(uint32_t i = 0; array[i]; ++i)
            array[i] = locale_lower(array[i]);
        return *this;
    }

    //! Makes the string upper case.
    string<T, TAlloc>& make_upper()
    {
        for(uint32_t i = 0; array[i]; ++i)
            array[i] = locale_upper(array[i]);
        return *this;
    }

    //! Compares the strings ignoring case.
    /** \param other: Other string to compare.
	\return True if the strings are equal ignoring case. */
    bool equals_ignore_case(const string<T, TAlloc>& other) const
    {
        for(uint32_t i = 0; array[i] && other[i]; ++i)
            if(locale_lower(array[i]) != locale_lower(other[i]))
                return false;

        return used == other.used;
    }

    //! Compares the strings ignoring case.
    /** \param other: Other string to compare.
		\param sourcePos: where to start to compare in the string
	\return True if the strings are equal ignoring case. */
    bool equals_substring_ignore_case(const string<T, TAlloc>& other, const int32_t sourcePos = 0) const
    {
        if((uint32_t)sourcePos >= used)
            return false;

        uint32_t i;
        for(i = 0; array[sourcePos + i] && other[i]; ++i)
            if(locale_lower(array[sourcePos + i]) != locale_lower(other[i]))
                return false;

        return array[sourcePos + i] == 0 && other[i] == 0;
    }

    //! Compares the strings ignoring case.
    /** \param other: Other string to compare.
	\return True if this string is smaller ignoring case. */
    bool lower_ignore_case(const string<T, TAlloc>& other) const
    {
        for(uint32_t i = 0; array[i] && other.array[i]; ++i)
        {
            int32_t diff = (int32_t)locale_lower(array[i]) - (int32_t)locale_lower(other.array[i]);
            if(diff)
                return diff < 0;
        }

        return used < other.used;
    }

    //! compares the first n characters of the strings
    /** \param other Other string to compare.
	\param n Number of characters to compare
	\return True if the n first characters of both strings are equal. */
    bool equalsn(const string<T, TAlloc>& other, uint32_t n) const
    {
        uint32_t i;
        for(i = 0; array[i] && other[i] && i < n; ++i)
            if(array[i] != other[i])
                return false;

        // if one (or both) of the strings was smaller then they
        // are only equal if they have the same length
        return (i == n) || (used == other.used);
    }

    //! compares the first n characters of the strings
    /** \param str Other string to compare.
	\param n Number of characters to compare
	\return True if the n first characters of both strings are equal. */
    bool equalsn(const T* const str, uint32_t n) const
    {
        if(!str)
            return false;
        uint32_t i;
        for(i = 0; array[i] && str[i] && i < n; ++i)
            if(array[i] != str[i])
                return false;

        // if one (or both) of the strings was smaller then they
        // are only equal if they have the same length
        return (i == n) || (array[i] == 0 && str[i] == 0);
    }

    //! Appends a character to this string
    /** \param character: Character to append. */
    string<T, TAlloc>& append(T character)
    {
        if(used + 1 > allocated)
            reallocate(used + 1);

        ++used;

        array[used - 2] = character;
        array[used - 1] = 0;

        return *this;
    }

    //! Appends a char string to this string
    /** \param other: Char string to append. */
    /** \param length: The length of the string to append. */
    string<T, TAlloc>& append(const T* const other, uint32_t length = 0xffffffff)
    {
        if(!other)
            return *this;

        uint32_t len = 0;
        const T* p = other;
        while(*p)
        {
            ++len;
            ++p;
        }
        if(len > length)
            len = length;

        if(used + len > allocated)
            reallocate(used + len);

        --used;
        ++len;

        for(uint32_t l = 0; l < len; ++l)
            array[l + used] = *(other + l);

        used += len;

        return *this;
    }

    //! Appends a string to this string
    /** \param other: String to append. */
    string<T, TAlloc>& append(const string<T, TAlloc>& other)
    {
        if(other.size() == 0)
            return *this;

        --used;
        uint32_t len = other.size() + 1;

        if(used + len > allocated)
            reallocate(used + len);

        for(uint32_t l = 0; l < len; ++l)
            array[used + l] = other[l];

        used += len;

        return *this;
    }

    //! Appends a string of the length l to this string.
    /** \param other: other String to append to this string.
	\param length: How much characters of the other string to add to this one. */
    string<T, TAlloc>& append(const string<T, TAlloc>& other, uint32_t length)
    {
        if(other.size() == 0)
            return *this;

        if(other.size() < length)
        {
            append(other);
            return *this;
        }

        if(used + length > allocated)
            reallocate(used + length);

        --used;

        for(uint32_t l = 0; l < length; ++l)
            array[l + used] = other[l];
        used += length;

        // ensure proper termination
        array[used] = 0;
        ++used;

        return *this;
    }

    //! Reserves some memory.
    /** \param count: Amount of characters to reserve. */
    void reserve(uint32_t count)
    {
        if(count < allocated)
            return;

        reallocate(count);
    }

    //! finds first occurrence of character in string
    /** \param c: Character to search for.
	\return Position where the character has been found,
	or -1 if not found. */
    int32_t findFirst(T c) const
    {
        for(uint32_t i = 0; i < used - 1; ++i)
            if(array[i] == c)
                return i;

        return -1;
    }

    //! finds first occurrence of a character of a list in string
    /** \param c: List of characters to find. For example if the method
	should find the first occurrence of 'a' or 'b', this parameter should be "ab".
	\param count: Amount of characters in the list. Usually,
	this should be strlen(c)
	\return Position where one of the characters has been found,
	or -1 if not found. */
    int32_t findFirstChar(const T* const c, uint32_t count = 1) const
    {
        if(!c || !count)
            return -1;

        for(uint32_t i = 0; i < used - 1; ++i)
            for(uint32_t j = 0; j < count; ++j)
                if(array[i] == c[j])
                    return i;

        return -1;
    }

    //! Finds first position of a character not in a given list.
    /** \param c: List of characters not to find. For example if the method
	should find the first occurrence of a character not 'a' or 'b', this parameter should be "ab".
	\param count: Amount of characters in the list. Usually,
	this should be strlen(c)
	\return Position where the character has been found,
	or -1 if not found. */
    template<class B>
    int32_t findFirstCharNotInList(const B* const c, uint32_t count = 1) const
    {
        if(!c || !count)
            return -1;

        for(uint32_t i = 0; i < used - 1; ++i)
        {
            uint32_t j;
            for(j = 0; j < count; ++j)
                if(array[i] == c[j])
                    break;

            if(j == count)
                return i;
        }

        return -1;
    }

    //! Finds last position of a character not in a given list.
    /** \param c: List of characters not to find. For example if the method
	should find the first occurrence of a character not 'a' or 'b', this parameter should be "ab".
	\param count: Amount of characters in the list. Usually,
	this should be strlen(c)
	\return Position where the character has been found,
	or -1 if not found. */
    template<class B>
    int32_t findLastCharNotInList(const B* const c, uint32_t count = 1) const
    {
        if(!c || !count)
            return -1;

        for(int32_t i = (int32_t)(used - 2); i >= 0; --i)
        {
            uint32_t j;
            for(j = 0; j < count; ++j)
                if(array[i] == c[j])
                    break;

            if(j == count)
                return i;
        }

        return -1;
    }

    //! finds next occurrence of character in string
    /** \param c: Character to search for.
	\param startPos: Position in string to start searching.
	\return Position where the character has been found,
	or -1 if not found. */
    int32_t findNext(T c, uint32_t startPos) const
    {
        for(uint32_t i = startPos; i < used - 1; ++i)
            if(array[i] == c)
                return i;

        return -1;
    }

    //! finds last occurrence of character in string
    /** \param c: Character to search for.
	\param start: start to search reverse ( default = -1, on end )
	\return Position where the character has been found,
	or -1 if not found. */
    int32_t findLast(T c, int32_t start = -1) const
    {
        start = core::clamp(start < 0 ? (int32_t)(used)-2 : start, 0, (int32_t)(used)-2);
        for(int32_t i = start; i >= 0; --i)
            if(array[i] == c)
                return i;

        return -1;
    }

    //! finds last occurrence of a character of a list in string
    /** \param c: List of strings to find. For example if the method
	should find the last occurrence of 'a' or 'b', this parameter should be "ab".
	\param count: Amount of characters in the list. Usually,
	this should be strlen(c)
	\return Position where one of the characters has been found,
	or -1 if not found. */
    int32_t findLastChar(const T* const c, uint32_t count = 1) const
    {
        if(!c || !count)
            return -1;

        for(int32_t i = (int32_t)used - 2; i >= 0; --i)
            for(uint32_t j = 0; j < count; ++j)
                if(array[i] == c[j])
                    return i;

        return -1;
    }

    //! finds another string in this string
    /** \param str: Another string
	\param start: Start position of the search
	\return Positions where the string has been found,
	or -1 if not found. */
    template<class B>
    int32_t find(const B* const str, const uint32_t start = 0) const
    {
        if(str && *str)
        {
            uint32_t len = 0;

            while(str[len])
                ++len;

            if(len > used - 1)
                return -1;

            for(uint32_t i = start; i < used - len; ++i)
            {
                uint32_t j = 0;

                while(str[j] && array[i + j] == str[j])
                    ++j;

                if(!str[j])
                    return i;
            }
        }

        return -1;
    }

    //! Returns a substring
    /** \param begin Start of substring.
	\param length Length of substring.
	\param make_lower_in copy only lower case */
    string<T> subString(uint32_t begin, int32_t length, bool make_lower_in = false) const
    {
        // if start after string
        // or no proper substring length
        if((length <= 0) || (begin >= size()))
            return string<T>("");
        // clamp length to maximal value
        if((length + begin) > size())
            length = size() - begin;

        string<T> o;
        o.reserve(length + 1);

        int32_t i;
        if(!make_lower_in)
        {
            for(i = 0; i < length; ++i)
                o.array[i] = array[i + begin];
        }
        else
        {
            for(i = 0; i < length; ++i)
                o.array[i] = locale_lower(array[i + begin]);
        }

        o.array[length] = 0;
        o.used = length + 1;

        return o;
    }

    //! Appends a char string to this string
    /** \param c Char string to append. */
    string<T, TAlloc>& operator+=(const T* const c)
    {
        append(c);
        return *this;
    }

    //! Appends a string to this string
    /** \param other String to append. */
    string<T, TAlloc>& operator+=(const string<T, TAlloc>& other)
    {
        append(other);
        return *this;
    }

    //! Replaces all characters of a special type with another one
    /** \param toReplace Character to replace.
	\param replaceWith Character replacing the old one. */
    string<T, TAlloc>& replace(T toReplace, T replaceWith)
    {
        for(uint32_t i = 0; i < used - 1; ++i)
            if(array[i] == toReplace)
                array[i] = replaceWith;
        return *this;
    }

    //! Replaces all instances of a string with another one.
    /** \param toReplace The string to replace.
	\param replaceWith The string replacing the old one. */
    string<T, TAlloc>& replace(const string<T, TAlloc>& toReplace, const string<T, TAlloc>& replaceWith)
    {
        if(toReplace.size() == 0)
            return *this;

        const T* other = toReplace.c_str();
        const T* replacePtr = replaceWith.c_str();
        const uint32_t other_size = toReplace.size();
        const uint32_t replace_size = replaceWith.size();

        // Determine the delta.  The algorithm will change depending on the delta.
        int32_t delta = replace_size - other_size;

        // A character for character replace.  The string will not shrink or grow.
        if(delta == 0)
        {
            int32_t pos = 0;
            while((pos = find(other, pos)) != -1)
            {
                for(uint32_t i = 0; i < replace_size; ++i)
                    array[pos + i] = replacePtr[i];
                ++pos;
            }
            return *this;
        }

        // We are going to be removing some characters.  The string will shrink.
        if(delta < 0)
        {
            uint32_t i = 0;
            for(uint32_t pos = 0; pos < used; ++i, ++pos)
            {
                // Is this potentially a match?
                if(array[pos] == *other)
                {
                    // Check to see if we have a match.
                    uint32_t j;
                    for(j = 0; j < other_size; ++j)
                    {
                        if(array[pos + j] != other[j])
                            break;
                    }

                    // If we have a match, replace characters.
                    if(j == other_size)
                    {
                        for(j = 0; j < replace_size; ++j)
                            array[i + j] = replacePtr[j];
                        i += replace_size - 1;
                        pos += other_size - 1;
                        continue;
                    }
                }

                // No match found, just copy characters.
                array[i] = array[pos];
            }
            array[i - 1] = 0;
            used = i;

            return *this;
        }

        // We are going to be adding characters, so the string size will increase.
        // Count the number of times toReplace exists in the string so we can allocate the new size.
        uint32_t find_count = 0;
        int32_t pos = 0;
        while((pos = find(other, pos)) != -1)
        {
            ++find_count;
            ++pos;
        }

        // Re-allocate the string now, if needed.
        uint32_t len = delta * find_count;
        if(used + len > allocated)
            reallocate(used + len);

        // Start replacing.
        pos = 0;
        while((pos = find(other, pos)) != -1)
        {
            T* start = array + pos + other_size - 1;
            T* ptr = array + used - 1;
            T* end = array + delta + used - 1;

            // Shift characters to make room for the string.
            while(ptr != start)
            {
                *end = *ptr;
                --ptr;
                --end;
            }

            // Add the new string now.
            for(uint32_t i = 0; i < replace_size; ++i)
                array[pos + i] = replacePtr[i];

            pos += replace_size;
            used += delta;
        }

        return *this;
    }

    //! Removes characters from a string.
    /** \param c: Character to remove. */
    string<T, TAlloc>& remove(T c)
    {
        uint32_t pos = 0;
        uint32_t found = 0;
        for(uint32_t i = 0; i < used - 1; ++i)
        {
            if(array[i] == c)
            {
                ++found;
                continue;
            }

            array[pos++] = array[i];
        }
        used -= found;
        array[used - 1] = 0;
        return *this;
    }

    //! Removes a string from the string.
    /** \param toRemove: String to remove. */
    string<T, TAlloc>& remove(const string<T, TAlloc>& toRemove)
    {
        uint32_t sizeToRemove = toRemove.size();
        if(sizeToRemove == 0)
            return *this;
        uint32_t pos = 0;
        uint32_t found = 0;
        for(uint32_t i = 0; i < used - 1; ++i)
        {
            uint32_t j = 0;
            while(j < sizeToRemove)
            {
                if(array[i + j] != toRemove[j])
                    break;
                ++j;
            }
            if(j == sizeToRemove)
            {
                found += sizeToRemove;
                i += sizeToRemove - 1;
                continue;
            }

            array[pos++] = array[i];
        }
        used -= found;
        array[used - 1] = 0;
        return *this;
    }

    //! Removes characters from a string.
    /** \param characters: Characters to remove. */
    string<T, TAlloc>& removeChars(const string<T, TAlloc>& characters)
    {
        if(characters.size() == 0)
            return *this;

        uint32_t pos = 0;
        uint32_t found = 0;
        for(uint32_t i = 0; i < used - 1; ++i)
        {
            // Don't use characters.findFirst as it finds the \0,
            // causing used to become incorrect.
            bool docontinue = false;
            for(uint32_t j = 0; j < characters.size(); ++j)
            {
                if(characters[j] == array[i])
                {
                    ++found;
                    docontinue = true;
                    break;
                }
            }
            if(docontinue)
                continue;

            array[pos++] = array[i];
        }
        used -= found;
        array[used - 1] = 0;

        return *this;
    }

    //! Trims the string.
    /** Removes the specified characters (by default, Latin-1 whitespace)
	from the begining and the end of the string. */
    string<T, TAlloc>& trim(const string<T, TAlloc>& whitespace = " \t\n\r")
    {
        // find start and end of the substring without the specified characters
        const int32_t begin = findFirstCharNotInList(whitespace.c_str(), whitespace.used);
        if(begin == -1)
            return (*this = "");

        const int32_t end = findLastCharNotInList(whitespace.c_str(), whitespace.used);

        return (*this = subString(begin, (end + 1) - begin));
    }

    //! Erases a character from the string.
    /** May be slow, because all elements
	following after the erased element have to be copied.
	\param index: Index of element to be erased. */
    string<T, TAlloc>& erase(uint32_t index)
    {
        _NBL_DEBUG_BREAK_IF(index >= used)  // access violation

        for(uint32_t i = index + 1; i < used; ++i)
            array[i - 1] = array[i];

        --used;
        return *this;
    }

    //! verify the existing string.
    string<T, TAlloc>& validate()
    {
        // terminate on existing null
        for(uint32_t i = 0; i < allocated; ++i)
        {
            if(array[i] == 0)
            {
                used = i + 1;
                return *this;
            }
        }

        // terminate
        if(allocated > 0)
        {
            used = allocated;
            array[used - 1] = 0;
        }
        else
        {
            used = 0;
        }

        return *this;
    }

    //! gets the last char of a string or null
    T lastChar() const
    {
        return used > 1 ? array[used - 2] : 0;
    }

    //! split string into parts.
    /** This method will split a string at certain delimiter characters
	into the container passed in as reference. The type of the container
	has to be given as template parameter. It must provide a push_back and
	a size method.
	\param ret The result container
	\param c C-style string of delimiter characters
	\param count Number of delimiter characters
	\param ignoreEmptyTokens Flag to avoid empty substrings in the result
	container. If two delimiters occur without a character in between, an
	empty substring would be placed in the result. If this flag is set,
	only non-empty strings are stored.
	\param keepSeparators Flag which allows to add the separator to the
	result string. If this flag is true, the concatenation of the
	substrings results in the original string. Otherwise, only the
	characters between the delimiters are returned.
	\return The number of resulting substrings
	*/
    template<class container>
    uint32_t split(container& ret, const T* const c, uint32_t count = 1, bool ignoreEmptyTokens = true, bool keepSeparators = false) const
    {
        if(!c)
            return 0;

        const uint32_t oldSize = ret.size();
        uint32_t lastpos = 0;
        bool lastWasSeparator = false;
        for(uint32_t i = 0; i < used; ++i)
        {
            bool foundSeparator = false;
            for(uint32_t j = 0; j < count; ++j)
            {
                if(array[i] == c[j])
                {
                    if((!ignoreEmptyTokens || i - lastpos != 0) &&
                        !lastWasSeparator)
                        ret.push_back(string<T, TAlloc>(&array[lastpos], i - lastpos));
                    foundSeparator = true;
                    lastpos = (keepSeparators ? i : i + 1);
                    break;
                }
            }
            lastWasSeparator = foundSeparator;
        }
        if((used - 1) > lastpos)
            ret.push_back(string<T, TAlloc>(&array[lastpos], (used - 1) - lastpos));
        return ret.size() - oldSize;
    }

private:
    //! Reallocate the array, make it bigger or smaller
    void reallocate(uint32_t new_size)
    {
        T* old_array = array;

        array = allocator.allocate(new_size);  //new T[new_size];
        allocated = new_size;

        uint32_t amount = used < new_size ? used : new_size;
        for(uint32_t i = 0; i < amount; ++i)
            array[i] = old_array[i];

        if(allocated < used)
            used = allocated;

        allocator.deallocate(old_array, allocated);  // delete [] old_array;
    }

    //--- member variables

    T* array;
    uint32_t allocated;
    uint32_t used;
    TAlloc allocator;
};

//! Typedef for character strings
typedef string<char> stringc;

//! Typedef for wide character strings
typedef string<wchar_t> stringw;

}  // end namespace core
}  // end namespace nbl

#endif
