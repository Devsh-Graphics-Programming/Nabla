// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_LOGGER_H_INCLUDED__
#define __NBL_I_LOGGER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include <string>

namespace nbl
{
//! Possible log levels.
//! When used has filter ELL_DEBUG means => log everything and ELL_NONE means => log (nearly) nothing.
//! When used to print logging information ELL_DEBUG will have lowest priority while ELL_NONE
//! messages are never filtered and always printed.
enum ELOG_LEVEL
{
    //! Used for printing information helpful in debugging
    ELL_DEBUG,

    //! Useful information to print. For example hardware infos or something started/stopped.
    ELL_INFORMATION,

    //! Warnings that something isn't as expected and can cause oddities
    ELL_WARNING,

    //! Something did go wrong.
    ELL_ERROR,

    //! Logs with ELL_NONE will never be filtered.
    //! And used as filter it will remove all logging except ELL_NONE messages.
    ELL_NONE
};

//! Interface for logging messages, warnings and errors
class ILogger : public virtual core::IReferenceCounted
{
public:
    //! Returns the current set log level.
    virtual ELOG_LEVEL getLogLevel() const = 0;

    //! Sets a new log level.
    /** With this value, texts which are sent to the logger are filtered
	out. For example setting this value to ELL_WARNING, only warnings and
	errors are printed out. Setting it to ELL_INFORMATION, which is the
	default setting, warnings, errors and informational texts are printed
	out.
	\param ll: new log level filter value. */
    virtual void setLogLevel(ELOG_LEVEL ll) = 0;

    //! Prints out a text into the log
    /** \param text: Text to print out.
	\param ll: Log level of the text. If the text is an error, set
	it to ELL_ERROR, if it is warning set it to ELL_WARNING, and if it
	is just an informational text, set it to ELL_INFORMATION. Texts are
	filtered with these levels. If you want to be a text displayed,
	independent on what level filter is set, use ELL_NONE. */
    virtual void log(const std::string& text, ELOG_LEVEL ll = ELL_INFORMATION) = 0;

    //! Prints out a text into the log
    /** \param text: Text to print out.
	\param hint: Additional info. This string is added after a " :" to the
	string.
	\param ll: Log level of the text. If the text is an error, set
	it to ELL_ERROR, if it is warning set it to ELL_WARNING, and if it
	is just an informational text, set it to ELL_INFORMATION. Texts are
	filtered with these levels. If you want to be a text displayed,
	independent on what level filter is set, use ELL_NONE. */
    virtual void log(const std::string& text, const std::string& hint, ELOG_LEVEL ll = ELL_INFORMATION) = 0;
    virtual void log(const std::string& text, const std::wstring& hint, ELOG_LEVEL ll = ELL_INFORMATION) = 0;

    //! Prints out a text into the log
    /** \param text: Text to print out.
	\param hint: Additional info. This string is added after a " :" to the
	string.
	\param ll: Log level of the text. If the text is an error, set
	it to ELL_ERROR, if it is warning set it to ELL_WARNING, and if it
	is just an informational text, set it to ELL_INFORMATION. Texts are
	filtered with these levels. If you want to be a text displayed,
	independent on what level filter is set, use ELL_NONE. */
    virtual void log(const std::wstring& text, const std::wstring& hint, ELOG_LEVEL ll = ELL_INFORMATION) = 0;

    //! Prints out a text into the log
    /** \param text: Text to print out.
	\param ll: Log level of the text. If the text is an error, set
	it to ELL_ERROR, if it is warning set it to ELL_WARNING, and if it
	is just an informational text, set it to ELL_INFORMATION. Texts are
	filtered with these levels. If you want to be a text displayed,
	independent on what level filter is set, use ELL_NONE. */
    virtual void log(const std::wstring& text, ELOG_LEVEL ll = ELL_INFORMATION) = 0;
};

}  // end namespace

#endif
