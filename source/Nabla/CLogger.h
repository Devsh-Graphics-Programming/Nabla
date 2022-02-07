// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_LOGGER_H_INCLUDED__
#define __NBL_C_LOGGER_H_INCLUDED__

#include "ILogger.h"
#include "os.h"
#include "IEventReceiver.h"

namespace nbl
{
//! Class for logging messages, warnings and errors to stdout
class CLogger : public ILogger
{
public:
    CLogger(IEventReceiver* r);

    //! Returns the current set log level.
    virtual ELOG_LEVEL getLogLevel() const;

    //! Sets a new log level.	virtual void setLogLevel(ELOG_LEVEL ll);
    virtual void setLogLevel(ELOG_LEVEL ll);

    //! Prints out a text into the log
    virtual void log(const std::string& text, ELOG_LEVEL ll = ELL_INFORMATION) { actualLog(text, ll); }

    //! Prints out a text into the log
    virtual void log(const std::wstring& text, ELOG_LEVEL ll = ELL_INFORMATION) { actualLog(text, ll); }

    //! Prints out a text into the log
    virtual void log(const std::string& text, const std::string& hint, ELOG_LEVEL ll = ELL_INFORMATION) { actualLog(text, hint, ll); }

    //! Prints out a text into the log
    virtual void log(const std::string& text, const std::wstring& hint, ELOG_LEVEL ll = ELL_INFORMATION) { actualLog(text, hint, ll); }

    //! Prints out a text into the log
    virtual void log(const std::wstring& text, const std::wstring& hint, ELOG_LEVEL ll = ELL_INFORMATION) { actualLog(text, hint, ll); }

    //! Sets a new event receiver
    void setReceiver(IEventReceiver* r);

private:
    //! Prints out a text into the log
    template<typename T>
    void actualLog(const std::basic_string<T>& text, ELOG_LEVEL ll = ELL_INFORMATION);

    //! Prints out a text into the log
    template<typename T1, typename T2>
    void actualLog(const std::basic_string<T1>& text, const std::basic_string<T2>& hint, ELOG_LEVEL ll = ELL_INFORMATION);

    ELOG_LEVEL LogLevel;
    IEventReceiver* Receiver;
};

}  // end namespace

#endif
