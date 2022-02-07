// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_C_LEAK_DEBUGGER_H_INCLUDED__
#define __NBL_CORE_C_LEAK_DEBUGGER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/system/compile_config.h"

#include <string>
#include <sstream>
#include "stddef.h"

namespace nbl
{
namespace core
{
core::vector<std::string> getBackTrace(void);

//! Utility class easing the process of finding memory leaks. Usable only in debug build. Thread-safe. No Windows implementation yet.
class NBL_FORCE_EBO CLeakDebugger : public AllocationOverrideDefault, public Uncopyable
{
    std::string name;

public:
    class StackTrace : public AllocationOverrideDefault
    {
        core::vector<std::string> stackTrace;

    public:
        //! Default constructor.
        StackTrace()
        {
        }

        //!
        StackTrace(const core::vector<std::string>& trc)
            : stackTrace(trc)
        {
        }

        const core::vector<std::string>& getTrace() const { return stackTrace; }

        /*
				//! Comparison operator. Needed for map/sorting.
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
                */

        //! Equality operator. Needed for unordered containers.
        bool operator==(const StackTrace& o) const
        {
            if(stackTrace.size() != o.stackTrace.size())
                return false;

            for(size_t i = 0; i < stackTrace.size(); i++)
            {
                if(stackTrace[i] != o.stackTrace[i])
                    return false;
            }
            return true;
        }

        //! Prints stack to given output stream.
        inline void printStackToOStream(std::ostringstream& strm) const
        {
            for(size_t i = 0; i < stackTrace.size(); i++)
            {
                for(size_t j = 0; j < i; j++)
                    strm << " ";

                strm << stackTrace[i] << "\n";
            }
            strm << "\n";
        }
    };

    CLeakDebugger(const std::string& nameOfDbgr);
    ~CLeakDebugger();

    void registerObj(const void* obj);

    void deregisterObj(const void* obj);

    void dumpLeaks();

    void clearLeaks();

private:
    std::mutex tsafer;
    core::unordered_map<const void*, StackTrace> tracker;
};

}  // end namespace core
}  // end namespace nbl

#endif