// Copyright (C) 2019 DevSH Graphics Programming Sp. z O.O.
// This file is part of the "IrrlichtBaW".
// For conditions of distribution and use, see LICENSE.md

#ifndef __C_LEAK_DEBUGGER_H_INCLUDED__
#define __C_LEAK_DEBUGGER_H_INCLUDED__

#include <string>
#include <sstream>
#include "stddef.h"
#include "irr/core/BaseClasses.h"
#include "irr/core/alloc/AlignedBase.h"
#include "irr/core/Types.h"

namespace irr
{
namespace core
{



core::vector<std::string> getBackTrace(void);

//! Utility class easing the process of finding memory leaks. Usable only in debug build. Thread-safe. No Windows implementation yet.
class CLeakDebugger : public AllocationOverrideDefault, public Uncopyable
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
                StackTrace(const core::vector<std::string>& trc) : stackTrace(trc)
                {
                }

                const core::vector<std::string>& getTrace() const {return stackTrace;}

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
                    if (stackTrace.size()!=o.stackTrace.size())
                        return false;

                    for (size_t i=0; i<stackTrace.size(); i++)
                    {
                        if (stackTrace[i]!=o.stackTrace[i])
                            return false;
                    }
                    return true;
                }

				//! Prints stack to given output stream.
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

		CLeakDebugger(const std::string& nameOfDbgr);
        ~CLeakDebugger();

        void registerObj(const void* obj);

        void deregisterObj(const void* obj);

        void dumpLeaks();

        void clearLeaks();
    private:
        std::mutex tsafer;
        core::unordered_map<const void*,StackTrace> tracker;
};

} // end namespace core
} // end namespace irr

#endif