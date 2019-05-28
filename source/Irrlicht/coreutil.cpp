#include "coreutil.h"
#include "utf8/unchecked.h"
#include "ConvertUTF.h"
#include "FW_Mutex.h"


namespace std
{
    template <>
    struct hash<irr::core::LeakDebugger::StackTrace>
    {
        std::size_t operator()(const irr::core::LeakDebugger::StackTrace& k) const noexcept
        {
            using std::size_t;
            using std::hash;
            using std::string;

            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:
            size_t retval = 0;

            for (irr::core::vector<string>::const_iterator it=k.getTrace().begin(); it!=k.getTrace().end(); it++)
                retval ^= std::hash<string>()(*it) + 0x9e3779b9 + (retval << 6) + (retval >> 2);

            return retval;
        }
    };

}



#ifdef _IRR_DEBUG
#ifdef _IRR_COMPILE_WITH_X11_DEVICE_

#include <execinfo.h>
#include <libunwind.h>

#include <cxxabi.h>

#endif // _IRR_COMPILE_WITH_X11_DEVICE_
#endif // _IRR_DEBUG

namespace irr
{
namespace core
{

std::string WStringToUTF8String(const std::wstring& inString)
{
	std::string utf8line;
	utf8line.reserve(inString.length());

	utf8::unchecked::utf16to8(inString.begin(), inString.end(), back_inserter(utf8line));
	return utf8line;
}

std::wstring UTF8StringToWString(const std::string& inString)
{
	std::string::const_iterator end_it = utf8::find_invalid(inString.begin(), inString.end());

	std::wstring utf16line;
	utf16line.reserve(end_it-inString.begin());
	utf8::unchecked::utf8to16(inString.begin(), end_it, back_inserter(utf16line));

	return utf16line;
}

std::wstring UTF8StringToWString(const std::string& inString, uint32_t inReplacementforInvalid)
{
	std::string replacedStr;
	replacedStr.reserve(inString.size());
	utf8::unchecked::replace_invalid(inString.begin(), inString.end(), back_inserter(replacedStr), inReplacementforInvalid);

	std::wstring utf16line;
	utf16line.reserve(replacedStr.length());
	utf8::unchecked::utf8to16(replacedStr.begin(), replacedStr.end(), back_inserter(utf16line));

	return utf16line;
}

/*
template<> bool equalsIgnoreCase<std::string>(const std::string&, const std::string&);
template<> bool equalsIgnoreCase<std::wstring>(const std::wstring&, const std::wstring&);

template<> int32_t strcmpi<>(const T& str1, const T& str2)
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
T lastChar(const std::basic_string<T>& str1)
*/




core::vector<std::string> getBackTrace(void)
{
    core::vector<std::string> retval;
#ifdef _IRR_DEBUG
#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
/*
    void* funcAddrs[256];
    int callStackLen = backtrace(funcAddrs,256);
    retval.resize(callStackLen);

    char** strings = backtrace_symbols(funcAddrs,callStackLen);
    for (int i=0; i<callStackLen; i++)
        retval[i] = strings[i];

    free(strings);
*/

    unw_cursor_t cursor;
    unw_context_t context;

    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    size_t n=0;
    while ( unw_step(&cursor) )
    {
        unw_word_t ip, sp, off;

        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        unw_get_reg(&cursor, UNW_REG_SP, &sp);

        char symbol[256] = {"<unknown>"};
        char *name = symbol;

        if ( !unw_get_proc_name(&cursor, symbol, sizeof(symbol), &off) )
        {
            int status;
            if ( (name = abi::__cxa_demangle(symbol, NULL, NULL, &status)) == 0 )
                name = symbol;
        }

        char outBuff[16*1024];
        sprintf(outBuff,"#%-2d 0x%016" PRIxPTR " sp=0x%016" PRIxPTR " %s + 0x%" PRIxPTR "\n",
                            ++n,
                            static_cast<uintptr_t>(ip),
                            static_cast<uintptr_t>(sp),
                            name,
                            static_cast<uintptr_t>(off));
        retval.push_back(outBuff);

        if ( name != symbol )
            free(name);
    }
#endif
#endif
    return retval;
}



LeakDebugger::LeakDebugger(const std::string& nameOfDbgr) : name(nameOfDbgr)
{
}
LeakDebugger::~LeakDebugger()
{
}

void LeakDebugger::registerObj(const void* obj)
{
#ifdef _IRR_DEBUG
    std::lock_guard<std::mutex> lock(tsafer);

    core::unordered_map<const void*,StackTrace>::const_iterator found = tracker.find(obj);
    if (found!=tracker.end())
    {
        printf("BAD REFCOUNTING IN LEAK DEBUGGER %s with item %p \t Previous supposed alloc was:\n",name.c_str(),obj);

        std::ostringstream strm;
        found->second.printStackToOStream(strm);
        printf(strm.str().c_str());
    }
    tracker[obj] = getBackTrace();
#endif // _IRR_DEBUG
}

void LeakDebugger::deregisterObj(const void* obj)
{
#ifdef _IRR_DEBUG
    std::lock_guard<std::mutex> lock(tsafer);

    core::unordered_map<const void*,StackTrace>::const_iterator found = tracker.find(obj);
    if (found==tracker.end())
    {
        printf("LEAK DEBUGGER %s found DOUBLE FREE item %p \t Allocated from:\n",name.c_str(),obj);

        std::ostringstream strm;
        found->second.printStackToOStream(strm);
        printf(strm.str().c_str());
    }
    else
        tracker.erase(obj);
#endif // _IRR_DEBUG
}

void LeakDebugger::clearLeaks()
{
#ifdef _IRR_DEBUG
    std::lock_guard<std::mutex> lock(tsafer);
    tracker.clear();
#endif // _IRR_DEBUG
}

void LeakDebugger::dumpLeaks()
{
#ifdef _IRR_DEBUG
    core::unordered_multiset<StackTrace> epicCounter;

    std::lock_guard<std::mutex> lock(tsafer);
    {
        printf("Printing the leaks of %s\n\n",name.c_str());

        for (core::unordered_map<const void*,StackTrace>::iterator it=tracker.begin(); it!=tracker.end(); it++)
            epicCounter.insert(it->second);

        {
            core::unordered_multiset<StackTrace>::iterator it=epicCounter.begin();
            while (it!=epicCounter.end())
            {
                size_t occurences = epicCounter.count(*it);

                std::ostringstream strm;
                strm << "Number of Leak Occurrences: " << occurences << "\n";
                it->printStackToOStream(strm);
                printf(strm.str().c_str());

                for (size_t j=0; j<occurences; j++)
                    it++;
            }
        }
    }
#else
    printf("Object Leak Tracking Not Enabled, _IRR_DEBUG not defined during Irrlicht compilation!\n");
#endif // _IRR_DEBUG
}

}
}
