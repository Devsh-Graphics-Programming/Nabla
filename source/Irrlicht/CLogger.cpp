// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "coreutil.h"
#include "CLogger.h"

namespace irr
{

	CLogger::CLogger(IEventReceiver* r)
		: LogLevel(ELL_INFORMATION), Receiver(r)
	{
		#ifdef _IRR_DEBUG
		setDebugName("CLogger");
		#endif
	}

	//! Returns the current set log level.
	ELOG_LEVEL CLogger::getLogLevel() const
	{
		return LogLevel;
	}

	//! Sets a new log level.
	void CLogger::setLogLevel(ELOG_LEVEL ll)
	{
		LogLevel = ll;
	}


	template<typename T>
	inline std::string quickUTF8(const T& str);

	template<> inline std::string quickUTF8<std::wstring>(const std::wstring& str)
	{
	    return core::WStringToUTF8String(str);
	    /*
	std::string utf8line;
	utf8line.reserve(inString.length());

	utf8::unchecked::utf16to8(inString.begin(), inString.end(), back_inserter(utf8line));
	return utf8line;
        */
	}
	template<> inline std::string quickUTF8<std::string>(const std::string& str)
	{
	    return str;
	}

	template std::string quickUTF8<std::string>(const std::string&);
	template std::string quickUTF8<std::wstring>(const std::wstring&);

	//! Prints out a text into the log
	template<typename T1, typename T2>
	void CLogger::actualLog(const std::basic_string<T1>& text, const std::basic_string<T2>& hint, ELOG_LEVEL ll)
	{
		if (ll < LogLevel)
			return;

		std::string s = quickUTF8<std::basic_string<T1> >(text);
		s += ": ";
		s += quickUTF8<std::basic_string<T2> >(hint);
		log (s, ll);
	}

	//! Prints out a text into the log
	template<typename T>
	void CLogger::actualLog(const std::basic_string<T>& text, ELOG_LEVEL ll)
	{
		if (ll < LogLevel)
			return;

		std::string s = quickUTF8<std::basic_string<T> >(text);
		if (Receiver)
		{
			SEvent event;
			event.EventType = EET_LOG_TEXT_EVENT;
			event.LogEvent.Text = s.data();
			event.LogEvent.Level = ll;
			if (Receiver->OnEvent(event))
				return;
		}

		os::Printer::print(s.c_str());
	}

	template void CLogger::actualLog<char>(const std::string&,ELOG_LEVEL);
	template void CLogger::actualLog<wchar_t>(const std::wstring&,ELOG_LEVEL);

	template void CLogger::actualLog<char   ,char   >(const std::string&,const std::string&,ELOG_LEVEL);
	template void CLogger::actualLog<wchar_t,char   >(const std::wstring&,const std::string&,ELOG_LEVEL);
	template void CLogger::actualLog<char   ,wchar_t>(const std::string&,const std::wstring&,ELOG_LEVEL);
	template void CLogger::actualLog<wchar_t,wchar_t>(const std::wstring&,const std::wstring&,ELOG_LEVEL);

	//! Sets a new event receiver
	void CLogger::setReceiver(IEventReceiver* r)
	{
		Receiver = r;
	}


} // end namespace irr

