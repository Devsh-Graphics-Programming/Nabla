bool COSOperator::getProcessorSpeedMHz(uint32_t* MHz) const
{
#if defined(_NBL_WINDOWS_API_) && !defined(_WIN32_WCE )
	LONG Error;

	HKEY Key;
	Error = RegOpenKeyEx(HKEY_LOCAL_MACHINE,
			__TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
			0, KEY_READ, &Key);

	if(Error != ERROR_SUCCESS)
		return false;

	DWORD Speed = 0;
	DWORD Size = sizeof(Speed);
	Error = RegQueryValueEx(Key, __TEXT("~MHz"), NULL, NULL, (LPBYTE)&Speed, &Size);

	RegCloseKey(Key);

	if (Error != ERROR_SUCCESS)
		return false;
	else if (MHz)
		*MHz = Speed;
	return true;

#elif defined(_NBL_OSX_PLATFORM_)
	struct clockinfo CpuClock;
	size_t Size = sizeof(clockinfo);

	if (!sysctlbyname("kern.clockrate", &CpuClock, &Size, NULL, 0))
		return false;
	else if (MHz)
		*MHz = CpuClock.hz;
	return true;
#else
	// could probably be read from "/proc/cpuinfo" or "/proc/cpufreq"
    std::ifstream infile("/proc/cpuinfo",std::ios::in);

    std::string line;
    while (std::getline(infile, line))
    {
        if (!core::equalsIgnoreCaseSubStr<std::string>(line,0,"cpu mhz",0,7))
            continue;

        size_t firstPos = line.find_first_of("0123456789");
        if (firstPos==std::string::npos)
            continue;

        size_t dec = line.find('.',firstPos+1);
        if (dec!=std::string::npos)
        {
            std::istringstream strm(line.substr(firstPos,dec-firstPos));
            strm >> *MHz;
        }
        else
        {
            std::istringstream strm(line.substr(firstPos));
            strm >> *MHz;
        }

        return true;
    }

	return false;
#endif
}

bool COSOperator::getSystemMemory(uint32_t* Total, uint32_t* Avail) const
{
#if defined(_NBL_WINDOWS_API_)
	MEMORYSTATUS MemoryStatus;
	MemoryStatus.dwLength = sizeof(MEMORYSTATUS);

	// cannot fail
	GlobalMemoryStatus(&MemoryStatus);

	if (Total)
		*Total = (uint32_t)(MemoryStatus.dwTotalPhys>>10);
	if (Avail)
		*Avail = (uint32_t)(MemoryStatus.dwAvailPhys>>10);

	return true;

#elif defined(_NBL_POSIX_API_) && !defined(__FreeBSD__)
#if defined(_SC_PHYS_PAGES) && defined(_SC_AVPHYS_PAGES)
        long ps = sysconf(_SC_PAGESIZE);
        long pp = sysconf(_SC_PHYS_PAGES);
        long ap = sysconf(_SC_AVPHYS_PAGES);

	if ((ps==-1)||(pp==-1)||(ap==-1))
		return false;

	if (Total)
		*Total = (uint32_t)((ps*(long long)pp)>>10);
	if (Avail)
		*Avail = (uint32_t)((ps*(long long)ap)>>10);
	return true;
#else
	// TODO: implement for non-availablity of symbols/features
	return false;
#endif
#else
	// TODO: implement for OSX
	return false;
#endif
}