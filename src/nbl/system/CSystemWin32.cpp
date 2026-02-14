#include "nbl/system/CSystemWin32.h"
#include "nbl/system/CFileWin32.h"

using namespace nbl;
using namespace nbl::system;

#ifdef _NBL_PLATFORM_WINDOWS_
#include <powerbase.h>

//LOL the struct definition wasn't added to winapi headers do they ask to declare them yourself
typedef struct _PROCESSOR_POWER_INFORMATION {
    ULONG Number;
    ULONG MaxMhz;
    ULONG CurrentMhz;
    ULONG MhzLimit;
    ULONG MaxIdleState;
    ULONG CurrentIdleState;
} PROCESSOR_POWER_INFORMATION, * PPROCESSOR_POWER_INFORMATION;

ISystem::SystemInfo CSystemWin32::getSystemInfo() const
{
    SystemInfo info;

    // TODO: improve https://forums.codeguru.com/showthread.php?519933-Windows-SDK-How-to-get-the-processor-frequency
    LARGE_INTEGER speed;
    QueryPerformanceFrequency(&speed);
    info.cpuFrequencyHz = speed.QuadPart;
    
    MEMORYSTATUS memoryStatus;
    memoryStatus.dwLength = sizeof(MEMORYSTATUS);
    GlobalMemoryStatus(&memoryStatus);
    info.totalMemory = memoryStatus.dwTotalPhys;
    info.availableMemory = memoryStatus.dwAvailPhys;

    info.desktopResX = GetSystemMetrics(SM_CXSCREEN);
    info.desktopResY = GetSystemMetrics(SM_CYSCREEN);

    return info;
}


core::smart_refctd_ptr<ISystemFile> CSystemWin32::CCaller::createFile(const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags)
{
    const bool writeAccess = flags.value&IFile::ECF_WRITE;
	const DWORD fileAccess = ((flags.value&IFile::ECF_READ) ? FILE_GENERIC_READ:0)|(writeAccess ? FILE_GENERIC_WRITE:0);
	DWORD shareMode = FILE_SHARE_READ;
	if (flags.value & IFile::ECF_SHARE_READ_WRITE)
		shareMode |= FILE_SHARE_WRITE;
	if (flags.value & IFile::ECF_SHARE_DELETE)
		shareMode |= FILE_SHARE_DELETE;

	SECURITY_ATTRIBUTES secAttribs{ sizeof(SECURITY_ATTRIBUTES), nullptr, FALSE };
	
	system::path p = filename;
	if (p.is_absolute()) 
		p.make_preferred(); // Replace "/" separators with "\"

    // only write access should create new files if they don't exist
	const auto creationDisposition = writeAccess ? OPEN_ALWAYS : OPEN_EXISTING;
	HANDLE _native = CreateFileA(p.string().data(), fileAccess, shareMode, &secAttribs, creationDisposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (_native==INVALID_HANDLE_VALUE)
    {
        auto e = GetLastError();
        return nullptr;
    }

    HANDLE _fileMappingObj = nullptr;
    void* _mappedPtr = nullptr;
    if ((flags.value&IFile::ECF_MAPPABLE) && (flags.value&IFile::ECF_READ_WRITE))
    {
        /*
        TODO: should think of a better way to cope with the max size of a file mapping object (those two zeroes after `access`).
        For now it equals the size of a file so it'll work fine for archive reading, but if we try to
        write outside those boungs, things will go bad.
        */
        _fileMappingObj = CreateFileMappingA(_native,nullptr,writeAccess ? PAGE_READWRITE:PAGE_READONLY, 0, 0, nullptr);
        if (!_fileMappingObj)
        {
            flags.value &= ~(IFile::ECF_COHERENT | IFile::ECF_MAPPABLE);
        }
		else
        {
            switch (flags.value&IFile::ECF_READ_WRITE)
            {
                case IFile::ECF_READ:
                    _mappedPtr = MapViewOfFile(_fileMappingObj,FILE_MAP_READ,0,0,0);
                    break;
                case IFile::ECF_WRITE:
                    _mappedPtr = MapViewOfFile(_fileMappingObj,FILE_MAP_WRITE,0,0,0);
                    break;
                case IFile::ECF_READ_WRITE:
                    _mappedPtr = MapViewOfFile(_fileMappingObj,FILE_MAP_ALL_ACCESS,0,0,0);
                    break;
                default:
                    assert(false); // should never happen
                    break;
            }
            if (!_mappedPtr)
            {
                CloseHandle(_fileMappingObj);
                _fileMappingObj = nullptr;
                flags.value &= ~(IFile::ECF_COHERENT | IFile::ECF_MAPPABLE);
            }
		}
    }
    return core::make_smart_refctd_ptr<CFileWin32>(core::smart_refctd_ptr<ISystem>(m_system),path(filename),flags,_mappedPtr,_native,_fileMappingObj);
}

bool isDebuggerAttached()
{
   return IsDebuggerPresent();
}

#endif
