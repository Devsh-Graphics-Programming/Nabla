#ifndef _NBL_SYSTEM_CSYSTEMWIN32_H_INCLUDED_
#define _NBL_SYSTEM_CSYSTEMWIN32_H_INCLUDED_
#include "ISystem.h"
#include "CFileWin32.h"
namespace nbl::system
{
	
class CSystemCallerWin32 : public ISystem::ISystemCaller        
{
protected:
    ~CSystemCallerWin32() override = default;

public:
    core::smart_refctd_ptr<IFile> createFile(ISystem* sys, const std::filesystem::path& filename, IFile::E_CREATE_FLAGS flags) override final
    {
        return core::make_smart_refctd_ptr<CFileWin32>(filename, flags);
    }
    size_t read(IFile* file, void* buffer, size_t offset, size_t size) override final
    {
        return file->read(buffer, offset, size); 
    }
    size_t write(IFile* file, const void* buffer, size_t offset, size_t size) override final
    {
        return file->write(buffer, offset, size);
    }
    //TODO:
    bool invalidateMapping(IFile* file, size_t offset, size_t size) override final
    {
        assert(false);
        return true;
    }
    bool flushMapping(IFile* file, size_t offset, size_t size) override final
    {
        assert(false);
        return true;
    }
};
}

#endif