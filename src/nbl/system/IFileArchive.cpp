#include "nbl/system/IFileArchive.h"
#include "nbl/system/IFileViewAllocator.h"
namespace nbl::system
{
	void IFileArchive::setFlagsVectorSize(size_t fileCount)
	{
		m_filesBuffer = (std::byte*)_NBL_ALIGNED_MALLOC(fileCount * sizeof(CInnerArchiveFile<CPlainHeapAllocator>), alignof(CInnerArchiveFile<CPlainHeapAllocator>));
		m_fileFlags = (std::atomic_flag*)_NBL_ALIGNED_MALLOC(fileCount * sizeof(std::atomic_flag), alignof(std::atomic_flag));
		for (int i = 0; i < fileCount; i++)
		{
			m_fileFlags[i].clear();
		}
		memset(m_filesBuffer, 0, fileCount * sizeof(CInnerArchiveFile<CPlainHeapAllocator>));
	}
}