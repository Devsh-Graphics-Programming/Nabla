#include "nbl/system/IFileArchive.h"
#include "nbl/system/IFileViewAllocator.h"
namespace nbl::system
{
	void IFileArchive::setFlagsVectorSize(size_t fileCount)
	{
		assert(!m_filesBuffer && !m_fileFlags);
		m_filesBuffer = (std::byte*)_NBL_ALIGNED_MALLOC(fileCount * SIZEOF_INNER_ARCHIVE_FILE, ALIGNOF_INNER_ARCHIVE_FILE);
		m_fileFlags = (std::atomic_flag*)_NBL_ALIGNED_MALLOC(fileCount * sizeof(std::atomic_flag), alignof(std::atomic_flag));
		for (int i = 0; i < fileCount; i++)
		{
			m_fileFlags[i].clear();
		}
		memset(m_filesBuffer, 0, fileCount * SIZEOF_INNER_ARCHIVE_FILE);
	}
}