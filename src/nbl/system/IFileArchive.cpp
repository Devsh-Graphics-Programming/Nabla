#include "nbl/system/IFileArchive.h"

#include "nbl/system/IFile.h"

using namespace nbl;
using namespace nbl::system;


IFileArchive::SFileList IFileArchive::listAssets(path pathRelativeToArchive) const
{
		auto trimmedList = listAssets();
		{
			// remove trailing slash chars from pathRelativeToArchive
			auto pathstr = pathRelativeToArchive.string();
			while (pathstr.ends_with("/") || pathstr.ends_with("\\"))
			{
				pathstr= pathstr.substr(0, pathstr.length() - 1);
			}
			pathRelativeToArchive = pathstr;

			auto begin = trimmedList.m_data->begin();
			auto end = trimmedList.m_data->end();
			auto real_end = trimmedList.m_range.end();

			// finding paths with pathRelativeToArchive prefixes
			// SEntry::operator> is equivalent to comparing strings lexicographically, in other words compares ascii codes of chars in paths, and shorter path is always lesser 
			// std::lower_bound finds first element >= SEntry arg
			// 
			// we want to find all matches that meet this criteria (> is lexicographical string comparing)
			// pathRelativeToArchive < match < pathRelativeToArchive+1
			//
			// lower bound
			// by appending slash and char with code 1, we essentially make sure that all matches will be lexicographically greater, and start with pathRelativeToArchive/
			auto lower = std::lower_bound(begin,end, SFileList::SEntry{pathRelativeToArchive/"\1"});
			if (lower!=end)
			{
				// upper bound
				// we cannot do std::upper_bound, because any value longer than pathRelativeToArchive is greater, thus the only matches between std::lower_bound and std::upper_bound would be ones exactly matching pathRelativeToArchive
				// by appending char with code 1 and taking the lower bound, we can find the first element with prefix greater than pathRelativeToArchive, thus the upper bound
				auto upper = std::lower_bound(lower,end,SFileList::SEntry{pathRelativeToArchive.string()+"\1"});
				trimmedList.m_range = {&(*lower),upper!=end ? &(*upper):real_end};
			}
			else
				trimmedList.m_range = {real_end,real_end};
		}
		return trimmedList;
}


core::smart_refctd_ptr<IFileArchive> IArchiveLoader::createArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password) const
{
	if (!(file->getFlags()&IFile::ECF_READ))
		return nullptr;

	return createArchive_impl(std::move(file),password);
}