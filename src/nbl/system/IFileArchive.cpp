#include "nbl/system/IFileArchive.h"

#include "nbl/system/IFile.h"

using namespace nbl;
using namespace nbl::system;


IFileArchive::SFileList IFileArchive::listAssets(const path& pathRelativeToArchive) const
{
		auto trimmedList = listAssets();
		{
			auto begin = trimmedList.m_data->begin();
			auto end = trimmedList.m_data->end();
			const SFileList::SEntry itemToFind = { pathRelativeToArchive };
			trimmedList.m_range = { &(*std::lower_bound(begin,end,itemToFind)),&(*std::upper_bound(begin,end,itemToFind)) };
		}
		return trimmedList;


	/*
	// future, cause lower/upper bound don't work like that
	auto begin = std::lower_bound(m_items.begin(), m_items.end(),asset_path);
	if (begin!=m_items.end())
	{
		auto end = std::upper_bound(begin,m_items.end(),asset_path);
		if (begin==end)
			return {&(*begin),&(*end)};
	}
	return {nullptr,nullptr};
	*/
}


core::smart_refctd_ptr<IFileArchive> IArchiveLoader::createArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password) const
{
	if (!(file->getFlags()&IFile::ECF_READ))
		return nullptr;

	return createArchive_impl(std::move(file),password);
}