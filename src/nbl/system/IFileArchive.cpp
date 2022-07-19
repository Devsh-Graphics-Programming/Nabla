#include "nbl/system/IFileArchive.h"


using namespace nbl;
using namespace nbl::system;


core::SRange<const IFileArchive::SListEntry> IFileArchive::listAssets(const path& asset_path) const
{
	// TODO: use something from ISystem for this?
	constexpr auto isSubDir = [](path p, path root) -> bool
	{
		while (p != path())
		{
			if (p==root)
				return true;
			p = p.parent_path();
		}
		return false;
	};

	const IFileArchive::SListEntry* begin = nullptr;
	const IFileArchive::SListEntry* end = nullptr;
	for (auto& entry : m_items)
	{
		if (isSubDir(entry.pathRelativeToArchive, asset_path))
		{
			if (begin)
				end = &entry;
			else
				begin = &entry;
		}
		else if (end)
			break;
	}
	return {begin,end};

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