#ifndef _NBL_SYSTEM_C_ARCHIVE_LOADER_TAR_H_INCLUDED_
#define _NBL_SYSTEM_C_ARCHIVE_LOADER_TAR_H_INCLUDED_


#include "nbl/system/CFileArchive.h"


namespace nbl::system
{

class CArchiveLoaderTar final : public IArchiveLoader
{
	public:
		class CArchive : public CFileArchive
		{
			public:
				CArchive(core::smart_refctd_ptr<IFile>&& _file, system::logger_opt_smart_ptr&& logger, core::vector<SListEntry>&& _items) :
					CFileArchive(path(_file->getFileName()),std::move(logger),std::move(_items)), m_file(std::move(_file)) {}

				//
			protected:
				core::smart_refctd_ptr<IFile> m_file;
		};

		CArchiveLoaderTar(system::logger_opt_smart_ptr&& logger) : IArchiveLoader(std::move(logger)) {}
		virtual bool isALoadableFileFormat(IFile* file) const override
		{
			return !!createArchive_impl(core::smart_refctd_ptr<IFile>(file),"");
		}

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "tar", nullptr };
			return ext;
		}

	private:
		core::smart_refctd_ptr<IFileArchive> createArchive_impl(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const override;
};

}
#endif