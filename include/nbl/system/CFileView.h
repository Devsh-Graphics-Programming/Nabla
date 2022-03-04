#ifndef _NBL_SYSTEM_C_FILE_VIEW_H_
#define _NBL_SYSTEM_C_FILE_VIEW_H_


#include "nbl/system/IFile.h"


namespace nbl::system
{

class IFileView : public IFile
{
	public:
		size_t getSize() const override final
		{
			return m_size;
		}

	protected:
		IFileView(IFileView&& other) : IFile(path(other.getFileName()),other.getFlags()), m_buffer(other.m_buffer), m_size(other.m_size)
		{
			other.m_buffer = nullptr;
		}
		IFileView(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, void* buffer, size_t fileSize) :
			IFile(std::move(_name),_flags), m_buffer(buffer), m_size(fileSize) {}

		const void* getMappedPointer_impl() const override final
		{
			return m_buffer;
		}
		void* getMappedPointer_impl() override final
		{
			return m_buffer;
		}

		//
		void* m_buffer;
		size_t m_size;
};

template<typename allocator_t>
class CFileView : public IFileView
{
		// what do I even need this "friend" for?
		friend class CAPKResourcesArchive;

	public:
		// constructor for making a file with memory already allocated by the allocator
		CFileView(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, void* buffer, const size_t fileSize, allocator_t&& _allocator) :
			IFileView(std::move(sys),std::move(_name),_flags,buffer,fileSize), allocator(std::move(_allocator)) {}

		// 
		static inline core::smart_refctd_ptr<CFileView<allocator_t>> create(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, size_t fileSize, allocator_t&& _allocator={})
		{
			auto mem = reintepret_cast<std::byte*>(_allocator.alloc(fileSize));
			if (!mem)
				return nullptr;
			auto retval = new CFileView(std::move(sys),_name,_flags,mem,fileSize,std::move(_allocator));
			return core::smart_refctd_ptr(retval,core::dont_grab);
		}

	protected:
		CFileView(CFileView<allocator_t>&& other) : IFileView(std::move(other)), allocator(std::move(other.allocator)) {}
		~CFileView()
		{
			if (m_buffer)
				allocator.dealloc(m_buffer, m_size);
		}

		allocator_t allocator;
};


// Forward declare the null file allocator callback
class CNullAllocator;

//
template<>
class CFileView<CNullAllocator> : public IFileView
{
	public:
		CFileView(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, void* buffer, const size_t fileSize) :
			IFileView(std::move(_name),_flags,buffer,fileSize) {}

	protected:
		CFileView(CFileView<CNullAllocator>&& other) : IFileView(std::move(other)) {}
		~CFileView() = default;
};

}

#endif