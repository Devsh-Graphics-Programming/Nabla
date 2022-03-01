#ifndef _NBL_SYSTEM_C_FILE_VIEW_H_
#define _NBL_SYSTEM_C_FILE_VIEW_H_


#include <nbl/system/IFile.h>


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
		// TODO: do we even need to keep a smartpointer back to the ISystem!?
		IFileView(core::smart_refctd_ptr<ISystem>&& sys, const path& _name, const core::bitflag<E_CREATE_FLAGS> _flags, void* buffer, size_t fileSize) :
			IFile(std::move(sys),_name,_flags|ECF_COHERENT), m_buffer((std::byte*)buffer), m_size(fileSize)
		{
		}

		size_t read_impl(void* buffer, size_t offset, size_t sizeToRead) override final
		{
			if (offset+sizeToRead > m_size)
				sizeToRead = m_size-offset;
			memcpy(buffer, m_buffer+offset, sizeToRead);
			return sizeToRead;
		}
		size_t write_impl(const void* buffer, size_t offset, size_t sizeToWrite) override final
		{
			if (offset+sizeToWrite > m_size)
				sizeToWrite = m_size-offset;
			memcpy(m_buffer+offset, buffer, sizeToWrite);
			return sizeToWrite;
		}

		const void* getMappedPointer_impl() const override final
		{
			return m_buffer;
		}
		void* getMappedPointer_impl() override final
		{
			return m_buffer;
		}

	private:
		std::byte* m_buffer;
		size_t m_size;
};

template<typename allocator_t>
class CFileView : public IFileView
{
		// what do I even need this "friend" for?
		friend class CAPKResourcesArchive;

	public:
#if 0
		CFileView(CFileView<allocator_t>&& other) : IFile(std::move(other.m_system), path(other.getFileName()), other.m_flags), m_size(other.m_size), m_buffer(other.m_buffer) 
		{
			other.m_buffer = nullptr;
		}
#endif
		// constructor for making a file with memory already allocated by the allocator
		CFileView(core::smart_refctd_ptr<ISystem>&& sys, const path& _name, core::bitflag<E_CREATE_FLAGS> _flags, void* buffer, size_t fileSize, allocator_t&& _allocator) :
			IFileView(std::move(sys),_name,_flags,buffer,fileSize), allocator(std::move(_allocator)) {}

		// 
		static inline core::smart_refctd_ptr<CFileView<allocator_t>> create(core::smart_refctd_ptr<ISystem>&& sys, const path& _name, core::bitflag<E_CREATE_FLAGS> _flags, size_t fileSize, allocator_t&& _allocator={})
		{
			auto mem = reintepret_cast<std::byte*>(_allocator.alloc(fileSize));
			if (!mem)
				return nullptr;
			auto retval = new CFileView(std::move(sys),_name,_flags,mem,fileSize,std::move(_allocator));
			return core::smart_refctd_ptr(retval,core::dont_grab);
		}

	protected:
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
#if 0
		CFileView(CFileView<CNullAllocator>&& other) :
			IFile(std::move(other.m_system),path(other.getFileName()),other.m_flags|ECF_COHERENT),
			m_size(other.m_size), m_buffer(other.m_buffer)
		{
			other.m_buffer = nullptr;
		}
#endif
		CFileView(core::smart_refctd_ptr<ISystem>&& sys, const path& _name, core::bitflag<E_CREATE_FLAGS> _flags, void* buffer, size_t fileSize) :
			IFileView(std::move(sys),_name,_flags,buffer,fileSize) {}

	protected:
		~CFileView() = default;
};

}

#endif