// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_MEMORY_READ_FILE_H_INCLUDED__
#define __C_MEMORY_READ_FILE_H_INCLUDED__

#include "IReadFile.h"
#include "IWriteFile.h"
#include "irr/core/Types.h"
#include "irr/core/memory/new_delete.h"
#include "irr/core/alloc/null_allocator.h"
#include "coreutil.h"

namespace irr
{

namespace io
{

	/*!
		Class for reading and writing from memory.
	*/
	class CMemoryFile
	{
	    protected:
            //! changes position in file, returns true if successful
            inline bool seek(const size_t& finalPos, bool relativeMovement = false)
            {
                if (relativeMovement)
                {
                    if (Pos + finalPos > Buffer.size())
                        return false;

                    Pos += finalPos;
                }
                else
                {
                    if (finalPos > Buffer.size())
                        return false;

                    Pos = finalPos;
                }

                return true;
            }

            //! returns size of file
            inline size_t getSize() const {return Buffer.size();}

            //! returns where in the file we are.
            inline size_t getPos() const {return Pos;}

            //! returns name of file
            inline const io::path& getFileName() const {return Filename;}

            //! Constructor
            CMemoryFile(const size_t& len, const io::path& fileName);

            //! Destructor
            virtual ~CMemoryFile();


            core::vector<uint8_t> Buffer;
            size_t      Pos;
        private:
            io::path    Filename;
	};

	class CMemoryWriteFile : public IWriteFile, public CMemoryFile
	{
        public:
            //! Constructor
            CMemoryWriteFile(const size_t& len, const io::path& fileName);

            //! changes position in file, returns true if successful
            virtual bool seek(const size_t& finalPos, bool relativeMovement = false);

            //! returns size of file
            virtual size_t getSize() const {return CMemoryFile::getSize();}

            //! returns where in the file we are.
            virtual size_t getPos() const {return CMemoryFile::getPos();}

            //! returns name of file
            virtual const io::path& getFileName() const {return CMemoryFile::getFileName();}

            //! returns how much was written
            virtual int32_t write(const void* buffer, uint32_t sizeToWrite);

            inline void* getPointer() { return Buffer.data(); }
	};


    template<
        typename Alloc = _IRR_DEFAULT_ALLOCATOR_METATYPE<uint8_t>,
        bool = std::is_same<Alloc, core::null_allocator<typename Alloc::value_type>>::value
    >
    class CCustomAllocatorMemoryReadFile;


    template<typename Alloc>
    class CCustomAllocatorMemoryReadFile<Alloc, true> : public IReadFile
    {
        static_assert(sizeof(typename Alloc::value_type)==1, "Alloc::value_type must be of size 1");

    protected:
        virtual ~CCustomAllocatorMemoryReadFile ()
        {
            m_allocator.deallocate(reinterpret_cast<typename Alloc::pointer>(m_storage), m_length);
        }

    public:
        CCustomAllocatorMemoryReadFile(void* _data, size_t _length, const io::path& _filename, core::adopt_memory_t, Alloc&& _alloc = Alloc()) :
            m_storage{_data}, m_length{_length}, m_position{0u}, m_filename{_filename}, m_allocator{std::move(_alloc)}
        {
        }

        virtual bool seek(const size_t& finalPos, bool relativeMovement = false) override
        {
            if (relativeMovement)
            {
                if (m_position + finalPos > m_length)
                    return false;
                m_position += finalPos;
            }
            else
            {
                if (finalPos > m_length)
                    return false;
                m_position = finalPos;
            }
            return true;
        }

        virtual size_t getSize() const override { return m_length; }

        virtual size_t getPos() const override { return m_position; }

        virtual const io::path& getFileName() const override { return m_filename; }

        virtual int32_t read(void* buffer, uint32_t sizeToRead) override
        {
            int64_t amount = static_cast<int64_t>(sizeToRead);
            if (m_position + amount > getSize())
                amount -= m_position + amount - m_length;

            if (amount <= 0ll)
                return 0;

            memcpy(buffer, reinterpret_cast<uint8_t*>(m_storage)+m_position, amount);

            m_position += amount;

            return static_cast<int32_t>(amount);
        }

    protected:
        void* m_storage;
        size_t m_length;
        size_t m_position;
        io::path m_filename;
        Alloc m_allocator;
    };

    template<typename Alloc>
    class CCustomAllocatorMemoryReadFile<Alloc, false> : public CCustomAllocatorMemoryReadFile<Alloc, true>
    {
        using Base = CCustomAllocatorMemoryReadFile<Alloc, true>;
    protected:
        virtual ~CCustomAllocatorMemoryReadFile() = default;

    public:
        using Base::Base;

        CCustomAllocatorMemoryReadFile(const void* _data, size_t _length, const io::path& _filename, Alloc&& _alloc = Alloc()) :
            Base(const_cast<void*>(_data), _length, _filename, core::adopt_memory, std::move(_alloc))
        {
            const void* tmp = Base::m_storage;
            Base::m_storage = Base::m_allocator.allocate(Base::m_length);
            memcpy(Base::m_storage, tmp, Base::m_length);
        }
    };

    class CMemoryReadFile : public CCustomAllocatorMemoryReadFile<>
    {
        using Base = CCustomAllocatorMemoryReadFile<>;

    protected:
        virtual ~CMemoryReadFile() = default;

    public:
        CMemoryReadFile(void* _data, size_t _length, const io::path& _filename, core::adopt_memory_t) :
            Base(_data, _length, _filename, core::adopt_memory)
        {
        }
        CMemoryReadFile(const void* _data, size_t _length, const io::path& _filename) :
            Base(_data, _length, _filename)
        {
        }
    };

} // end namespace io
} // end namespace irr

#endif

