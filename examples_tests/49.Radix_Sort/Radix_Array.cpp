#include "Radix_Array.h"
#include "Vec2D.h"

void Radix_Sort::Radix_Array::Deallocate_Resource() noexcept
{
	if (!Data_Vec2D)
	{
		return;
	}
	Data_Size = 0;
	delete[] Data_Vec2D;
}

Radix_Sort::Radix_Array::Radix_Array(const std::size_t Array_Size) :
	Data_Size(Array_Size),
	Data_Vec2D(new Vec2D[Data_Size])
{
	//for(const auto& elem : Data_Vec2D)
	//{
	//	//elem.static_cast<uint32_t>(Data_Size - Index(elem));
	//}

	for (std::size_t i = 0; i < Data_Size; ++i)
	{
		Data_Vec2D[i].Set_Key(static_cast<uint32_t>(Data_Size - i));
	}

	Index(Data_Vec2D);
	int a = 0;
}

Radix_Sort::Radix_Array::Radix_Array(const Radix_Array& Object) :
	Data_Size(Object.Data_Size),
	Data_Vec2D(new Vec2D[Data_Size])
{
	for (std::size_t i = 0; i < Data_Size; ++i)
	{
		Data_Vec2D[i] = Object.Data_Vec2D[i];
	}
}

Radix_Sort::Radix_Array::Radix_Array(Radix_Array&& Object) noexcept:
	Data_Size(std::exchange(Object.Data_Size, {})),
	Data_Vec2D(std::exchange(Object.Data_Vec2D, nullptr))
{

}

std::size_t Radix_Sort::Radix_Array::Index(const Vec2D* const Elem) const noexcept
{
	if(!Data_Vec2D)
	{
		static_assert("Invalid elem index\n");
		return -1;
	}


	//const std::size_t index = begin() - end() - Elem;
	
	//return static_cast<std::size_t>();
	return 1;
}

Radix_Sort::Vec2D* Radix_Sort::Radix_Array::begin() const noexcept
{
	return Data_Vec2D ? Data_Vec2D : nullptr;
}

Radix_Sort::Vec2D* Radix_Sort::Radix_Array::end() const noexcept
{
	return Data_Vec2D ? (Data_Vec2D + (Data_Size - 1u)) : nullptr;
}

Radix_Sort::Radix_Array& Radix_Sort::Radix_Array::operator=(const Radix_Array& Object)
{
	if(this != &Object)
	{
		Deallocate_Resource();
		Data_Size = Object.Data_Size;
		Data_Vec2D = new Vec2D[Data_Size];

		for (std::size_t i = 0; i < Data_Size; ++i)
		{
			Data_Vec2D[i] = Object.Data_Vec2D[i];
		}
	}
	return *this;
}

Radix_Sort::Radix_Array& Radix_Sort::Radix_Array::operator=(Radix_Array&& Object) noexcept
{
	if(this != &Object)
	{
		Data_Size = std::exchange(Object.Data_Size, {});
		Data_Vec2D = std::exchange(Object.Data_Vec2D, nullptr);
	}
	return *this;
}

Radix_Sort::Radix_Array::~Radix_Array()
{
	Deallocate_Resource();
}