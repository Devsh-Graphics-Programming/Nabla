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

Radix_Sort::Radix_Array::Radix_Array():
	Data_Size(10),
	Data_Vec2D(new Vec2D[Data_Size])
{
	
}

Radix_Sort::Radix_Array::Radix_Array(const std::size_t Array_Size) :
	Data_Size(Array_Size),
	Data_Vec2D(new Vec2D[Data_Size])
{
	
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

//Radix_Sort::Vec2D* Radix_Sort::Radix_Array::begin() const noexcept
//{
//	return Data_Vec2D ? Data_Vec2D : nullptr;
//}
//
//Radix_Sort::Vec2D* Radix_Sort::Radix_Array::end() const noexcept
//{
//	return Data_Vec2D ? (Data_Vec2D + (Data_Size - 1u)) : nullptr;
//}

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