#pragma once
#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include <assert.h>
#include "Log.h"

#include <windows.h>
#include <stdio.h>

// allocat memory on the sherd (device and host) space
template<class T>
void MemAllocSherd(T **devPtr, size_t size) {
	cudaError err = cudaMallocManaged(devPtr, size);
	if (err != cudaSuccess) {
		LOG(ERR) << "MemAllocSherd: " << err << "size: " << size << "\n";
	}
}

template<class T>
void MemFreeSherd(T* devPtr) {
	cudaFree(devPtr);
}

std::string cmd(char* cmd);

enum Cell : char {
	UNKNOWN	   	 = 0b00000000,
	BLACK		 = 0b00000001,
	WHITE		 = 0b00000010,
	BOTH		 = 0b00000011,
	UNIQUE_BLACK = 0b00000100,
	UNIQUE_WHITE = 0b00000101,
	ALL		= 0b01111111
};

void printLine(const Cell* line, const int len);

// creates a array of heigth x width on the shered memory
class Grid {
public:
	int heigth, width;
	Grid() {}
	Grid(const Grid& other);
	Grid& operator=(const Grid& other);
	Grid(int height, int width);
	void Set(const Grid& other);
	~Grid();
	void Read(Cell* buffer, bool isRow, int index)const;
	void ReadRow(Cell* buffer, int index) const;
	void ReadCol(Cell* buffer, int index) const;
	int Size()const;
	void Setup(int height, int width);
	const Cell& at(int row, int col)const;
	const Cell* at(int row) const;
	Cell& at(int row, int col);
	Cell* at(int row);

private:
	Cell* data;
};

template<class T>
class customArray {
public:
	customArray(const int size=1);
	~customArray();
	__host__ void clear();
	__host__ void resize(int newSize);
	__host__ void setCapacity(int cap);
	__host__ int push_back(T val);
	__host__ T* getRaw();
	__host__ __device__ T& operator[] (const int index);
	__host__ __device__ int Size()const;
	__host__ __device__ int getCapacity()const;
private:
	T* arr;
	int capacity;
	int size;
};




template<class T>
customArray<T>::customArray(const int size) : capacity(size)
{
	MemAllocSherd(&arr, sizeof(T) * capacity);
	this->size = 0;
}

template<class T>
customArray<T>::~customArray()
{
	MemFreeSherd(arr);
}

template<class T>
__host__ void customArray<T>::clear()
{
	size = 0;
}

template<class T>
__host__ void customArray<T>::resize(int newSize)
{
	assert(newSize < capacity);
	size = newSize;
}

template<class T>
__host__ void customArray<T>::setCapacity(int cap)
{
	assert(capacity < cap);
	T* temp = arr;
	MemAllocSherd(&arr, sizeof(T)*cap);

	cudaMemcpy(arr, temp, sizeof(T)*capacity, cudaMemcpyDeviceToDevice);
	capacity = cap;
	MemFreeSherd(temp);
}

template<class T>
__host__ int customArray<T>::push_back(T val)
{
	assert(size < capacity);
	arr[size] = val;
	return size++;
}

template<class T>
__host__ T * customArray<T>::getRaw()
{
	return arr;
}

template<class T>
__host__ __device__ T & customArray<T>::operator[](const int index)
{
	assert(index < size);
	return arr[index];
}

template<class T>
__host__ __device__ int customArray<T>::Size() const
{
	return size;
}

template<class T>
__host__ __device__ int customArray<T>::getCapacity() const
{
	return capacity;
}
