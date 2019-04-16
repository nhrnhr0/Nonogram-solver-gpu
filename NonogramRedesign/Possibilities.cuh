#pragma once
#include <vector>
#include <iostream>
#include "Utils.cuh"

#define _DEV_HOST_ __device__ __host__

//#define MAX_POSSIBILITIES_TO_CALCULATE 10'000'000'000
#define MAX_POSSIBILITIES_TO_CALCULATE 5'000'000

// LineDescription is the rules of the line and the length of the line
typedef std::pair<int, std::vector<int>&> LineDescription;

class Possibilities {
public:
	Possibilities(LineDescription ld);
	_DEV_HOST_ const Cell* getRawPossibilities()const;
	_DEV_HOST_       Cell* getRawPossibilities();
	~Possibilities();
	_DEV_HOST_       Cell* getLinePtr(int lineIndex);
	_DEV_HOST_ const Cell* getLinePtr(int lineIndex) const;
	const std::vector<int>& getRules() const;
	_DEV_HOST_ int getLineLen() const;
	_DEV_HOST_ int getPossSize() const;

	_DEV_HOST_ static       Cell* getLinePtr(Cell* raw, int lineIndex, int lineSize);
	_DEV_HOST_ static const Cell* getLinePtr(const Cell* raw, int lineIndex, int lineSize);
	bool isLoaded()const;
	std::string toString() const;
private:
	Possibilities(const Possibilities& other);
	int possSize;
	Cell* possibilities; // [possibilitiesSize][lineSize]
	const LineDescription desc;
	void LoadAllPossibilities();
	int calcPossSize();
	bool loaded;
	void fillPossibilities(
		int startIndex,
		std::vector<int>::const_iterator currRule,
		int& insertLineIndex,
		std::vector<Cell> currLine);
	void paint(std::vector<Cell>& line, int startIndex, int rule);
	Cell* tempMemory;
};

