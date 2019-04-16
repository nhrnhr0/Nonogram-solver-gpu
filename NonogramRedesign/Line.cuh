#pragma once
#include "Possibilities.cuh"
#include "Utils.cuh"
#include <vector>
#include "cuda.h"
#include "Timer.h"
#include "thrust/device_vector.h"

class Line {
public:
	Line(const Possibilities& possibilities, int lineIndex, bool isRow);
	Line(const Line& other);
	~Line();
	
	void cpuFilter(const Grid& gridRef);
	void Filter(const Grid& gridRef);
	Cell* cpuUnify();
	Cell* Unify();
	bool isLoaded()const;
	int getLineLen()const;
	int getLineIndex()const;
	void resetValids();
	const bool isRow;
	std::string toString()const;

	

private:
	Cell* tempLine;
	const Possibilities& possibilities;
	bool* validPossibilities;
	customArray<int> validPossIndexs;
	int lineIndex;
	void init();

	Cell* tempGridRef;
	
};

