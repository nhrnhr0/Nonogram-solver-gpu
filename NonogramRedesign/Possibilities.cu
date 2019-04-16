#include "Possibilities.cuh"
#include <iostream>
#include "Timer.h"
#include "LargeNumber.h"
#include <numeric>
#include "InfInt.h"

Possibilities::Possibilities(LineDescription ld): desc(std::move(ld)), loaded(false)
{
	std::cout << "poss ctor: ";
	for (int i = 0; i < getRules().size(); i++)
		std::cout << getRules().at(i);
	std::cout << "\t\t";
	
	LoadAllPossibilities();
	std::cout << getPossSize();
	std::cout << std::endl;
	
	
	/*char syms[] = { ' ', 'X', '-' };
	for (int i = 0; i < possSize; i++) {
		std::cout << i + 1 << ")\t";
		for (int j = 0; j < ld.first; j++) {
			std::cout << syms[getLinePtr(possSize - i - 1)[j]];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/
}

std::string Possibilities::toString() const {
	std::string ret = getPossSize() + ") ";
	for (int i = 0; i < getRules().size(); i++) {
		ret += getRules().at(i) + " ";
	}
	return ret;
}

_DEV_HOST_ const Cell * Possibilities::getRawPossibilities() const
{
	return possibilities;
}

_DEV_HOST_ Cell * Possibilities::getRawPossibilities()
{
	return possibilities;
}

Possibilities::Possibilities(const Possibilities & other) :
	desc(other.desc)
{
	std::cout << "possibilities copy constractor" << std::endl;
}

Possibilities::~Possibilities()
{
	MemFreeSherd(possibilities);
}

_DEV_HOST_ Cell * Possibilities::getLinePtr(int lineIndex)
{
	return &possibilities[lineIndex * getLineLen()];
}

_DEV_HOST_ const Cell * Possibilities::getLinePtr(int lineIndex) const
{
	return &possibilities[lineIndex * getLineLen()];
}

const std::vector<int>& Possibilities::getRules() const
{
	return desc.second;
}

_DEV_HOST_ int Possibilities::getLineLen() const
{
	return desc.first;
}

_DEV_HOST_ int Possibilities::getPossSize() const
{
	return possSize;
}

_DEV_HOST_ Cell * Possibilities::getLinePtr(Cell * raw, int lineIndex, int lineSize)
{
	return &raw[lineIndex * lineSize];
}

_DEV_HOST_ const Cell * Possibilities::getLinePtr(const Cell * raw, int lineIndex, int lineSize)
{
	return &raw[lineIndex * lineSize];
}

void Possibilities::LoadAllPossibilities()
{
	Timer::start("possibi");
	possSize = calcPossSize();
	if (possSize > MAX_POSSIBILITIES_TO_CALCULATE) {
		loaded = false;
		Timer::abort("possibi");
	}
	else {
		int numInserted = 0;
		MemAllocSherd(&possibilities, sizeof(Cell)*possSize*getLineLen());
		tempMemory = new Cell[sizeof(Cell) * possSize * getLineLen()];
		fillPossibilities(0, getRules().begin(), numInserted, std::vector<Cell>(getLineLen(), Cell::WHITE));
		cudaMemcpy(possibilities, tempMemory, sizeof(Cell)*possSize*getLineLen(), cudaMemcpyHostToDevice);
		loaded = true;
		delete tempMemory;
		Timer::stop("possibi");
	}
	 
}

int binomialCoeff(int n, int k) {
	// Base Cases 
	if (k == 0 || k == n)
		return 1;
	// Recur 
	return  binomialCoeff(n - 1, k - 1) + binomialCoeff(n - 1, k);
}

int Possibilities::calcPossSize()
{
	int res;
	const int ruleSum = std::accumulate(getRules().begin(), getRules().end(), 0);
	const int W = getLineLen() - ruleSum - getRules().size() + 1;
	/*{
		Timer::start("LargeNumber");
		LargeNumber n1 = LargeNumber::Factorial(getRules().size() + W);
		LargeNumber n2 = LargeNumber::Factorial(getRules().size());
		LargeNumber n3 = LargeNumber::Factorial(W);
		n2.Multiply(n3);
		res1 = n1.Divide(n2);
		Timer::stop("LargeNumber");
	}*/
	Timer::start("pos size");
	res = binomialCoeff(getRules().size() + W, W);
	Timer::stop("pos size");

	//assert(res1 == res2);
	return res;
}

bool Possibilities::isLoaded() const
{
	return loaded;
}

void Possibilities::fillPossibilities(
	int startIndex,
	std::vector<int>::const_iterator currRule,
	int & insertLineIndex,
	std::vector<Cell> currLine) 
{
	if (currRule == getRules().end()) {
		memcpy(tempMemory + insertLineIndex * getLineLen(),
			currLine.data(), sizeof(Cell)*getLineLen());
		insertLineIndex++;
		return;
	}

	const int dist = std::distance(currRule, getRules().end()) - 1;
	const int sum = std::accumulate(currRule, getRules().end(), 0);
	if (startIndex + dist + sum > getLineLen())
		return;

	if (startIndex + dist + sum + 1 <= getLineLen())
		fillPossibilities(startIndex + 1, currRule, insertLineIndex, currLine);
	paint(currLine, startIndex, *currRule);
	fillPossibilities(startIndex + *currRule + 1, currRule + 1, insertLineIndex, currLine);
}

void Possibilities::paint(std::vector<Cell>& line, int startIndex, int rule)
{
	for (int i = startIndex; i < startIndex + rule; i++) {
		line.at(i) = Cell::BLACK;
	}
}
