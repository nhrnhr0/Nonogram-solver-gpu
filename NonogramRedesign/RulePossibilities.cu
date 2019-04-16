/*#include "RulePossibilities.cuh"
#include <numeric>
#include "LargeNumber.h"
#include "Bigint.h"
#include "Timer.h"
int temp;


RulePossibilities::RulePossibilities(
	int lineSize, int lineIndex, bool isRow, const std::vector<int>& rules) :
	lineSize(lineSize), lineIndex(lineIndex), isRow(isRow), rules(rules) {
	LoadAllPossibilities();
}

RulePossibilities::~RulePossibilities()
{
}

Cell * RulePossibilities::getLinePtr(int lineIndex)
{
	return &possibilities[lineIndex * lineSize];
}

void RulePossibilities::LoadAllPossibilities()
{
	Timer::start("calcAllPossibilities");
	possSize = calcPossSize();

	MemAllocSherd(&possibilities, sizeof(Cell)*possSize*lineSize);
	
	std::cout << "rule: (";
	for (int i = 0; i < rules.size(); i++)
		std::cout << rules.at(i) << " ";
	std::cout << ") pos: " << possSize;

	temp = 0;

	int numInserted = 0;
	fillPossibilities(0,rules.begin(), numInserted, std::vector<Cell>(lineSize, Cell::WHITE));
	std::cout << " recur iteration: " << temp << " num inserted: " << numInserted << std::endl;
	Timer::stop("calcAllPossibilities");
}
void RulePossibilities::fillPossibilities(
	int startIndex, 
	std::vector<int>::const_iterator currRule,
	int& insertLineIndex,
	std::vector<Cell> currLine) {
	temp++;
	if (currRule == rules.end()) {
		//MemAllocSherd(&possibilities[insertLineIndex], sizeof(Cell) * lineSize);
		
		cudaMemcpy(getLinePtr(insertLineIndex), currLine.data(), sizeof(Cell)*lineSize, cudaMemcpyHostToDevice);
		insertLineIndex++;
		return;
	}

	const int dist = std::distance(currRule, rules.end()) - 1;
	const int sum = std::accumulate(currRule, rules.end(), 0);
	if (startIndex + dist + sum > lineSize)
		return;

	if(startIndex + dist + sum + 1 <= lineSize)
		fillPossibilities(startIndex + 1, currRule, insertLineIndex, currLine);
	paint(currLine, startIndex, *currRule);
	fillPossibilities(startIndex + *currRule + 1, currRule + 1, insertLineIndex, currLine);
}

void RulePossibilities::paint(std::vector<Cell>& line, int startIndex, int rule) {
	for (int i = startIndex; i < startIndex + rule; i++) {
		line.at(i) = Cell::BLACK;
	}
}

int RulePossibilities::calcPossSize()
{
	int res1;
	const int ruleSum = std::accumulate(rules.begin(), rules.end(), 0);
	const int W = lineSize - ruleSum - rules.size() + 1;
	{
		Timer::start("LargeNumber");
		LargeNumber n1 = LargeNumber::Factorial(rules.size() + W);
		LargeNumber n2 = LargeNumber::Factorial(rules.size());
		LargeNumber n3 = LargeNumber::Factorial(W);
		n2.Multiply(n3);
		res1 = n1.Divide(n2);
		Timer::stop("LargeNumber");
}
	//assert(res2 == res1);
	return res1;
}
*/