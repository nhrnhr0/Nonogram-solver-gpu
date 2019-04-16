#pragma once
#include <string>
#include <istream>
#include "Utils.cuh"
#include <vector>
#include "RulePossibilities.cuh"
#include "Possibilities.cuh"
#include <map>
#include "Line.cuh"
#include <queue>

typedef Line* LineTask;

class Nonogram
{
public:
	Nonogram();
	void printBoard()const;
	void Solve();
	void Load(std::string jsonString);
	int LineSolver();
	bool Probing();
	Grid& getGrid();
	~Nonogram();

private:
	Grid grid;
	std::vector<std::vector<int>> colRules;
	std::vector<std::vector<int>> rowRules;
	int lastProbX = 0;
	int lastProbY = 0;

	std::vector<Line> rowLines;
	std::vector<Line> colLines;

	int longestRowRule=-1, longestColRule=-1;
	
	std::queue<LineTask> tasks;
	std::map<LineDescription, Possibilities> possibilitiesMap;

	bool Nonogram::Prob(int i, int j);
	Grid tempGrid;
};

