#include "Nonogram.h"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <iomanip>
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>

#include "Timer.h"

Nonogram::Nonogram()
{
}

void Nonogram::printBoard()const
{
	//char syms[6] = { '_', 'X', '.', '?', '#', '*' };
	char syms[6] = { '_', '.', 'X', '?', '#', '*' };
	const int setw = 3;
	std::string result = "";
	std::stringstream ss;
	// print the col rules:
	const int colRulesOffset = (longestRowRule + 1) * setw;
	for (int i = longestColRule - 1; i >= 0; i--) {
		ss << std::string(colRulesOffset, ' ');
		for (int j = 0; j < grid.width; j++) {
			if (i < colRules[j].size()) {
				ss << std::setw(setw) << colRules[j][colRules[j].size() - 1 - i];
			}
			else {
				ss << std::setw(setw) << "X";
			}
		}
		ss << std::endl;
	}

	// print the row rules and the grid:
	for (int j = 0; j < grid.heigth; j++) {
		ss << std::string(setw, ' ');
		// print the row rules:
		for (int i = longestRowRule - 1; i >= 0; i--) {
			if (i < rowRules[j].size()) {
				ss << std::setw(setw) << rowRules[j][rowRules[j].size() - 1 - i];
			}
			else {
				ss << std::setw(setw) << "X";
			}
		}


		// print the grid:
		for (int i = 0; i < grid.width; i++) {
			//std::cout << std::setw(setw) << symbles[grid[j][i]];
			ss << std::setw(setw) << syms[grid.at(j, i)];
		}

		ss << std::endl;
	}
	system("cls");
	std::cout << ss.str();
	Sleep(5);
}

void Nonogram::Solve()
{
	int solvedSum = 0;
	bool done = false;
	while (true) {
		printBoard();
		int solved = LineSolver();
		solvedSum += solved;
		if (solvedSum == getGrid().Size()) {
			done = true;
			break;
		}
		else if(Probing()) {
			solvedSum++;
			continue;
		}
	}
	printBoard();
	std::cout << "done solveing!" << std::endl;
}

void Nonogram::Load(std::string jsonString) {
	Timer::start("jsonReading");
	rapidjson::Document d;
	d.Parse(jsonString.c_str());
	int height = d["height"].GetInt();
	int width = d["width"].GetInt();
	grid.Setup(height, width); // allocate memory for the grid


	auto ruleReader = [](const rapidjson::GenericValue<rapidjson::UTF8<>>& jsonInput,
		std::vector<std::vector<int>>& output, int& longest) {
		int i = 0;
		auto rules = jsonInput.GetArray();
		output.resize(rules.Size());
		for (auto& ruleIter : rules) {
			for (auto& ruleNum : ruleIter.GetArray()) {
				output[i].push_back(ruleNum.GetInt());
			}
			longest = longest > (int)output[i].size() ? longest : (int)output[i].size();
			i++;
		}
	};

	// read row rules:
	ruleReader(d["row"], rowRules, longestRowRule);
	ruleReader(d["col"], colRules, longestColRule);
	Timer::stop("jsonReading");

	auto& insertToMap = [](std::map<LineDescription, Possibilities>& possibilitiesMap, int lineSize, std::vector<int>& rules) {
		LineDescription ld = std::make_pair(lineSize, ref(rules));
		auto find = possibilitiesMap.find(ld);
		if (find == possibilitiesMap.end()) {
			return possibilitiesMap.emplace(ld, ld).first;
		}
		return find;
	};

	rowLines.reserve(rowRules.size());
	for (int i = 0; i < rowRules.size(); i++) {
		auto& possibilities = insertToMap(possibilitiesMap, colRules.size(), rowRules.at(i))->second;
		rowLines.emplace_back(possibilities, i, true);
		tasks.push(&rowLines.at(i));
	}
	colLines.reserve(colRules.size());
	for (int i = 0; i < colRules.size(); i++) {
		auto& possibilities = insertToMap(possibilitiesMap, rowRules.size(), colRules.at(i))->second;
		colLines.emplace_back(possibilities, i, false);
		tasks.push(&colLines.at(i));
	}
}

int Nonogram::LineSolver()
{
	//LOG(DEBUG) << "start line solver\n";
	Cell* tempLine;
	int solved = 0;
	int k = 0;
	while (tasks.empty() == false) {
		auto& task = tasks.front();
		if (task->isLoaded() == false) {
			tasks.pop();
			continue;
		}
		task->Filter(grid);
		if ((tempLine = task->Unify()) == NULL)
			return -1;
		Timer::start("add tasks");
		for (int i = 0; i < task->getLineLen();i++) {
			if (tempLine[i] != Cell::BOTH) {
				Cell& cell = task->isRow ? grid.at(task->getLineIndex(), i) : grid.at(i, task->getLineIndex());
				if (cell == Cell::UNKNOWN) {
					cell = Cell(tempLine[i] + 3);
					printBoard();
					//printBoard();
					cell = tempLine[i];
					//Sleep(250);
					//system("pause");
					solved++;
					/*LOG(DEBUG) << std::to_string(++k) << std::string(") add: ") << std::string(task->isRow ? "row" : "col")
						<< i << " tasks: " << tasks.size() << "\n";
					std::cout << "solved: " << solved << "\\" <<
						grid.heigth*grid.width << " present: " << ((float)solved / (float)(grid.heigth*grid.width)) * 100.0  << "%" << std::endl;
						*/

					tasks.push(task->isRow ? &colLines.at(i): &rowLines.at(i));
				}
			}
		}
		Timer::stop("add tasks");
		tasks.pop();
	}
	return solved;
}

bool Nonogram::Prob(int i, int j) {
	if (i < 0 || j < 0 || i >= grid.heigth || j >= grid.width)
		return false;
	auto resetNonogram = [this](Grid& orig) {
		grid.Set(orig);
		for (int i = 0; i < rowLines.size(); i++) {
			rowLines.at(i).resetValids();
		}
		for (int i = 0; i < colLines.size(); i++) {
			colLines.at(i).resetValids();
		}
	};
	Cell& cell = grid.at(i, j);
	if (cell == Cell::UNKNOWN) {
		cell = Cell::BLACK;
		tasks.push(&rowLines.at(i));
		tasks.push(&colLines.at(j));

		int solved = LineSolver();
		if (solved == -1) {
			resetNonogram(tempGrid);
			cell = Cell::UNIQUE_WHITE;
			printBoard();
			cell = Cell::WHITE;
			tasks = std::queue<LineTask>();
			tasks.push(&rowLines.at(i));
			tasks.push(&colLines.at(j));
			lastProbX = i;
			lastProbY = j;
			return true;
		}
		else {
			//TODO: score[probVal] = std::make_pair(newI, newJ);
			resetNonogram(tempGrid);
			/*
			cell = Cell::WHITE;
			tasks.push(&rowLines.at(i));
			tasks.push(&colLines.at(j));

			int solved = LineSolver();
			if (solved == -1) {
				resetNonogram(tempGrid);
				cell = Cell::UNIQUE_BLACK;
				printBoard();
				//Sleep(500);
				cell = Cell::BLACK;
				tasks = std::queue<LineTask>();
				tasks.push(&rowLines.at(i));
				tasks.push(&colLines.at(j));
				lastProbX = i;
				lastProbY = j;
				return true;
			}
			else {
				resetNonogram(tempGrid);
			}*/
		}
	}
	return false;
}

bool Nonogram::Probing()
{
	
	tempGrid = grid;

	const int neighbors[8][2] = 
	{
		{-1,-1},
		{-1, 0},
		{-1,1},
		{0,-1},
		{0,1},
		{1,-1},
		{1,0},
		{1,1}
	};
	for (int i = 0; i < 8; i++) {
		if (Prob(lastProbX + neighbors[i][0], lastProbY + neighbors[i][1]) == true)
			return true;
	}
	//TODO: std::map<int, std::pair<int,int>> score;
	for (int i = 0; i < grid.heigth; i++) {
		for (int j = 0; j < grid.width; j++) {
			int newI = (i + lastProbX) % grid.heigth;
			int newJ = (j + lastProbY) % grid.width;
			Cell& cell = grid.at(newI, newJ);
			if (Prob(newI, newJ))
				return true;
		}
	}
	printf("done probing with no result\n");
	return false;
}

Grid & Nonogram::getGrid()
{
	return grid;
}


Nonogram::~Nonogram()
{
}
