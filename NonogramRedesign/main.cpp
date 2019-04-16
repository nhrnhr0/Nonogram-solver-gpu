#pragma once

#include "Utils.cuh"
#include "Nonogram.h"
#include "LargeNumber.h"
#include "Timer.h"
#include "Possibilities.cuh"
#include "main.h"

int main() {
	/*std::vector<int> rules = { 1,7,3,2,2,2,2,2,1,1,1 };
	LineDescription ld = std::make_pair(40, ref(rules));
	Possibilities ps(ld);
	Line l(ps, 0, true);
	Grid g(40, 40);
	l.Filter(g);
	Cell* res = l.Unify();
	printLine(res, 40);
	g.at(0, 7) = Cell::BLACK;
	g.at(0, 8) = Cell::BLACK;
	g.at(0, 16) = Cell::BLACK;
	g.at(0, 17) = Cell::BLACK;
	g.at(0, 18) = Cell::WHITE;
	g.at(0, 19) = Cell::BLACK;
	g.at(0, 20) = Cell::BLACK;
	g.at(0, 21) = Cell::WHITE;
	l.Filter(g);
	res = l.Unify();
	printLine(res, 40);
	system("pause");
	return 0;*/
	Timer::start("all");
	Nonogram n;
	n.Load(cmd("python nonogramHtml2Json.py"));
	n.Solve();

	std::cout << "done line solver" << std::endl;
	Timer::stop("all");
	Timer::printAll();
	system("pause");
	//Grid grid;
	/*grid.Setup(4, 15);
	for (int i = 0; i < grid.heigth; i++) {
		auto row = grid.at(i);
		for (int j = 0; j < grid.width; j++) {
			row[j] = Cell::BOTH;
			std::cout << grid.at(i, j);
		}
		std::cout << "\n";
	}*/
	return 0;
}