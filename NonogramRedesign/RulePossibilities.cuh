/*#include <vector>
#include "Utils.cuh"

#define MAX_POSSIBILITIES_TO_CALCULATE 10000

class RulePossibilities {
public:
	RulePossibilities(int lineSize, int lineIndex, bool isRow, const std::vector<int>& rules);
	~RulePossibilities();
private:
	Cell* possibilities; // [possibilitiesSize][lineSize]
	int lineSize, lineIndex;
	int possSize;
	bool isRow;
	const std::vector<int>& rules;

	Cell* getLinePtr(int lineIndex);
	void LoadAllPossibilities();
	int calcPossSize();
	void fillPossibilities(
		int startIndex,
		std::vector<int>::const_iterator currRule,
		int& insertLineIndex,
		std::vector<Cell> currLine);
	void paint(std::vector<Cell>& line, int startIndex, int rule);

};


*/