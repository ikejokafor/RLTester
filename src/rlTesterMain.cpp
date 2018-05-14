#include "RL.hpp"
#include "FixedPoint.hpp"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
	
	int numStates = 16;
	int numActionsPerState = 3;
	
	bool *validActionsPerState = (bool*)malloc(sizeof(bool) * numStates * numActionsPerState);	
	// 1 frac bit
	index2D(numStates, numActionsPerState, validActionsPerState, 0, 0) = false;		 // dec frac bits
	index2D(numStates, numActionsPerState, validActionsPerState, 0, 1) = true;      // inc frac bits	
	index2D(numStates, numActionsPerState, validActionsPerState, 0, 2) = true;       // do nothing	
	for(int i = 1 ; i < (numStates - 1); i++) {
		index2D(numStates, numActionsPerState, validActionsPerState, i, 0) = true;   // dec frac bits
		index2D(numStates, numActionsPerState, validActionsPerState, i, 1) = true;    // inc frac bits
		index2D(numStates, numActionsPerState, validActionsPerState, i, 2) = true;        // do nothing	
	}
	// 8 frac bits
	index2D(numStates, numActionsPerState, validActionsPerState, (numStates - 1), 0) = true;   // dec frac bits
	index2D(numStates, numActionsPerState, validActionsPerState, (numStates - 1), 1) = false;    // inc frac bits	
	index2D(numStates, numActionsPerState, validActionsPerState, (numStates - 1), 2) = true;         // do nothing	
	
	
	int *transitionMatrix = (int*)malloc(numStates * numActionsPerState * sizeof(int));
	for (int i = 0; i < numStates; i++) {
		if (index2D(numStates, numActionsPerState, validActionsPerState, i, 0)) {
			index2D(numStates, numActionsPerState, transitionMatrix, i, 0) = i - 1;
		} else {
			index2D(numStates, numActionsPerState, transitionMatrix, i, 0) = -1;
		}
		
		if (index2D(numStates, numActionsPerState, validActionsPerState, i, 1)) {
			index2D(numStates, numActionsPerState, transitionMatrix, i, 1) = i + 1;
		} else {
			index2D(numStates, numActionsPerState, transitionMatrix, i, 1) = -1;
		}
		
		if (index2D(numStates, numActionsPerState, validActionsPerState, i, 2)) {
			index2D(numStates, numActionsPerState, transitionMatrix, i, 2) = i;
		} else {
			index2D(numStates, numActionsPerState, transitionMatrix, i, 2) = -1;
		}
	}
	
	for (int i = 0; i < numStates; i++){
		cout << "state:" << " " << i << " " << "numFracBits:" << " " << (i + 1) << " " << index2D(numStates, numActionsPerState, transitionMatrix, i, 0) << " " << index2D(numStates, numActionsPerState, transitionMatrix, i, 1) << " " << index2D(numStates, numActionsPerState, transitionMatrix, i, 2) << endl;
	}
	cout << endl;
	
	
	QLearner agent(numStates, numActionsPerState, validActionsPerState, transitionMatrix, 0.0f);
	
	float fl_a = 3.14f;
	float fl_b = 1.23f;
	float fl_c = fl_a * fl_b;
	int length = 32;
	int numFracBits = 16;
	int numIntBits = length - numFracBits;
	
	FixedPoint_t fx_a;
	FixedPoint_t fx_b;
	FixedPoint_t fx_c;

	int action;
	float fl_c_dout;
	float reward;
	int numEpisodes = 100000;
	int doutNumFracBits;
	
	
	//length++;
	//numFracBits++;
	//for (int i = 0; i < 8; i++) {
	//		length--;
	//		numFracBits--;
	//	
	//	
	//	fx_a = FixedPoint::create(numFracBits, 3.14f);
	//	fx_b = FixedPoint::create(numFracBits, 1.23f);
	//	doutNumFracBits = numFracBits * 2;
	//	fx_c = fx_a * fx_b;
	//	fl_c_dout = FixedPoint::toFloat(doutNumFracBits, fx_c);
	//
	//	reward = (1.0f - (fabsf(fl_c - fl_c_dout) / fl_c)) + float(length * length);		
	//	cout << reward << endl;
	//	
	//
	//}
	//exit(0);
	
	
	for(int i = (numStates - 1) ; i >= 0; i--) {
		agent.m_currentState = i;
		numFracBits = i + 1;
		length = numFracBits + numIntBits;
		for(int j = 0; j < numEpisodes; j++) {
			action = agent.GetNextAction();
		
			if(action == 0) {
				length--;
				numFracBits--;
			} else if(action == 1){
				length++;
				numFracBits++;
			} 
			doutNumFracBits = numFracBits * 2;
		
			fx_a = FixedPoint::create(numFracBits, 3.14f);
			fx_b = FixedPoint::create(numFracBits, 1.23f);	
			fx_c = fx_a * fx_b;
			fl_c_dout = FixedPoint::toFloat(doutNumFracBits, fx_c);

			reward = ((1.0f - (fabsf(fl_c - fl_c_dout) / fl_c)) + float(length));			
			agent.UpdateQTable(reward);
		
			//agent.printQMatrix();
			int a = 0;
		}
		//agent.printQMatrix();
		int a = 0;
	}
	agent.printQMatrix();
	int a = 0;
		
	return 0;
}