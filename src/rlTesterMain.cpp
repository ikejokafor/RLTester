#include "RL.hpp"
#include "FixedPoint.hpp"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
	
	int numFracBits = 16;
	int length = numFracBits * 2;
	int numIntBits = length - numFracBits;

	int numStages = 1;
	int numStates = numFracBits * numStages;
	int numActionsPerState = 3;


	
	bool *validActionsPerState = (bool*)malloc(sizeof(bool) * numStates * numActionsPerState);	
	// 1 frac bit
	index2D(numStates, numActionsPerState, validActionsPerState, 0, 0) = false;  		 // dec frac bits
	index2D(numStates, numActionsPerState, validActionsPerState, 0, 1) = true;        // inc frac bits	
	index2D(numStates, numActionsPerState, validActionsPerState, 0, 2) = true;         // do nothing	
	for(int i = 1 ; i < (numStates - 1) ; i++) {
		index2D(numStates, numActionsPerState, validActionsPerState, i, 0) = true;     // dec frac bits
		index2D(numStates, numActionsPerState, validActionsPerState, i, 1) = true;      // inc frac bits
		index2D(numStates, numActionsPerState, validActionsPerState, i, 2) = true;          // do nothing	
	}
	// 8 frac bits
	index2D(numStates, numActionsPerState, validActionsPerState, (numStates - 1), 0) = true;     // dec frac bits
	index2D(numStates, numActionsPerState, validActionsPerState, (numStates - 1), 1) = false;      // inc frac bits	
	index2D(numStates, numActionsPerState, validActionsPerState, (numStates - 1), 2) = true;           // do nothing	
	
	
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
	
	
	QLearner agent(numStates, numActionsPerState, validActionsPerState, transitionMatrix, 0.0f);
	
	srand(time(NULL));
	float fl_a = (float)rand()/(float)(RAND_MAX/100.f);
	float fl_b = (float)rand()/(float)(RAND_MAX/100.f);
	//float fl_d = (float)rand() / (float)(RAND_MAX / 100.f);
	//float fl_c = fl_a * fl_b + fl_d;
	float fl_c = fl_a * fl_b;
	
	FixedPoint_t fx_a;
	FixedPoint_t fx_b;
	FixedPoint_t fx_c;

	int action;
	float fl_c_dout;
	float reward;
	int numEpisodes = 100000;
	int doutNumFracBits;
	float minError = 0.15f;
	float target = fl_c - fl_c * minError;

	
	//length++;
	//numFracBits++;
	//for (int i = 0; i < numStates; i++) {
	//	length--;
	//	numFracBits--;
	//	
	//	doutNumFracBits = numFracBits * 2;
	//	fx_a = FixedPoint::create(numFracBits, fl_a);
	//	fx_b = FixedPoint::create(numFracBits, fl_b);	
	//	fx_c = fx_a * fx_b;
	//	fl_c_dout = FixedPoint::toFloat(doutNumFracBits, fx_c);
	//
	//	//reward = (fabsf(target - fl_c_dout) / fmax(target, fl_c_dout));
	//	reward = fl_c_dout;
	//	cout << reward << endl;
	//
	//}
	//exit(0);
	

	int k = 0;
	while (true) {
		int state = 0;
		for (int i = (numStates - 1); i >= 0; i--) {
			agent.m_currentState = i;
			numFracBits = i + 1;
			length = numFracBits + numIntBits;
			for (int j = 0; j < numEpisodes; j++) {
				action = agent.GetNextAction();
		
				if (action == 0) {
					length--;
					numFracBits--;
				} else if (action == 1) {
					length++;
					numFracBits++;
				} 
				doutNumFracBits = numFracBits * 2;
		
				fx_a = FixedPoint::create(numFracBits, fl_a);
				fx_b = FixedPoint::create(numFracBits, fl_b);	
				fx_c = fx_a * fx_b;
				fl_c_dout = FixedPoint::toFloat(doutNumFracBits, fx_c);

				reward = 1.0f - ((fabsf(target - fl_c_dout) / fmax(target, fl_c_dout)) / float(length));
			
				agent.UpdateQTable(reward);
		
			}
		}
	
		//agent.printQMatrix();
	
	

		for (int i = 0; i < numStates; i++) {
			action = agent.GetBestAction(state);
			state = index2D(numStates, numActionsPerState, transitionMatrix, state, action);
		}
		//cout << state << endl;
		if(state != 15) {
			exit(0);
		}
		k++;
		cout << k << endl;
	}
		
	return 0;
}