#include "RL.hpp"
#include "FixedPoint.hpp"
#include <iostream>
#include <fstream>

using namespace std;

typedef struct stateTuple {
	int param0_numFracBits;
	int param1_numFracBits;
	int stateEncoding;
} stateTuple_t;

// 0 param0_numFracBits up
// 1 param0_numFracBits dwn
// 2 param1_numFracBits up
// 3 param1_numFracBits dwn
// 4 param2_numFracBits up
// 5 param2_numFracBits dwn
// 6 stay

stateTuple_t *createStateSpace(int numStates, int numActionsPerState, int numFracBits, bool *validActionsPerState, int *transitionMatrix) {
	stateTuple_t *stateTuple = (stateTuple_t*)malloc(sizeof(stateTuple_t) * numStates);
	for (int i = 0; i < numStates; i++) {
		stateTuple[i].stateEncoding = -1;
	}
	
	stateTuple[0].param0_numFracBits = numFracBits;
	stateTuple[0].param1_numFracBits = numFracBits;
	stateTuple[0].stateEncoding = 0;
	
	for (int i = 0; i < numStates; i++) {		
		for (int j = 0; j < numActionsPerState; j++) {
			stateTuple_t nextState;
			nextState.param0_numFracBits = stateTuple[i].param0_numFracBits; 
			nextState.param1_numFracBits = stateTuple[i].param1_numFracBits; 
			nextState.stateEncoding		 = stateTuple[i].stateEncoding;
			// param 0 up
			if(j == 0 && stateTuple[i].param0_numFracBits < numFracBits) {
				nextState.param0_numFracBits = stateTuple[i].param0_numFracBits + 1;
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = true;
			} else if(j == 0 && stateTuple[i].param0_numFracBits == numFracBits) {
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = false;
			}
			// param 0 down
			if(j == 1 && stateTuple[i].param0_numFracBits > 2) {
				nextState.param0_numFracBits = stateTuple[i].param0_numFracBits - 1;
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = true;
			} else if(j == 1 && stateTuple[i].param0_numFracBits == 2) {
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = false;
			}
			// param 1 up
			if(j == 2 && stateTuple[i].param1_numFracBits < numFracBits) {
				nextState.param1_numFracBits = stateTuple[i].param1_numFracBits + 1;
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = true;
			} else if(j == 2 && stateTuple[i].param1_numFracBits == numFracBits) {
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = false;
			}
			// param 1 down
			if(j == 3 && stateTuple[i].param1_numFracBits > 2) {
				nextState.param1_numFracBits = stateTuple[i].param1_numFracBits - 1;
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = true;
			} else if(j == 3 && stateTuple[i].param1_numFracBits == 2) {
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = false;
			}
			// stay
			if(j == 4) {
				index2D(numStates, numActionsPerState, validActionsPerState, i, j) = true;
			}
			
			// check if next state encoding already exists
			int k;
			for (k = 0; k < numStates; k++) {
				if (stateTuple[k].param0_numFracBits == nextState.param0_numFracBits 
						&& stateTuple[k].param1_numFracBits == nextState.param1_numFracBits
						&& stateTuple[k].stateEncoding != -1) {
					index2D(numStates, numActionsPerState, transitionMatrix, i, j) = k;
					goto label0;
				}
			}
			// find a free State
			if(k == numStates) {
				for (k = 0; k < numStates; k++) {
					if (stateTuple[k].stateEncoding == -1) {
						stateTuple[k].param0_numFracBits = nextState.param0_numFracBits;
						stateTuple[k].param1_numFracBits = nextState.param1_numFracBits;
						stateTuple[k].stateEncoding = k;
						index2D(numStates, numActionsPerState, transitionMatrix, i, j) = k;
						break;
					}
				}
			}
			
			label0: continue;
		}
	}
	
	return stateTuple;
}


int main(int argc, char **argv) {
	
	int numFracBits = 8;
	int numIntBits = 8;
	int length = (numFracBits) * 2;
	
	int numStates = 49;
	int numActionsPerState = 5;
	ofstream fd;
	
	bool *validActionsPerState = (bool*)malloc(sizeof(bool) * numStates * numActionsPerState);
	int *transitionMatrix = (int*)malloc(numStates * numActionsPerState * sizeof(int));
	stateTuple_t *stateTuple = createStateSpace(numStates, numActionsPerState, numFracBits, validActionsPerState, transitionMatrix);

		
	QLearner agent(numStates, numActionsPerState, validActionsPerState, transitionMatrix, 0.0f);
	
	// print out valid actions per state
	fd.open("file1.txt");
	for (int i = 0; i < numStates; i++) {
		fd << "state: " << i << " , " << "FracBits: " << stateTuple[i].param0_numFracBits << ", " << stateTuple[i].param1_numFracBits << ", ";
		fd << "Valid Actions: ";
		for (int j = 0; j < numActionsPerState; j++) {
			fd << index2D(numStates, numActionsPerState, validActionsPerState, i, j) << " ";
		}
		fd << endl;
	}
	fd.close();
	
	// print out transition matrix
	fd.open("file2.txt");
	for (int i = 0; i < numStates; i++) {
		fd << "state: " << i << " , " << "FracBits: " << stateTuple[i].param0_numFracBits << ", " << stateTuple[i].param1_numFracBits << ", ";
		fd << "Transitions: ";	
		for(int j = 0; j < numActionsPerState; j++) {
			if (index2D(numStates, numActionsPerState, validActionsPerState, i, j)) {
				fd << index2D(numStates, numActionsPerState, transitionMatrix, i, j) << " ";
			} else {
				fd << -1 << " ";
			}
		}
		fd << endl;
	}
	fd.close();	
	
	srand(time(NULL));
	//float fl_a = (float)rand() / (float)(RAND_MAX / 5.0f);
	//float fl_b = (float)rand() / (float)(RAND_MAX / 5.0f);
	
	float fl_a = 6.5f;
	float fl_b = 9.5f;
	float fl_c = fl_a * fl_b;

	
	FixedPoint_t fx_a;
	FixedPoint_t fx_b;
	FixedPoint_t fx_c;

	int action;
	float fl_c_dout;
	float reward = 0;
	int numEpisodes = 10000;
	float errorThresh = 0.00f;
	float percentError;
	float gain;
	float gainFactor = 300.0f;

	
	int param0_numFracBits;
	int param1_numFracBits;


	int state = 0;
	agent.m_currentState = 0;
	param0_numFracBits = 8;
	param1_numFracBits = 8;				
	length = numFracBits * 2;
	

	for (int j = 0; j < numEpisodes; j++) {
		action = agent.GetNextAction();
		
		if (action == 0) {
			length++;
			param0_numFracBits++;
		} else if (action == 1) {
			length--;
			param0_numFracBits--;
		} else if (action == 2) {
			length++;
			param1_numFracBits++;
		} else if (action == 3) {
			length--;
			param1_numFracBits--;
		}

		
		fx_a = FixedPoint::create(param0_numFracBits, fl_a);
		fx_b = FixedPoint::create(param1_numFracBits, fl_b);

				
		fx_c = fx_a * fx_b;

				
		fl_c_dout = FixedPoint::toFloat((param0_numFracBits + param1_numFracBits), fx_c);

		percentError = fabsf(fl_c - fl_c_dout) / fl_c;
		gain = pow(0.5f, (percentError * gainFactor));
				
		if (percentError <= errorThresh) {
			reward = (1.0f - percentError) / float(length);
		} else {
			reward = 0;
		}
				
				
		reward = reward * gain;
		agent.UpdateQTable(reward);
	}


	
	agent.PrintQMatrix();
	
	
	for (int i = 0; i < numStates; i++) {
		action = agent.GetBestAction(state);
		state = index2D(numStates, numActionsPerState, transitionMatrix, state, action);
	}
	cout << state << endl;
	param0_numFracBits = stateTuple[state].param0_numFracBits;
	param1_numFracBits = stateTuple[state].param1_numFracBits;
	fx_a = FixedPoint::create(param0_numFracBits, fl_a);
	fx_b = FixedPoint::create(param1_numFracBits, fl_b);
	fx_c = fx_a	* fx_b;

	fl_c_dout = FixedPoint::toFloat((param0_numFracBits + param1_numFracBits), fx_c);

	cout << fl_c << endl;
	cout << fl_c_dout << endl;
	cout << param0_numFracBits << endl;
	cout << param1_numFracBits << endl;
	cout << percentError << endl;

		
	return 0;
}