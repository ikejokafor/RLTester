#include "RL.hpp"
#include "FixedPoint.hpp"
#include <iostream>
#include <fstream>
#include <algorithm> 


using namespace std;


typedef struct stateTuple {
	// state params
	int param0_numFracBits;
	int param1_numFracBits;
	int param2_numFracBits;
	int param3_numFracBits;
	int param4_numFracBits;
	int param5_numFracBits;
	int param6_numFracBits;
	int param7_numFracBits;
	int param8_numFracBits;	
	// state matrix encoding
	int stateEncoding;
} stateTuple_t;


typedef struct filter {
	float filter0;
	float filter1;
	float filter2;
	float filter3;
	float filter4;
	float filter5;
	float filter6;
	float filter7;
	float filter8;
} filter_t;


typedef struct input {
	float input0;
	float input1;
	float input2;
	float input3;
	float input4;
	float input5;
	float input6;
	float input7;
	float input8;
} input_t;


typedef struct filter_fxPt {
	FixedPoint_t filter0;
	FixedPoint_t filter1;
	FixedPoint_t filter2;
	FixedPoint_t filter3;
	FixedPoint_t filter4;
	FixedPoint_t filter5;
	FixedPoint_t filter6;
	FixedPoint_t filter7;
	FixedPoint_t filter8;
} filter_fxPt_t;



typedef struct input_fxPt {
	FixedPoint_t input0;
	FixedPoint_t input1;
	FixedPoint_t input2;
	FixedPoint_t input3;
	FixedPoint_t input4;
	FixedPoint_t input5;
	FixedPoint_t input6;
	FixedPoint_t input7;
	FixedPoint_t input8;
} input_fxPt_t;


FixedPoint_t doConvolution(stateTuple_t stateTuple0, stateTuple_t stateTuple1, filter_t filter, input_t input, FixedPoint_t &fxPtRes) {
	input_fxPt_t input_fxPt;
	input_fxPt.input0 = FixedPoint::create(stateTuple0.param0_numFracBits, input.input0);
	input_fxPt.input1 = FixedPoint::create(stateTuple0.param1_numFracBits, input.input1);
	input_fxPt.input2 = FixedPoint::create(stateTuple0.param2_numFracBits, input.input2);
	input_fxPt.input3 = FixedPoint::create(stateTuple0.param3_numFracBits, input.input3);
	input_fxPt.input4 = FixedPoint::create(stateTuple0.param4_numFracBits, input.input4);
	input_fxPt.input5 = FixedPoint::create(stateTuple0.param5_numFracBits, input.input5);
	input_fxPt.input6 = FixedPoint::create(stateTuple0.param6_numFracBits, input.input6);
	input_fxPt.input7 = FixedPoint::create(stateTuple0.param7_numFracBits, input.input7);
	input_fxPt.input8 = FixedPoint::create(stateTuple0.param8_numFracBits, input.input8);

	
	filter_fxPt filter_fxPt;
	filter_fxPt.filter0 = FixedPoint::create(stateTuple1.param0_numFracBits, filter.filter0);
	filter_fxPt.filter1 = FixedPoint::create(stateTuple1.param1_numFracBits, filter.filter1);
	filter_fxPt.filter2 = FixedPoint::create(stateTuple1.param2_numFracBits, filter.filter2);
	filter_fxPt.filter3 = FixedPoint::create(stateTuple1.param3_numFracBits, filter.filter3);
	filter_fxPt.filter4 = FixedPoint::create(stateTuple1.param4_numFracBits, filter.filter4);
	filter_fxPt.filter5 = FixedPoint::create(stateTuple1.param5_numFracBits, filter.filter5);
	filter_fxPt.filter6 = FixedPoint::create(stateTuple1.param6_numFracBits, filter.filter6);
	filter_fxPt.filter7 = FixedPoint::create(stateTuple1.param7_numFracBits, filter.filter7);
	filter_fxPt.filter8 = FixedPoint::create(stateTuple1.param8_numFracBits, filter.filter8);
	
	
	return  (	filter_fxPt.filter0 * input_fxPt.input0 
					+ filter_fxPt.filter1 * input_fxPt.input1
					+ filter_fxPt.filter2 * input_fxPt.input2
					+ filter_fxPt.filter3 * input_fxPt.input3
					+ filter_fxPt.filter4 * input_fxPt.input4
					+ filter_fxPt.filter5 * input_fxPt.input5
					+ filter_fxPt.filter6 * input_fxPt.input6
					+ filter_fxPt.filter7 * input_fxPt.input7
					+ filter_fxPt.filter8 * input_fxPt.input8
			);
}


float doConvolution(filter_t filter, input_t input) {
	return (	filter.filter0 * input.input0 
					+ filter.filter1 * input.input1
					+ filter.filter2 * input.input2
					+ filter.filter3 * input.input3
					+ filter.filter4 * input.input4
					+ filter.filter5 * input.input5
					+ filter.filter6 * input.input6
					+ filter.filter7 * input.input7
					+ filter.filter8 * input.input8
			);
}


float getReward(stateTuple_t *stateTuple, 
				float fl_a, 
				float fl_b, 
				int currentState, 
				int action, 
				int nextState, 
				float errorThresh,
				float maxPrec,
				float penalty,
				float penaltyFactor,
				float currStateBias,
				float nextStateBias,
				bool &positiveReward) {
	float currentStateLength = stateTuple[currentState].param0_numFracBits + stateTuple[currentState].param1_numFracBits;
	float nextStateLength = stateTuple[nextState].param0_numFracBits + stateTuple[nextState].param1_numFracBits;
	FixedPoint_t fx_curr_a = FixedPoint::create(stateTuple[currentState].param0_numFracBits, fl_a);
	FixedPoint_t fx_curr_b = FixedPoint::create(stateTuple[currentState].param1_numFracBits, fl_b);
	FixedPoint_t fx_curr_res = fx_curr_a * fx_curr_b;
	FixedPoint_t fx_next_a = FixedPoint::create(stateTuple[nextState].param0_numFracBits, fl_a);
	FixedPoint_t fx_next_b = FixedPoint::create(stateTuple[nextState].param1_numFracBits, fl_b);
	FixedPoint_t fx_next_res = fx_next_a * fx_next_b;	
	float fl_res = fl_a * fl_b;
	float fl_curr_res_dout = FixedPoint::toFloat((stateTuple[currentState].param0_numFracBits + stateTuple[currentState].param1_numFracBits), fx_curr_res);
	float fl_next_res_dout = FixedPoint::toFloat((stateTuple[nextState].param0_numFracBits + stateTuple[nextState].param1_numFracBits), fx_next_res);
	float currStateErrNorm = 1.0f - (fabsf(fl_res - fl_curr_res_dout) / fl_res);
	float nextStateErrNorm = 1.0f - (fabsf(fl_res - fl_next_res_dout) / fl_res);
	float nextStateErr = (fabsf(fl_res - fl_next_res_dout) / fl_res);
	float errNormDiff = fabsf(currStateErrNorm - nextStateErrNorm);
	float lengthDiffNorm = maxPrec - fabsf(currentStateLength - nextStateLength);
	
	cout << "[REWARD] Current State Error Normalized is" << " " << currStateErrNorm << "." << endl;
	cout << "[REWARD] Current State Length is" << " " << currentStateLength << "." << endl;
	cout << "[REWARD] Next State Error Normalized is" << " " << nextStateErrNorm << "." << endl;	
	cout << "[REWARD] Next State Length is" << " " << nextStateLength << "." << endl;
		
	float reward = 0.0f;				
	if (nextStateErr <= errorThresh && nextState == currentState) {
		reward = currStateErrNorm;
		positiveReward = true;
	} else if (nextStateErr <= errorThresh && nextStateErrNorm >= currStateErrNorm && nextStateLength < currentStateLength) {	// best case
		reward = currStateErrNorm + (errNormDiff + 1.0f) / lengthDiffNorm;
		positiveReward = true;	
	} else if (nextStateErr <= errorThresh && nextStateErrNorm < currStateErrNorm && nextStateLength > currentStateLength) {	// worst case
		reward = currStateErrNorm - errNormDiff / lengthDiffNorm;
		positiveReward = false;
	} else if (nextStateErr <= errorThresh) {
		reward = currStateErrNorm + (errNormDiff + 1.0f) / lengthDiffNorm;
		if(nextStateLength > currentStateLength) {
			positiveReward = false;
		} else {
			positiveReward = true;
		}
	} else {
		reward = -FLT_MAX;
		positiveReward = false;
	}
					
	
	return reward;
}


stateTuple_t *createStateSpace(int numStates, int numActionsPerState, bool **validActionsPerState, int **transitionMatrix, int minNumFracBits, int maxNumFracBits) {
	
	(*validActionsPerState) = (bool*)malloc(numStates * numActionsPerState * sizeof(bool));
	memset((*validActionsPerState), 1, numStates * numActionsPerState * sizeof(bool));
	(*transitionMatrix) = (int*)malloc(numStates * numActionsPerState * sizeof(int));

	
	stateTuple_t *stateTuple = (stateTuple_t*)malloc(numStates * sizeof(stateTuple_t));
	for (int i = 0, a = minNumFracBits, b = minNumFracBits, k = 0; i < numStates; i++, k++) {
		stateTuple[i].stateEncoding = k;
		stateTuple[i].param0_numFracBits = a;
		stateTuple[i].param1_numFracBits = b;
		if (b == maxNumFracBits) {
			b = minNumFracBits;
			a++;			
		} else {
			b++;
		}
	}
	
	
	for (int i = 0; i < numStates; i++) {		
		for (int j = 0, k = 0; j < numActionsPerState; j++, k++) {
			index2D(numStates, numActionsPerState, (*transitionMatrix), i, j) = k;
		}
	}
	
	
	return stateTuple;
}


int main(int argc, char **argv) {
	// BEGIN CODE -----------------------------------------------------------------------------------------------------------------------------------
	srand(time(NULL));
	ofstream fd;	
	int maxNumFracBits = 28;
	int minNumFracBits = 8;
	int numStates = ((maxNumFracBits - minNumFracBits) + 1) * ((maxNumFracBits - minNumFracBits) + 1);	
	int numActionsPerState = numStates;
	bool *validActionsPerState;
	int *transitionMatrix;
	stateTuple_t *stateTuple = createStateSpace(numStates, numActionsPerState, &validActionsPerState, &transitionMatrix, minNumFracBits, maxNumFracBits);
	int numEpochs =  numStates * numStates;
	float epsilon = 1.0f;
	float epsilonDecayFactor = 1.0f / (numStates * numStates);
	float discountFactor = 1.0f;
	float learningRate = 1.0f;
	QLearner agent(numStates, numActionsPerState, validActionsPerState, transitionMatrix, epsilon, epsilonDecayFactor, discountFactor, learningRate);
	// END CODE -------------------------------------------------------------------------------------------------------------------------------------

	
	// BEGIN CODE -----------------------------------------------------------------------------------------------------------------------------------
	// fd.open("file0.csv");
	// fd << "State Encoding" << "," << "param0_numFracBits" << "," << "param1_numFracBits" << endl;
	// for(int i = 0; i < numStates; i++) {
	// 	fd << stateTuple[i].stateEncoding << "," << stateTuple[i].param0_numFracBits << "," << stateTuple[i].param1_numFracBits << endl;
	// }
	// fd.close();
	// fd.open("file1.csv");
	// fd << "," << endl;
	// for (int i = 0; i < numStates; i++) {
	// 	fd << "State" << " " << i;
	// 	for (int j = 0, k = 0; j < numActionsPerState; j++) {
	// 		fd << "," << index2D(numStates, numActionsPerState, transitionMatrix, i, j);
	// 	}
	// 	fd << endl;
	// }
	// fd.close();
	// END CODE -------------------------------------------------------------------------------------------------------------------------------------
	
	
	// BEGIN CODE -----------------------------------------------------------------------------------------------------------------------------------
	vector<int> states;
	// set some values:
	for(int i = 0 ; i < numStates; i++) {
		states.push_back(i); 
	}
	// using built-in random generator:
	std::random_shuffle(states.begin(), states.end());
	// END CODE -------------------------------------------------------------------------------------------------------------------------------------

	
	// BEGIN CODE -----------------------------------------------------------------------------------------------------------------------------------
	int action;
	int nextState;
	float reward;
	bool positiveReward;
	float penalty = -2.0f;
	float penaltyFactor = 2.0f;
	float errorThresh = 0.0002f;
	float fl_a = 6.6f;
	float fl_b = 9.5f;	
	float currStateBias = (float)rand() / (float)RAND_MAX;
	float nextStateBias = (float)rand() / (float)RAND_MAX;
	fd.open("file2.csv");
	fd << "Epoch" << "," << "CurrentBitWidth" << "," << "param0_numFracBits" << "," << "param1_numFracBits" << endl;
	agent.m_currentState = numStates - 1;
	fd	<< 0
		<< "," 
		<< stateTuple[agent.m_currentState].param0_numFracBits + stateTuple[agent.m_currentState].param1_numFracBits 
		<< "," 
		<< stateTuple[agent.m_currentState].param0_numFracBits
		<< ","
		<< stateTuple[agent.m_currentState].param1_numFracBits << endl;
	
	cout << "[INIT] Initial state" << " " << agent.m_currentState << "." << endl;
	cout << "[INIT] Value is" << " " << fl_a * fl_b << "." << endl << endl;
	for (int j = 0; j < numEpochs; j++) {
		cout << "[INFO] Epoch" << " " << j << endl;
		action = agent.GetNextAction();
		nextState = index2D(numStates, numActionsPerState, transitionMatrix, agent.m_currentState, action);		
		reward = getReward(	stateTuple, 
							fl_a, 
							fl_b, 
							agent.m_currentState, 
							action, 
							nextState, 
							errorThresh,
							(maxNumFracBits + maxNumFracBits),
							penalty, 
							penaltyFactor,
							currStateBias,
							nextStateBias,
							positiveReward);
		agent.UpdateQTable(reward);
		cout << "[REWARD] Reward for state:" << " " << agent.m_currentState << ", and action:" << " " << action << " " << "is" << " " << reward << "." << endl;
		if(positiveReward == true) {
			cout << "[REWARD] Reward was positive so transition to new state." << endl;
			agent.NextState();
		}
		fd	<< (j + 1) 
			<< "," 
			<< stateTuple[agent.m_currentState].param0_numFracBits + stateTuple[agent.m_currentState].param1_numFracBits 
			<< "," 
			<< stateTuple[agent.m_currentState].param0_numFracBits
			<< ","
			<< stateTuple[agent.m_currentState].param1_numFracBits << endl;
	}
	fd.close();	
	// END CODE -------------------------------------------------------------------------------------------------------------------------------------

	
	// BEGIN CODE -----------------------------------------------------------------------------------------------------------------------------------
	FixedPoint_t fx_curr_a = FixedPoint::create(stateTuple[agent.m_currentState].param0_numFracBits, fl_a);
	FixedPoint_t fx_curr_b = FixedPoint::create(stateTuple[agent.m_currentState].param1_numFracBits, fl_b);
	FixedPoint_t fx_curr_res = fx_curr_a * fx_curr_b;
	float fl_res = fl_a * fl_b;
	float fl_curr_res_dout = FixedPoint::toFloat((stateTuple[agent.m_currentState].param0_numFracBits + stateTuple[agent.m_currentState].param1_numFracBits), fx_curr_res);
	float currStateErr = (fabsf(fl_res - fl_curr_res_dout) / fl_res);
	cout << "[INFO] Final Error is" << 	" " << currStateErr << " " << "for parm0FracBits:" << " " << stateTuple[agent.m_currentState].param0_numFracBits << ", parm1FracBits:" << " " << stateTuple[agent.m_currentState].param1_numFracBits << "." << endl;
	// END CODE -------------------------------------------------------------------------------------------------------------------------------------

		
	return 0;
}