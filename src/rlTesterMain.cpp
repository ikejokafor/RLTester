#include "RL.hpp"
#include "FixedPoint.hpp"


int main(int argc, char **argv) {
	
	int numStates = 3;
	bool *validStates = (bool*)malloc(sizeof(bool) * numStates * numStates);
	index2D(3, 3, validStates, 0, 0) = false;
	index2D(3, 3, validStates, 0, 1) = true;
	index2D(3, 3, validStates, 0, 2) = true;
	index2D(3, 3, validStates, 1, 0) = false;
	index2D(3, 3, validStates, 1, 1) = true;
	index2D(3, 3, validStates, 1, 2) = true;	
	index2D(3, 3, validStates, 2, 0) = false;
	index2D(3, 3, validStates, 2, 1) = true;
	index2D(3, 3, validStates, 2, 2) = true;
	
	QLearner agent(3, validStates, 0.9f, 3.4f, 0.2f, NULL);
	
	float fl_a = 3.14f;
	float fl_b = 1.23f;
	float fl_c = fl_a * fl_b;
	int length = 16;
	int numFracBits = 8;
	float upperLimit = fl_c;
	float lowerLimit = fl_c - (fl_c * 0.05f);

	
	FixedPoint_t fx_a;
	FixedPoint_t fx_b;
	FixedPoint_t fx_c;

	int action;
	float fl_c_dout;
	float reward;
	
	while(true) {
		action = agent.GetNextAction();
		
		if(action == 1) {
			length--;
			numFracBits--;
		} else if(action == 2){
			length++;
			numFracBits++;
		}
		
		
		fx_a = FixedPoint::create(numFracBits, 3.14f);
		fx_b = FixedPoint::create(numFracBits, 1.23f);
		fx_c = fx_a * fx_b;
		FixedPoint_t fl_fx_c_dout = fx_c;
		FixedPoint::SetParam((length * 2), (numFracBits * 2), length, numFracBits, fl_fx_c_dout);
		fl_c_dout = FixedPoint::toFloat(numFracBits, fl_fx_c_dout);
		if (fl_c_dout >= lowerLimit && fl_c_dout <= upperLimit) {
			reward = 1.0f / length;
		} else {
			reward = -1.0f;
		}
		agent.UpdateQTable(reward);
		
	}	
		
	return 0;
}