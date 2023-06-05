#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Simple neural network that can learn XOR
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

double init_wights() {return ((double)rand()) / ((double)RAND_MAX); }

void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i< n-1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - 1) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
            
        }
    }
}

#define numInputs 2
#define numHiddedNodes 2
#define numOutputs 1
#define numTrainingSets 4


int main(void) {
    
    const double lr = 0.1f;
    
    double hiddedLayer[numHiddedNodes];
    double outputLayer[numOutputs];
    
    double hiddedLayerBias[numHiddedNodes];
    double outputLayerBias[numOutputs];
    
    double hiddedWeights[numInputs][numHiddedNodes];
    double outputWeights[numHiddedNodes][numOutputs];
    
    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                          {1.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 1.0f}};

    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};

    for ( int i = 0; i < numInputs; i++) {
        
        for (int j = 0; j < numHiddedNodes ; j++) {
            
            //associate with random values
            hiddedWeights[i][j] = init_wights();
        }
    }

    for ( int i = 0; i < numHiddedNodes; i++) {
        
        for (int j = 0; j < numOutputs ; j++) {
            
            //associate with random values
            outputWeights[i][j] = init_wights();
        }
    }
    
    for ( int i = 0; i < numOutputs; i++) {
        
        outputLayerBias[i] = init_wights();
    }
    
    int trainingSetOrder[] = {0,1,2,3};
    
    int numberOfEpochs = 80000; 
    
    // Train the neural network for a numner of epochs
    for(int epochs = 0; epochs < numberOfEpochs; epochs++) {
        
        shuffle(trainingSetOrder, numTrainingSets);
        
        for(int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];
            
            // Forward pass
            
            // Computer hidden layer activation
            
            for( int j = 0; j < numHiddedNodes; j++) {
                double activation = hiddedLayerBias[j];
                for(int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddedWeights[k][j];
                }
                
                hiddedLayer[j] = sigmoid(activation);
                
            }
            
            for( int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for(int k = 0; k < numHiddedNodes; k++) {
                    activation += hiddedLayer[k] * outputWeights[k][j];
                }
                
                outputLayer[j] = sigmoid(activation);
                
                
            }
            
            printf("\nInput:%g  %g  Output: %g    Predicted Output:  %g \n",
                training_inputs[i][0], training_inputs[i][1],
                outputLayer[0], training_outputs[i][0]);
                
            // Backpropagation
            
            // Computer change in output weights
            
            double deltaOutput[numOutputs]; 
            
            for(int j = 0; j < numOutputs; j++)  {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }
            
            // Compute change in hidden weights
            double deltaHidden[numHiddedNodes];
            for(int j = 0; j < numHiddedNodes; j++) {
                double error = 0.0f;
                for(int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddedLayer[j]);
            }
            
            // Apply change in output weights
            for(int j= 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for(int k = 0; k < numHiddedNodes; k++) {
                    outputWeights[k][j] += hiddedLayer[k] * deltaOutput[j] * lr;
                }
                
            }
            
            // Apply change in hidden weights
            for(int j= 0; j < numHiddedNodes; j++) {
                hiddedLayerBias[j] += deltaHidden[j] * lr;
                for(int k = 0; k < numInputs; k++) {
                    hiddedWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
                
            }
            
            
        }
        

    }
    // Print Final weights adter done training
        
    fputs("Final Hidden Weights\n[ ", stdout);
    for (int j = 0; j < numHiddedNodes; j++) {
        fputs ("[ ", stdout); 
        for(int k = 0; k < numInputs; k++) {
            printf("%f ", hiddedWeights[k][j]);
        }
        fputs("] ", stdout);
    }
    
    fputs ( "]\nFinal Hidden Biases\n[ ", stdout); 
    for(int j = 0; j < numHiddedNodes; j++) {
        printf("%f ", hiddedLayerBias[j]);
    }
    
    fputs ( "]\nFinal Output Biases\n[ ", stdout); 
    for(int j = 0; j < numOutputs; j++) {
        printf("%f ", outputLayerBias[j]);
    }
    
	return 0;
}
