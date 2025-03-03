// #include "engine.c"
#include <stdlib.h>
#include <time.h>


struct Module {
    ValueArray parameters;
    ValueArray gradients;
    ValueArray outputs;
};

typedef struct {
    ValueArray* weight;
    Value* bias;
    Value* output;
    int input_size;
} Neuron;

struct Linear {
    Value* weight;
    Value* bias;
    Value* output;
    int input_size;
    int output_size;
};

Neuron* create_neuron(int input_size) {
    srand(time(NULL));
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (!neuron) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    neuron->weight = (ValueArray*)malloc(sizeof(ValueArray*));
    if (!&neuron->weight) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }

    init_value_array(neuron->weight, input_size);
    for (int i = 0; i < input_size; i++) {
        // initialize weights randomly between -1 and 1
        push_value_array(neuron->weight, create_value(((float)rand() / RAND_MAX) * 2 - 1));
    };

    neuron->bias = create_value((float)rand() / RAND_MAX);
    neuron->input_size = input_size;
    return neuron;
}

void forward_neuron(Neuron* neuron, ValueArray* inputs) {
    float sum = 0;
    for (int i = 0; i < neuron->input_size; i++) {
        sum += neuron->weight->data[i]->data * inputs->data[i]->data;
    }
    sum += neuron->bias->data;
    neuron->output = create_value(sum);
}

