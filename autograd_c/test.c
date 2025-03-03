#include "engine.c"
#include "nn.c"

int main(){
    Value* v1 = create_value(3);
    Value* v2 = create_value(4);
    Value* v3 = add(v1, v2);
    Value* v4 = mul(v1, v2);
    Value* v5 = add(v3, v4);

    printf("v1.data: %f\n", v1->data);
    printf("v2.data: %f\n", v2->data);
    printf("v3.data: %f\n", v3->data);
    printf("v4.data: %f\n", v4->data);
    printf("v5.data: %f\n", v5->data);

    // for (int i = 0; i < 5; i++){
    //     printf("%f ", topo_sort[i]->data);
    // }
    // printf("\n");

    backward(v5);
    printf("v1.grad: %f\n", v1->grad);
    printf("v2.grad: %f\n", v2->grad);
    printf("v3.grad: %f\n", v3->grad);
    printf("v4.grad: %f\n", v4->grad);
    printf("v5.grad: %f\n", v5->grad);

    free(v1);
    free(v2);
    free(v3);
    free(v4);

    Neuron* n = create_neuron(10);
    ValueArray* inputs = (ValueArray*)malloc(sizeof(ValueArray));
    if (!inputs) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    init_value_array(inputs, 10);
    for (int i = 0; i < 10; i++) {
        push_value_array(inputs, create_value(i));
    }

    forward_neuron(n, inputs);

    ValueArray* weights = n->weight;
    for (int i = 0; i < weights->size; i++){
        printf("%f ", weights->data[i]->data);
    }
    printf("\n");

    printf("Neuron created: %ld\n", n->weight->size);
    printf("Neuron output: %f\n", n->output->data);
    printf("Neuron bias: %f\n", n->bias->data);

    return 0;
}
