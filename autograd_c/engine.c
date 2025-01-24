#include <stdio.h> 
#include <stdlib.h>

// started creating an autograd engine finally!
typedef struct Value Value;

struct Value{
    float data;
    float grad;
    void (*_backward)(Value*);
    Value** _prev;
    char _op;
};

Value* create_value(float data) {
    Value* new_value = (Value*)malloc(sizeof(Value));
    new_value->data = data;
    new_value->grad = 0;
    new_value->_backward = NULL;
    new_value->_prev = NULL;
    new_value->_op = '\0';
    return new_value;
}

void add_backward(Value* self) {
        self->_prev[0]->grad += self->grad;
        self->_prev[1]->grad += self->grad;
}

void mul_backward(Value* self) {
        self->_prev[0]->grad += self->_prev[1]->data * self->grad;
        self->_prev[1]->grad += self->_prev[0]->data * self->grad;
}

Value* add(Value* v1, Value* v2) {
    int data = v1->data + v2->data;
    Value* new_value = create_value(data);

    new_value->_prev = (Value**)malloc(2*sizeof(Value*));
    new_value->_prev[0] = v1;
    new_value->_prev[1] = v2;
    new_value->_op = '+';
    new_value->grad = 0;
    new_value->_backward = add_backward;
    return new_value;
}

Value* mul(Value* v1, Value* v2) {
    int data = v1->data * v2->data;
    Value* new_value = create_value(data);

    new_value->_prev = (Value**)malloc(2*sizeof(Value*));
    new_value->_prev[0] = v1;
    new_value->_prev[1] = v2;
    new_value->_op = '*';
    new_value->grad = 0;
    new_value->_backward = mul_backward;
    return new_value;
}

// void backward(Value* v){
//     // topological sort of the children in the graph
//     Value** topo[] = {};
//     // make a visited set
//     Value** visited[] = {};

//     void build_top(Value* v); {
//         if (v == NULL){
//             return;
//         }
//         if (v in visited){
//             return;
//         }
//         visited.push(v);
//         for (int i = 0; i < v->_prev.length; i++){
//             build_top(v->_prev[i]);
//         }
//         topo.push(v);
//     }


// }

// test struct
// int main(){
//     Value* v1 = create_value(3);
//     Value* v2 = create_value(4);

//     printf("v1.data: %f\n", v1->data);
//     printf("v1.grad: %f\n", v1->grad);

//     printf("v2.data: %f\n", v2->data);
//     printf("v2.grad: %f\n", v2->grad);

//     Value* v3 = add(v1, v2);
//     Value* v4 = mul(v1, v3);

//     printf("v3.data: %f\n", v3->data);
//     printf("v4.data: %f\n", v4->data);

//     // backward(v3);
//     // backward(v4);

//     printf("Printing the gradients after backward pass\n");
//     printf("v1.grad: %f\n", v1->grad);
//     printf("v2.grad: %f\n", v2->grad);
//     printf("v3.grad: %f\n", v3->grad);

//     free(v3->_prev);
//     free(v4->_prev);
//     free(v3);
//     free(v4);
//     free(v1);
//     free(v2);

//     return 0;
// }