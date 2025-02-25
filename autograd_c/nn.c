#include "engine.c"

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
    return 0;
}
