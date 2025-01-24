#include "engine.c"

int main(){
    Value* v1 = create_value(3);
    Value* v2 = create_value(4);
    Value* v3 = add(v1, v2);
    Value* v4 = mul(v1, v2);

    printf("v1.data: %f\n", v1->data);
    printf("v2.data: %f\n", v2->data);
    printf("v3.data: %f\n", v3->data);
    printf("v4.data: %f\n", v4->data);

    free(v1);
    free(v2);
    free(v3);
    free(v4);
    return 0;
}
