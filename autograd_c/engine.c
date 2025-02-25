#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

//------------------------------
// Autograd engine definitions
//------------------------------

typedef struct Value Value;

struct Value {
    float data;
    float grad;
    void (*_backward)(Value*);
    Value** _prev;  
    char _op;
};

typedef struct {
    Value** data;   
    size_t size;     
    size_t capacity; 
} ValueArray;

//------------------------------
// Dynamic array helper functions
//------------------------------

void init_value_array(ValueArray* arr, size_t initial_capacity) {
    arr->data = (Value**)malloc(initial_capacity * sizeof(Value*));
    if (!arr->data) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    arr->size = 0;
    arr->capacity = initial_capacity;
}

void push_value_array(ValueArray* arr, Value* v) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        Value** temp = (Value**)realloc(arr->data, arr->capacity * sizeof(Value*));
        if (!temp) {
            perror("realloc failed");
            free(arr->data);
            exit(EXIT_FAILURE);
        }
        arr->data = temp;
    }
    arr->data[arr->size++] = v;
}

bool is_in_array(ValueArray* arr, Value* v) {
    for (size_t i = 0; i < arr->size; i++) {
        if (arr->data[i] == v) {
            return true;
        }
    }
    return false;
}

//------------------------------
// Value creation and operations
//------------------------------

Value* create_value(float data) {
    Value* new_value = (Value*)malloc(sizeof(Value));
    if (!new_value) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    new_value->data = data;
    new_value->grad = 0;
    new_value->_backward = NULL;
    new_value->_prev = NULL;
    new_value->_op = '\0';
    return new_value;
}

void add_backward(Value* self) {
    if (self->_prev != NULL) {
        self->_prev[0]->grad += self->grad;
        self->_prev[1]->grad += self->grad;
    }
}

void mul_backward(Value* self) {
    if (self->_prev != NULL) {
        self->_prev[0]->grad += self->_prev[1]->data * self->grad;
        self->_prev[1]->grad += self->_prev[0]->data * self->grad;
    }
}

Value* add(Value* v1, Value* v2) {
    float data = v1->data + v2->data;
    Value* new_value = create_value(data);

    new_value->_prev = (Value**)malloc(2 * sizeof(Value*));
    if (!new_value->_prev) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    new_value->_prev[0] = v1;
    new_value->_prev[1] = v2;
    new_value->_op = '+';
    new_value->grad = 0;
    new_value->_backward = add_backward;
    return new_value;
}

Value* mul(Value* v1, Value* v2) {
    float data = v1->data * v2->data;
    Value* new_value = create_value(data);

    new_value->_prev = (Value**)malloc(2 * sizeof(Value*));
    if (!new_value->_prev) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    new_value->_prev[0] = v1;
    new_value->_prev[1] = v2;
    new_value->_op = '*';
    new_value->grad = 0;
    new_value->_backward = mul_backward;
    return new_value;
}

//------------------------------
// Topological sort
//------------------------------

/*
   The build_top function performs a depth-first search on the computation graph.
   It uses the dynamic arrays "visited" to track nodes we have already seen and
   "topo" to store the topological ordering.
*/
void build_top(Value* v, ValueArray* topo, ValueArray* visited) {
    if (v == NULL)
        return;

    if (is_in_array(visited, v))
        return;
    
    push_value_array(visited, v);
    
    if (v->_prev != NULL) {
        build_top(v->_prev[0], topo, visited);
        build_top(v->_prev[1], topo, visited);
    }
    push_value_array(topo, v);
}

ValueArray* topological_sort(Value* v) {
    ValueArray* topo = (ValueArray*)malloc(sizeof(ValueArray));
    ValueArray* visited = (ValueArray*)malloc(sizeof(ValueArray));
    if (!topo || !visited) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    init_value_array(topo, 10);
    init_value_array(visited, 10);

    build_top(v, topo, visited);

    free(visited->data);
    free(visited);
    return topo;
}

//------------------------------
// Backward pass
//------------------------------

/*
   The backward function recursively calls _backward on each node.
   (In the topologically sorted order, you typically iterate in reverse order.)
*/

void backward(Value* v) {
    v->grad = 1.0;

    ValueArray* topo = topological_sort(v);

    for (int i = (int)topo->size - 1; i >= 0; i--) {
        Value* node = topo->data[i];
        if (node->_backward != NULL) {
            node->_backward(node);
        }
    }

    free(topo->data);
    free(topo);
}
