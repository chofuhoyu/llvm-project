// Test file for #pragma learn functionality
#include <stdio.h>

void test_warn_directive() {
    printf("Testing #pragma learn warn\n");
    
    // This should generate a warning
    #pragma learn warn
    
    printf("After #pragma learn warn\n");
}

void test_print_directive() {
    printf("Testing #pragma learn print\n");
    
    // This should mark where to insert printf
    #pragma learn print
    
    printf("After #pragma learn print\n");
}

void test_unknown_directive() {
    printf("Testing unknown directive\n");
    
    // This should generate an error
    #pragma learn unknown
    
    printf("After unknown directive\n");
}

void test_multiple_directives() {
    printf("Testing multiple directives\n");
    
    #pragma learn warn
    #pragma learn print
    
    printf("After multiple directives\n");
}

int main() {
    printf("=== Testing #pragma learn ===\n\n");
    
    test_warn_directive();
    printf("\n");
    
    test_print_directive();
    printf("\n");
    
    test_multiple_directives();
    printf("\n");
    
    printf("=== Test complete ===\n");
    return 0;
}
