// 测试 #pragma learn print 功能
// 原始测试文件

void test_without_identifier() {
#pragma learn print
}

void test_with_identifier() {
#pragma learn print abc
}

void test_with_different_identifier() {
#pragma learn print test_variable
}

int main() {
  printf("=== Testing #pragma learn print ===\n\n");

  test_without_identifier();
  test_with_identifier();
  test_with_different_identifier();

  printf("\n=== Test complete ===\n");
  return 0;
}
