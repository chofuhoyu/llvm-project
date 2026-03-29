// 测试 C++ 模式下的 #pragma learn print 功能
// 步骤1: 使用 -fsyntax-only -Xclang -fixit 修改文件（不编译）
//   clang++ -fsyntax-only -Xclang -fixit test_pragma_print.cpp
// 步骤2: 正常编译修改后的文件
//   clang++ test_pragma_print.cpp -o test_pragma_print_cpp

#include <cstdio>

void test_cpp_without_identifier() {
#pragma learn print
}

void test_cpp_with_identifier() {
#pragma learn print cpp_test
}

int main() {
  test_cpp_without_identifier();
  test_cpp_with_identifier();
  return 0;
}
