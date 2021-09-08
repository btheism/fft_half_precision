#include <filesystem>
#include <iostream>
int main(int argc, char *argv[]) {
  std::filesystem::path p{argv[1]};

  std::cout << "The size of " << p.u8string() << " is " <<
      std::filesystem::file_size(p) << " bytes.\n";
}

