if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-error")

add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Transforms)

add_llvm_library(FIROptimizer
  CodeGen.cpp
  IteratedDominanceFrontier.cpp
  StdConverter.cpp
)

target_link_libraries(FIROptimizer FIRSupport)

install (TARGETS FIROptimizer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
