set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-error")

add_subdirectory(Analysis)
add_subdirectory(CodeGen)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Transforms)
