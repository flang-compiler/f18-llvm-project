add_llvm_library(FIRSupport
  InternalNames.cpp
  KindMapping.cpp
)

target_link_libraries(FIRSupport FIRDialect)

install (TARGETS FIRSupport
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
